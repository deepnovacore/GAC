#!/usr/bin/env bash
# Execute generated Python only inside an isolated user/mount/network/PID
# namespace.  This is defense in depth, not a substitute for a dedicated
# container or a separate evaluation host when candidates are truly hostile.
set -euo pipefail

usage() {
    cat >&2 <<'USAGE'
Usage: sandbox_code_eval.sh --output_dir DIR --bigcode_root DIR --hf_home DIR
       [--python_bin PATH] [--extra_pythonpath DIR]
       [--hide_path PATH ...] [--benchmarks mbpp humaneval]
       [--mbpp_config full|sanitized] [--num_workers N] [--timeout SEC]

The output directory is the only writable host mount.  On a shared machine,
pass each host data root that must be hidden with --hide_path (for example,
the cluster's private workspace mount).  Do not point --output_dir at a broad
directory or at a directory containing unrelated user data.
USAGE
}

OUTPUT_DIR=""
BIGCODE_ROOT=""
HF_HOME_DIR=""
PYTHON_BIN="python3"
EXTRA_PYTHONPATH=""
HIDE_PATHS=(/home /root)
SCORER_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output_dir)       OUTPUT_DIR="$2"; shift 2 ;;
        --bigcode_root)     BIGCODE_ROOT="$2"; shift 2 ;;
        --hf_home)          HF_HOME_DIR="$2"; shift 2 ;;
        --python_bin)       PYTHON_BIN="$2"; shift 2 ;;
        --extra_pythonpath) EXTRA_PYTHONPATH="$2"; shift 2 ;;
        --hide_path)        HIDE_PATHS+=("$2"); shift 2 ;;
        --benchmarks)
            SCORER_ARGS+=("$1")
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                SCORER_ARGS+=("$1")
                shift
            done
            ;;
        --mbpp_config|--num_workers|--timeout)
            SCORER_ARGS+=("$1" "$2")
            shift 2
            ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done

[[ -n "$OUTPUT_DIR" && -n "$BIGCODE_ROOT" && -n "$HF_HOME_DIR" ]] || {
    usage
    exit 2
}

for path in "$OUTPUT_DIR" "$BIGCODE_ROOT" "$HF_HOME_DIR" "$PYTHON_BIN"; do
    [[ -e "$path" ]] || { echo "Missing path: $path" >&2; exit 1; }
done
if [[ -n "$EXTRA_PYTHONPATH" && ! -e "$EXTRA_PYTHONPATH" ]]; then
    echo "Missing path: $EXTRA_PYTHONPATH" >&2
    exit 1
fi

OUTPUT_DIR="$(realpath -e "$OUTPUT_DIR")"
BIGCODE_ROOT="$(realpath -e "$BIGCODE_ROOT")"
HF_HOME_DIR="$(realpath -e "$HF_HOME_DIR")"
PYTHON_BIN="$(realpath -e "$PYTHON_BIN")"
if [[ -n "$EXTRA_PYTHONPATH" ]]; then
    EXTRA_PYTHONPATH="$(realpath -e "$EXTRA_PYTHONPATH")"
fi

# Refuse obvious broad or system targets.  The wrapper is intentionally
# conservative because the scorer writes execution artifacts.
case "$OUTPUT_DIR" in
    /|/bin|/boot|/dev|/etc|/lib|/lib/*|/proc|/root|/sbin|/sys|/usr|/var)
        echo "Refusing broad/system output_dir: $OUTPUT_DIR" >&2
        exit 2
        ;;
esac

for i in "${!HIDE_PATHS[@]}"; do
    [[ -d "${HIDE_PATHS[$i]}" ]] || {
        echo "Missing hide_path: ${HIDE_PATHS[$i]}" >&2
        exit 1
    }
    HIDE_PATHS[$i]="$(realpath -e "${HIDE_PATHS[$i]}")"
done

# A hidden parent would hide the output or an evaluator input before the exact
# bind mount can be restored.  Refuse this ambiguous layout instead of
# silently writing somewhere unexpected.
for hidden in "${HIDE_PATHS[@]}"; do
    for source in "$OUTPUT_DIR" "$BIGCODE_ROOT" "$HF_HOME_DIR" "$PYTHON_BIN"; do
        case "$source" in
            "$hidden"|"$hidden"/*)
                echo "source $source is below hidden path $hidden; choose explicit non-hidden staging" >&2
                exit 2
                ;;
        esac
    done
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
EVAL_HOST="$(realpath -e "$SCRIPT_DIR")"

for hidden in "${HIDE_PATHS[@]}"; do
    case "$EVAL_HOST" in
        "$hidden"|"$hidden"/*)
            echo "evaluator source is below hidden path $hidden; choose explicit non-hidden staging" >&2
            exit 2
            ;;
    esac
done

if [[ -d /dev/shm && -w /dev/shm ]]; then
    STAGE_ROOT="$(mktemp -d /dev/shm/gac_code_eval_stage.XXXXXX)"
else
    STAGE_ROOT="$(mktemp -d)"
fi
cleanup_stage() {
    [[ -n "${STAGE_ROOT:-}" && -d "$STAGE_ROOT" ]] && rm -rf -- "$STAGE_ROOT"
}
trap cleanup_stage EXIT

command -v unshare >/dev/null 2>&1 || {
    echo "unshare is required; refusing to execute generated code" >&2
    exit 1
}

# The source directories are first mounted read-only into a private staging
# tree.  The namespace then hides the requested host roots and re-exposes only
# the exact evaluator/cache/output mount points.
unshare --user --map-root-user --mount --net --pid --fork --mount-proc \
    /bin/bash -s -- \
    "$OUTPUT_DIR" "$BIGCODE_ROOT" "$HF_HOME_DIR" "$PYTHON_BIN" \
    "${EXTRA_PYTHONPATH:-}" "$EVAL_HOST" "$STAGE_ROOT" \
    "${#HIDE_PATHS[@]}" "${HIDE_PATHS[@]}" \
    "${SCORER_ARGS[@]}" <<'NAMESPACE'
set -euo pipefail

OUTPUT_HOST="$1"
BIGCODE_HOST="$2"
HF_HOST="$3"
PYTHON="$4"
EXTRA_HOST="$5"
EVAL_HOST="$6"
STAGE_ROOT="$7"
HIDE_COUNT="$8"
shift 8

for ((i = 0; i < HIDE_COUNT; i++)); do
    HIDE_PATH="$1"
    shift
    mount -t tmpfs -o size=128M,nosuid,nodev,noexec tmpfs "$HIDE_PATH"
done

mount --make-rprivate /
mkdir -p "$STAGE_ROOT"/{out,bigcode,hf,eval}
mount --bind "$OUTPUT_HOST" "$STAGE_ROOT/out"
mount --bind "$BIGCODE_HOST" "$STAGE_ROOT/bigcode"
mount --bind "$HF_HOST" "$STAGE_ROOT/hf"
mount --bind "$EVAL_HOST" "$STAGE_ROOT/eval"
for source in bigcode hf eval; do
    mount -o remount,bind,ro "$STAGE_ROOT/$source"
done
if [[ -n "$EXTRA_HOST" ]]; then
    mkdir -p "$STAGE_ROOT/extra"
    mount --bind "$EXTRA_HOST" "$STAGE_ROOT/extra"
    mount -o remount,bind,ro "$STAGE_ROOT/extra"
fi

mkdir -p /tmp /var/tmp /mnt
mount -t tmpfs -o size=2G,nosuid,nodev,noexec tmpfs /tmp
mount -t tmpfs -o size=64M,nosuid,nodev,noexec tmpfs /var/tmp
mount -t tmpfs -o size=64M,nosuid,nodev,noexec tmpfs /mnt

# Restore only exact mount points after hiding common private roots.  Output is
# the sole writable mount; evaluator, code, and cache sources remain read-only.
mount --bind "$STAGE_ROOT/out" "$OUTPUT_HOST"
mount --bind "$STAGE_ROOT/bigcode" "$BIGCODE_HOST"
mount -o remount,bind,ro "$BIGCODE_HOST"
mount --bind "$STAGE_ROOT/hf" "$HF_HOST"
mount -o remount,bind,ro "$HF_HOST"
mount --bind "$STAGE_ROOT/eval" "$EVAL_HOST"
mount -o remount,bind,ro "$EVAL_HOST"
if [[ -n "$EXTRA_HOST" ]]; then
    mount --bind "$STAGE_ROOT/extra" "$EXTRA_HOST"
    mount -o remount,bind,ro "$EXTRA_HOST"
fi

mkdir -p /tmp/gac_home /tmp/gac_cache /tmp/gac_hf_home/datasets /tmp/gac_hf_home/hub
# Dataset caches are copied into namespace-local tmpfs because datasets may
# create lock files even when HF offline mode is enabled.
for cache in \
    "$HF_HOST/datasets/google-research-datasets___mbpp" \
    "$HF_HOST/datasets/openai___openai_humaneval"; do
    if [[ -d "$cache" ]]; then
        cp -a -- "$cache" /tmp/gac_hf_home/datasets/
    fi
done

export HOME=/tmp/gac_home
export TMPDIR=/tmp
export XDG_CACHE_HOME=/tmp/gac_cache
export PYTHONNOUSERSITE=1
export HF_HOME=/tmp/gac_hf_home
export HF_DATASETS_CACHE=/tmp/gac_hf_home/datasets
export HF_HUB_CACHE=/tmp/gac_hf_home/hub
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export GAC_CODE_EVAL_SANDBOX=1
export HF_ALLOW_CODE_EVAL=1
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy || true
if [[ -n "$EXTRA_HOST" ]]; then
    export PYTHONPATH="$BIGCODE_HOST:$EXTRA_HOST:$EVAL_HOST"
else
    export PYTHONPATH="$BIGCODE_HOST:$EVAL_HOST"
fi
cd /tmp

ulimit -n 512 || true
ulimit -u 512 || true
ulimit -f 1048576 || true

exec timeout --signal=TERM 7200 \
    "$PYTHON" "$EVAL_HOST/score_code.py" \
    --output_dir "$OUTPUT_HOST" \
    --bigcode_root "$BIGCODE_HOST" \
    "$@"
NAMESPACE
