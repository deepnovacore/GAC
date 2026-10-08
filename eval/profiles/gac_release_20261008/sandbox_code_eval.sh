#!/usr/bin/env bash
# Narrow chroot in fresh user/mount/network/PID namespaces. Never fall back
# to direct execution. The host home, datasets, credentials and GPUs are absent.
set -euo pipefail

OUTPUT_DIR=""
BIGCODE_ROOT=""
PYTHON_BIN="python3"
EXTRA_PYTHONPATH=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --bigcode_root) BIGCODE_ROOT="$2"; shift 2 ;;
        --python_bin) PYTHON_BIN="$2"; shift 2 ;;
        --extra_pythonpath) EXTRA_PYTHONPATH="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: bash sandbox_code_eval.sh --output_dir DIR --bigcode_root DIR"
            echo "       [--python_bin PATH] [--extra_pythonpath DIR]"
            exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ -n "$OUTPUT_DIR" && -n "$BIGCODE_ROOT" ]] || { echo "Missing required arguments" >&2; exit 2; }
command -v unshare >/dev/null || { echo "unshare required; refusing code execution" >&2; exit 1; }
command -v chroot >/dev/null || { echo "chroot required; refusing code execution" >&2; exit 1; }
command -v setpriv >/dev/null || { echo "setpriv required; refusing code execution" >&2; exit 1; }
OUTPUT_DIR="$(realpath -e "$OUTPUT_DIR")"
BIGCODE_ROOT="$(realpath -e "$BIGCODE_ROOT")"
PYTHON_BIN="$(realpath -e "$(command -v "$PYTHON_BIN")")"
PROFILE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
[[ -f "$OUTPUT_DIR/code_requests.jsonl" ]] || { echo "Prepare code requests first" >&2; exit 1; }
case "$OUTPUT_DIR" in
    /|/home|/root|/usr|/var|/tmp|/dev|/proc|/sys|/mnt)
        echo "Refusing broad/system output directory" >&2; exit 2 ;;
esac
[[ ! -e "$OUTPUT_DIR/code_scores.json" ]] || { echo "Use an unscored output directory" >&2; exit 2; }
EXPECTED_BIGCODE="8fc5bae6479c4fbbb28c3f8b644f6a15b3f3b5bd"
[[ "$(git -C "$BIGCODE_ROOT" rev-parse HEAD)" == "$EXPECTED_BIGCODE" ]] || {
    echo "BigCode must be checked out at $EXPECTED_BIGCODE" >&2; exit 2;
}
git -C "$BIGCODE_ROOT" diff --quiet HEAD -- bigcode_eval || {
    echo "BigCode executor has local changes; refusing a different scorer" >&2; exit 2;
}
RUNTIME_DIR="$($PYTHON_BIN -I -c 'import sys; print(sys.prefix)')"
RUNTIME_DIR="$(realpath -e "$RUNTIME_DIR")"
PYTHON_RELATIVE="${PYTHON_BIN#"$RUNTIME_DIR/"}"
[[ "$PYTHON_RELATIVE" != "$PYTHON_BIN" ]] || {
    echo "Python must be inside its runtime prefix" >&2; exit 2;
}
if [[ -n "$EXTRA_PYTHONPATH" ]]; then EXTRA_PYTHONPATH="$(realpath -e "$EXTRA_PYTHONPATH")"; fi
[[ -d /dev/shm && -w /dev/shm ]] || { echo "Private tmpfs staging required" >&2; exit 1; }
STAGE_DIR="$(mktemp -d /dev/shm/gac_release_score.XXXXXX)"
trap 'rmdir -- "$STAGE_DIR" 2>/dev/null || true' EXIT

env -i PATH=/usr/sbin:/usr/bin:/sbin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
unshare --user --map-root-user --mount --net --pid --fork --mount-proc \
    /bin/bash -s -- "$STAGE_DIR" "$OUTPUT_DIR" "$BIGCODE_ROOT" \
    "$PROFILE_DIR" "$RUNTIME_DIR" "$PYTHON_RELATIVE" "$EXTRA_PYTHONPATH" <<'ISOLATED'
set -euo pipefail
STAGE="$1"
OUTPUT_HOST="$2"
BIGCODE_HOST="$3"
PROFILE_HOST="$4"
RUNTIME_HOST="$5"
PYTHON_RELATIVE="$6"
EXTRA_HOST="$7"
mount --make-rprivate /
mount -t tmpfs -o size=2G,nosuid,nodev tmpfs "$STAGE"
mkdir -p "$STAGE"/{usr,bin,lib,lib64,etc,proc,dev,dev/shm,tmp,work,profile,bigcode,runtime,extra}
bind_readonly() {
    mount --bind "$1" "$2"
    mount -o remount,bind,ro "$2"
}
for directory in usr bin lib lib64; do
    if [[ -d "/$directory" ]]; then bind_readonly "/$directory" "$STAGE/$directory"; fi
done
if [[ -f /etc/ld.so.cache ]]; then
    touch "$STAGE/etc/ld.so.cache"
    bind_readonly /etc/ld.so.cache "$STAGE/etc/ld.so.cache"
fi
for device in null zero random urandom; do
    touch "$STAGE/dev/$device"
    bind_readonly "/dev/$device" "$STAGE/dev/$device"
done
mount -t tmpfs -o size=256M,nosuid,nodev tmpfs "$STAGE/dev/shm"
mount -t proc proc "$STAGE/proc"
mount --bind "$OUTPUT_HOST" "$STAGE/work"
bind_readonly "$PROFILE_HOST" "$STAGE/profile"
bind_readonly "$BIGCODE_HOST" "$STAGE/bigcode"
bind_readonly "$RUNTIME_HOST" "$STAGE/runtime"
if [[ -n "$EXTRA_HOST" ]]; then bind_readonly "$EXTRA_HOST" "$STAGE/extra"; fi
mkdir -p "$STAGE/tmp/gac_home" "$STAGE/tmp/cache"
chmod 1777 "$STAGE/tmp"
ulimit -n 512
ulimit -u 512
ulimit -f 1048576
# chroot removes aliases to host mounts; env -i prevents credential leakage.
# Only /work is a writable host mount. /tmp is disposable namespace memory.
exec chroot "$STAGE" /usr/bin/setpriv \
    --bounding-set=-all --inh-caps=-all --ambient-caps=-all --no-new-privs \
    /usr/bin/timeout --signal=TERM 7200 /usr/bin/env -i \
    PATH=/runtime/bin:/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    HOME=/tmp/gac_home TMPDIR=/tmp XDG_CACHE_HOME=/tmp/cache \
    LD_LIBRARY_PATH=/runtime/lib PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/profile:/bigcode:/extra \
    HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_ALLOW_CODE_EVAL=1 \
    GAC_CODE_EVAL_SANDBOX=1 \
    "/runtime/$PYTHON_RELATIVE" /profile/score_code.py \
    --output_dir /work --bigcode_root /bigcode
ISOLATED
