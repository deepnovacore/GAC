"""Small privacy and dataset-integrity helpers shared by the public evaluator."""

from __future__ import annotations

from pathlib import Path


def public_model_label(model_path: str) -> str:
    """Return a stable short label without embedding a local absolute path."""

    value = str(model_path).rstrip("/")
    if not value:
        return "model"
    label = Path(value).name
    return label or "model"


def require_dataset_size(name: str, actual: int, expected: int) -> None:
    """Fail closed when a mirror silently returns a partial or wrong split."""

    if actual != expected:
        raise RuntimeError(
            f"{name}: expected {expected} examples, found {actual}; "
            "refusing to report a misleading score"
        )
