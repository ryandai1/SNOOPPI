"""Shared, dependency-light helpers for the SNOOPPI draft notebooks.

The original notebooks are preserved research records.  New experiments import
this module so that path handling, column normalization, hashing, metrics, and
embedding-cache loading remain identical across ablations.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
)


SEQUENCE_A = "partner_A_sequence"
SEQUENCE_B = "partner_B_sequence"
LABEL = "label"
REQUIRED_COLUMNS = (SEQUENCE_A, SEQUENCE_B, LABEL)
MAX_RESIDUES = 1_024


def resolve_root() -> Path:
    """Return the experiment root without silently inventing a data location."""

    configured = os.environ.get("SNOOPPI_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()

    colab_root = Path("/content/drive/MyDrive/SNOOPPI")
    if colab_root.exists():
        return colab_root

    # Local fallback is useful when reviewing code from a clone.  Data are still
    # required under this directory; load_splits gives an actionable error if not.
    return Path.cwd().resolve()


def experiment_paths(root: Path | None = None) -> dict[str, Path]:
    """Centralize all paths used by the draft suite."""

    root = Path(root or resolve_root())
    return {
        "root": root,
        "splits": root / "splits",
        "embeddings": root / "embeddings",
        "tables": root / "tables" / "drafting_code",
        "figures": root / "figures" / "drafting_code",
        "checkpoints": root / "checkpoints" / "drafting_code",
        "candidate_negatives": root / "raw" / "snooppi_unknown_candidates.csv",
        "metadata": root / "raw" / "protein_metadata.csv",
    }


def ensure_output_directories(paths: Mapping[str, Path]) -> None:
    """Create only derived-output folders; input folders are never fabricated."""

    for key in ("tables", "figures", "checkpoints"):
        paths[key].mkdir(parents=True, exist_ok=True)


def normalize_sequence(sequence: object, max_residues: int = MAX_RESIDUES) -> str:
    """Match the ESM notebook normalization for leakage audits and cache lookup."""

    normalized = str(sequence).upper().replace(" ", "")
    normalized = re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "X", normalized)
    return normalized[:max_residues]


def sequence_id(sequence: object) -> str:
    """Stable non-reversible identifier; avoids writing raw sequences to audits."""

    return hashlib.sha256(normalize_sequence(sequence).encode()).hexdigest()


def canonical_pair_id(sequence_a: object, sequence_b: object) -> str:
    """Order-invariant pair identifier after normalization and truncation."""

    members = sorted((sequence_id(sequence_a), sequence_id(sequence_b)))
    return hashlib.sha256("|".join(members).encode()).hexdigest()


def normalize_split_columns(frame: pd.DataFrame, source: Path) -> pd.DataFrame:
    """Accept the column spellings documented by the preserved notebooks."""

    aliases = {
        "Partner_A_sequence": SEQUENCE_A,
        "Partner_B_sequence": SEQUENCE_B,
        "partner_a_sequence": SEQUENCE_A,
        "partner_b_sequence": SEQUENCE_B,
    }
    frame = frame.rename(columns={k: v for k, v in aliases.items() if k in frame})

    if LABEL not in frame and "SNOOPPI_final_label" in frame:
        label_text = frame["SNOOPPI_final_label"].astype(str).str.lower()
        unknown = ~label_text.isin({"positive", "negative"})
        if unknown.any():
            raise ValueError(
                f"{source} contains {int(unknown.sum())} non-binary labels; "
                "unknown pairs must not be converted to negatives."
            )
        frame[LABEL] = label_text.eq("positive").astype(np.int8)

    missing = [name for name in REQUIRED_COLUMNS if name not in frame]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")

    frame = frame.copy()
    frame[LABEL] = pd.to_numeric(frame[LABEL], errors="raise").astype(np.int8)
    if not set(frame[LABEL].unique()).issubset({0, 1}):
        raise ValueError(f"{source} has labels outside {{0, 1}}")
    if frame[[SEQUENCE_A, SEQUENCE_B]].isna().any().any():
        raise ValueError(f"{source} contains missing protein sequences")
    return frame


def load_splits(paths: Mapping[str, Path]) -> dict[str, pd.DataFrame]:
    """Load the exact train/validation/test CSVs used by the encoder notebooks."""

    split_files = {
        "train": paths["splits"] / "train.csv",
        "validation": paths["splits"] / "validation.csv",
        "test": paths["splits"] / "test.csv",
    }
    missing = [str(path) for path in split_files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing SNOOPPI split files. Run notebook 01's documented split export "
            f"or set SNOOPPI_ROOT. Missing: {missing}"
        )
    splits = {
        name: normalize_split_columns(pd.read_csv(path), path)
        for name, path in split_files.items()
    }
    for name, frame in splits.items():
        if set(frame[LABEL].unique()) != {0, 1}:
            raise ValueError(
                f"{split_files[name]} must contain both binary classes for the "
                "configured metrics and threshold selection"
            )
    return splits


def file_sha256(path: Path, chunk_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_manifest(paths: Mapping[str, Path]) -> dict[str, object]:
    """Record input identity before comparing models or random seeds."""

    files = {
        name: paths["splits"] / f"{name}.csv"
        for name in ("train", "validation", "test")
    }
    return {
        "files": {name: str(path) for name, path in files.items()},
        "sha256": {name: file_sha256(path) for name, path in files.items()},
    }


def save_json(payload: object, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _load_trusted_torch_file(path: Path):
    """Load only caches created by the repository notebooks."""

    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required to load the existing .pt caches") from exc

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # Compatibility with older Colab PyTorch versions.
        return torch.load(path, map_location="cpu")


def load_embedding_cache(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Load direct ProtT5 dictionaries and metadata-wrapped ESM caches."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Embedding cache not found: {path}")
    payload = _load_trusted_torch_file(path)
    if isinstance(payload, dict) and "embeddings" in payload:
        raw_cache = payload["embeddings"]
        metadata = {key: value for key, value in payload.items() if key != "embeddings"}
    elif isinstance(payload, dict):
        raw_cache = payload
        metadata = {"format": "legacy_direct_dictionary"}
    else:
        raise TypeError(f"Unsupported embedding cache payload in {path}")

    cache: dict[str, np.ndarray] = {}
    for sequence, value in raw_cache.items():
        array = value.detach().cpu().float().numpy() if hasattr(value, "detach") else np.asarray(value)
        array = np.asarray(array, dtype=np.float32)
        if array.ndim != 1 or not np.isfinite(array).all():
            raise ValueError(f"Invalid embedding for one sequence in {path}")
        cache[str(sequence)] = array
    return cache, metadata


def build_pair_features(
    frame: pd.DataFrame,
    cache: Mapping[str, np.ndarray],
    operations: Iterable[str] = ("sum", "absdiff", "product"),
) -> np.ndarray:
    """Build bounded, explicit pair features from frozen protein embeddings."""

    operations = tuple(operations)
    valid = {"sum", "absdiff", "product", "concat"}
    unknown = set(operations) - valid
    if unknown:
        raise ValueError(f"Unknown pair operations: {sorted(unknown)}")

    missing = {
        str(sequence)
        for column in (SEQUENCE_A, SEQUENCE_B)
        for sequence in frame[column]
        if str(sequence) not in cache
    }
    if missing:
        raise KeyError(
            f"Embedding cache misses {len(missing):,} sequences. Rebuild the cache "
            "with the matching encoder notebook before fitting baselines."
        )

    embedding_a = np.stack([cache[str(value)] for value in frame[SEQUENCE_A]])
    embedding_b = np.stack([cache[str(value)] for value in frame[SEQUENCE_B]])
    feature_blocks = {
        "sum": embedding_a + embedding_b,
        "absdiff": np.abs(embedding_a - embedding_b),
        "product": embedding_a * embedding_b,
        # Ordered concatenation is intentionally asymmetric and should be used
        # only as a diagnostic against the biologically symmetric formulation.
        "concat": np.concatenate([embedding_a, embedding_b], axis=1),
    }
    return np.concatenate([feature_blocks[name] for name in operations], axis=1).astype(
        np.float32,
        copy=False,
    )


def macro_average_precision(labels: np.ndarray, probabilities: np.ndarray) -> float:
    positive_ap = average_precision_score(labels, probabilities)
    negative_ap = average_precision_score(1 - labels, 1 - probabilities)
    return float((positive_ap + negative_ap) / 2)


def select_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    grid: np.ndarray | None = None,
) -> float:
    """Tune the decision threshold on validation labels only."""

    grid = np.asarray(grid if grid is not None else np.linspace(0.05, 0.95, 181))
    scores = [f1_score(labels, probabilities >= value, average="macro") for value in grid]
    return float(grid[int(np.argmax(scores))])


def summarize_predictions(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Metrics requested for strict, imbalanced PPI evaluation."""

    labels = np.asarray(labels, dtype=np.int8)
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-7, 1 - 1e-7)
    predictions = probabilities >= threshold
    positive_ap = average_precision_score(labels, probabilities)
    negative_ap = average_precision_score(1 - labels, 1 - probabilities)
    return {
        "auroc": float(roc_auc_score(labels, probabilities)),
        "positive_ap": float(positive_ap),
        "negative_ap": float(negative_ap),
        "macro_ap": float((positive_ap + negative_ap) / 2),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "brier": float(brier_score_loss(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "threshold": float(threshold),
        "prevalence": float(labels.mean()),
        "n": int(len(labels)),
    }


def expected_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Equal-width ECE with empty bins omitted from the weighted sum."""

    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(probabilities, edges[1:-1]), 0, n_bins - 1)
    error = 0.0
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if mask.any():
            error += mask.mean() * abs(labels[mask].mean() - probabilities[mask].mean())
    return float(error)
