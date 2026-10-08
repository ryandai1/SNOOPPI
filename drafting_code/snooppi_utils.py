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
import warnings
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
SPLIT_NAMES = ("train", "validation", "test")
ENCODER_SPECS = {
    "ProtT5": {
        "filename": "prot_t5_embedding_cache.pt",
        "model_id": "Rostlab/prot_t5_xl_half_uniref50-enc",
        "embedding_dim": 1024,
    },
    "ESM-2 650M": {
        "filename": "esm2_t33_650M_snooppi_mean_pool_embeddings.pt",
        "model_id": "facebook/esm2_t33_650M_UR50D",
        "embedding_dim": 1280,
    },
    "ESM-C 600M": {
        "filename": "esmc_600m_snooppi_mean_pool_embeddings.pt",
        "model_id": "esmc_600m",
        "embedding_dim": 1152,
    },
}


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

    root = Path(root or resolve_root()).expanduser().resolve()
    configured_split = os.environ.get("SNOOPPI_SPLIT_DIR")
    split_dir = (
        Path(configured_split).expanduser() if configured_split else root / "splits"
    )
    if not split_dir.is_absolute():
        split_dir = root / split_dir
    return {
        "root": root,
        "splits": split_dir.resolve(),
        "embeddings": root / "embeddings",
        "tables": root / "tables" / "drafting_code",
        "figures": root / "figures" / "drafting_code",
        "checkpoints": root / "checkpoints" / "drafting_code",
        "candidate_negatives": root / "raw" / "snooppi_unknown_candidates.csv",
        "metadata": root / "raw" / "protein_metadata.csv",
        "clusters": root / "raw" / "mmseqs2_clusters.csv",
    }


def normalize_sequence(sequence: object, max_residues: int = MAX_RESIDUES) -> str:
    """Match the ESM input rule for audits; caches remain keyed by raw strings."""

    normalized = str(sequence).upper().replace(" ", "")
    normalized = re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "X", normalized)
    return normalized[:max_residues]


def sequence_id(sequence: object) -> str:
    """Stable non-reversible identifier; avoids writing raw sequences to audits."""

    return hashlib.sha256(normalize_sequence(sequence).encode()).hexdigest()


def full_sequence_id(sequence: object) -> str:
    """Metadata identity: uppercase, literal spaces removed, NO truncation/substitution."""
    return hashlib.sha256(str(sequence).upper().replace(" ", "").encode()).hexdigest()


def prott5_normalize_sequence(sequence: object) -> str:
    """Exactly the preserved ProtT5 input rule, including literal spaces."""
    return re.sub(r"[UZOB]", "X", str(sequence).upper())[:MAX_RESIDUES]


def full_pair_id(sequence_a: object, sequence_b: object) -> str:
    return hashlib.sha256(
        "|".join(
            sorted((full_sequence_id(sequence_a), full_sequence_id(sequence_b)))
        ).encode()
    ).hexdigest()


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
    if frame.columns.duplicated().any():
        raise ValueError(
            f"{source} has ambiguous duplicate columns after alias normalization"
        )

    if LABEL not in frame and "SNOOPPI_final_label" in frame:
        label_text = frame["SNOOPPI_final_label"].astype(str).str.strip().str.lower()
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
    numeric_labels = pd.to_numeric(frame[LABEL], errors="raise")
    if numeric_labels.isna().any() or not numeric_labels.isin([0, 1]).all():
        raise ValueError(f"{source} has labels outside {{0, 1}}")
    frame[LABEL] = numeric_labels.astype(np.int8)
    if "SNOOPPI_final_label" in frame:
        documented = frame["SNOOPPI_final_label"].astype(str).str.strip().str.lower()
        if not documented.isin(["positive", "negative"]).all():
            raise ValueError(f"{source} contains unknown/non-binary documented labels")
        if not np.array_equal(frame[LABEL], documented.eq("positive").astype(np.int8)):
            raise ValueError(
                f"{source} numeric labels disagree with SNOOPPI_final_label"
            )
    if frame[[SEQUENCE_A, SEQUENCE_B]].isna().any().any():
        raise ValueError(f"{source} contains missing protein sequences")
    for column in (SEQUENCE_A, SEQUENCE_B):
        if (
            not frame[column]
            .map(lambda x: isinstance(x, str) and bool(x.strip()))
            .all()
        ):
            raise ValueError(f"{source} has empty or non-string sequences in {column}")
    # Explicit provenance: documented source labels or the existing binary CSV.
    if "label_evidence_source" not in frame:
        frame["label_evidence_source"] = (
            "SNOOPPI_final_label"
            if "SNOOPPI_final_label" in frame
            else f"binary_split_csv:{source.name}"
        )
    if (
        frame["label_evidence_source"].isna().any()
        or frame["label_evidence_source"].astype(str).str.strip().eq("").any()
    ):
        raise ValueError(f"{source} has missing label evidence sources")
    return frame


def split_csv_paths(paths) -> dict[str, Path]:
    """Use the three explicit CSV paths configured in each notebook's setup."""
    files = paths.get(
        "split_files", {name: paths["splits"] / f"{name}.csv" for name in SPLIT_NAMES}
    )
    if set(files) != set(SPLIT_NAMES):
        raise ValueError(
            "Split file paths must name train, validation and test exactly"
        )
    resolved = {}
    for name, value in files.items():
        path = Path(value).expanduser()
        resolved[name] = (
            path if path.is_absolute() else paths["splits"] / path
        ).resolve()
    if len(set(resolved.values())) != len(SPLIT_NAMES):
        raise ValueError(
            "Train, validation and test must use three different CSV files"
        )
    return resolved


def load_splits(paths) -> dict[str, pd.DataFrame]:
    """Load the selected train/validation/test CSVs without recomputing partitions."""

    split_files = split_csv_paths(paths)
    missing = [str(path) for path in split_files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing selected split files. Restore reviewed CSVs or check "
            f"the CSV paths in the notebook setup; do not regenerate existing splits. Missing: {missing}"
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
    saved_manifest = paths["splits"] / "manifest.json"
    if saved_manifest.is_file():
        manifest = json.loads(saved_manifest.read_text())
        current_hashes = split_manifest(paths)["sha256"]
        if manifest.get("sha256") != current_hashes:
            raise ValueError(
                "Selected split CSV hashes disagree with their saved manifest; do not use edited/stale partitions"
            )
        protocol = manifest.get("protocol", {})
        group_col = protocol.get("group_column")
        if group_col:
            if any(
                group_col not in part or part[group_col].isna().any()
                for part in splits.values()
            ):
                raise ValueError(
                    "Selected protocol is missing its declared grouping column"
                )
            groups = {name: set(part[group_col]) for name, part in splits.items()}
            if any(
                groups[a] & groups[b]
                for i, a in enumerate(SPLIT_NAMES)
                for b in SPLIT_NAMES[i + 1 :]
            ):
                raise ValueError(
                    "Declared strict grouping column crosses split boundaries"
                )
        if protocol.get("name") == "cold_target":
            target = protocol["target_side"]
            for earlier, later in (
                ("train", "validation"),
                ("train", "test"),
                ("validation", "test"),
            ):
                earlier_ids = {
                    sequence_id(s)
                    for col in (SEQUENCE_A, SEQUENCE_B)
                    for s in splits[earlier][col]
                }
                if earlier_ids & set(splits[later][target].map(sequence_id)):
                    raise ValueError(
                        "A held-out target appears in an earlier partition"
                    )
                t5_earlier = {
                    prott5_normalize_sequence(s)
                    for col in (SEQUENCE_A, SEQUENCE_B)
                    for s in splits[earlier][col]
                }
                if t5_earlier & set(
                    splits[later][target].map(prott5_normalize_sequence)
                ):
                    raise ValueError(
                        "A held-out ProtT5 target input appears in an earlier partition"
                    )
        if protocol.get("name") in {"cold_protein", "cold_cluster", "cold_family"}:
            if leakage_report(splits).overlapping_proteins.sum():
                raise ValueError(
                    "The declared strict protocol shares normalized proteins"
                )
            identified = {n: add_pair_ids(f) for n, f in splits.items()}
            proteins = {
                n: set(f.prott5_protein_A_id) | set(f.prott5_protein_B_id)
                for n, f in identified.items()
            }
            if any(
                proteins[a] & proteins[b]
                for i, a in enumerate(SPLIT_NAMES)
                for b in SPLIT_NAMES[i + 1 :]
            ):
                raise ValueError(
                    "Declared strict protocol shares ProtT5 input identities"
                )
            field = protocol.get("mapping_field")
            if field:
                cols = [f"A_{field}", f"B_{field}"]
                if any(any(c not in f for c in cols) for f in splits.values()):
                    raise ValueError(
                        "Declared mapped protocol is missing partner group assignments"
                    )
                groups = {
                    n: set(f[cols[0]]) | set(f[cols[1]]) for n, f in splits.items()
                }
                if any(
                    groups[a] & groups[b]
                    for i, a in enumerate(SPLIT_NAMES)
                    for b in SPLIT_NAMES[i + 1 :]
                ):
                    raise ValueError(
                        "Declared mapped groups occur across partitions in either partner role"
                    )
    return splits


def file_sha256(path: Path, chunk_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_manifest(paths) -> dict[str, object]:
    """Record input identity before comparing models or random seeds."""

    files = split_csv_paths(paths)
    hashes = {name: file_sha256(path) for name, path in files.items()}
    return {
        "schema_version": 2,
        "split_directory": str(paths["splits"]),
        "files": {name: str(path) for name, path in files.items()},
        "sha256": hashes,
        "split_set_sha256": hashlib.sha256(
            json.dumps(hashes, sort_keys=True).encode()
        ).hexdigest(),
    }


def save_json(payload: object, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _load_trusted_torch_file(path: Path):
    """Load trusted cache/checkpoint exports with the shared PyTorch compatibility rule."""

    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required to load the existing .pt caches"
        ) from exc

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # Compatibility with older Colab PyTorch versions.
        return torch.load(path, map_location="cpu")


def load_embedding_cache(
    path: Path, encoder: str | None = None
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
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

    if not isinstance(raw_cache, dict) or not raw_cache:
        raise ValueError(f"Empty or invalid embedding dictionary in {path}")
    if encoder is not None:
        spec = ENCODER_SPECS[encoder]
        if encoder != "ProtT5" and metadata.get("format") == "legacy_direct_dictionary":
            raise ValueError(
                f"{encoder} requires the metadata-wrapped cache from its original notebook"
            )
        expected = {
            "model_id": spec["model_id"],
            "max_residues": MAX_RESIDUES,
            "embedding_dim": spec["embedding_dim"],
            "pooling": "mean over non-special residue tokens",
        }
        if encoder == "ESM-C 600M":
            expected.update(
                normalization="canonical_X_spaces_removed_prefix_v1",
                esm_version="3.2.1",
            )
        if metadata.get("format") != "legacy_direct_dictionary":
            for key, value in expected.items():
                if metadata.get(key) != value:
                    raise ValueError(
                        f"{path}: cache {key}={metadata.get(key)!r}, expected {value!r}"
                    )
        else:
            warnings.warn(
                "Legacy ProtT5 cache has no encoder revision/preprocessing metadata; dimension and coverage will be checked.",
                stacklevel=2,
            )

    cache: dict[str, np.ndarray] = {}
    dimensions = set()
    for sequence, value in raw_cache.items():
        array = (
            value.detach().cpu().float().numpy()
            if hasattr(value, "detach")
            else np.asarray(value)
        )
        array = np.asarray(array, dtype=np.float32)
        if array.ndim != 1 or not np.isfinite(array).all():
            raise ValueError(f"Invalid embedding for one sequence in {path}")
        if not isinstance(sequence, str):
            raise ValueError(f"Cache keys must be original sequence strings: {path}")
        dimensions.add(len(array))
        cache[str(sequence)] = array
    if len(dimensions) != 1 or next(iter(dimensions)) == 0:
        raise ValueError(
            f"Inconsistent or empty embedding dimensions in {path}: {dimensions}"
        )
    if encoder is not None and dimensions != {ENCODER_SPECS[encoder]["embedding_dim"]}:
        raise ValueError(f"Wrong embedding dimension for {encoder}: {dimensions}")
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
    if unknown or not operations:
        raise ValueError(f"Unknown pair operations: {sorted(unknown)}")

    missing = {
        str(sequence)
        for column in (SEQUENCE_A, SEQUENCE_B)
        for sequence in frame[column]
        if str(sequence) not in cache
    }
    if missing:
        raise KeyError(
            f"Embedding cache misses {len(missing):,} original sequence keys. "
            "Check SNOOPPI_ROOT, selected split directory, and cache path; no rows were dropped."
        )

    embedding_a = np.stack([cache[str(value)] for value in frame[SEQUENCE_A]])
    embedding_b = np.stack([cache[str(value)] for value in frame[SEQUENCE_B]])
    feature_blocks = {
        "sum": lambda: embedding_a + embedding_b,
        "absdiff": lambda: np.abs(embedding_a - embedding_b),
        "product": lambda: embedding_a * embedding_b,
        # Ordered concatenation is intentionally asymmetric and should be used
        # only as a diagnostic against the biologically symmetric formulation.
        "concat": lambda: np.concatenate([embedding_a, embedding_b], axis=1),
    }
    return np.concatenate(
        [feature_blocks[name]() for name in operations], axis=1
    ).astype(
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
    scores = [
        f1_score(labels, probabilities >= value, average="macro") for value in grid
    ]
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


def cache_paths(paths):
    """All three cached PLMs; no encoder/model download is performed."""
    defaults = {
        name: paths["embeddings"] / spec["filename"]
        for name, spec in ENCODER_SPECS.items()
    }
    files = paths.get("cache_files", defaults)
    if set(files) != set(ENCODER_SPECS):
        raise ValueError("Cache paths must name the three configured PLMs exactly")
    result = {}
    for name, value in files.items():
        path = Path(value).expanduser()
        result[name] = (
            path if path.is_absolute() else paths["embeddings"] / path
        ).resolve()
    return result


def add_pair_ids(frame):
    frame = frame.copy()
    frame["protein_A_id"] = frame[SEQUENCE_A].map(sequence_id)
    frame["protein_B_id"] = frame[SEQUENCE_B].map(sequence_id)
    frame["pair_id"] = [
        canonical_pair_id(a, b) for a, b in zip(frame[SEQUENCE_A], frame[SEQUENCE_B])
    ]
    frame["full_pair_id"] = [
        full_pair_id(a, b) for a, b in zip(frame[SEQUENCE_A], frame[SEQUENCE_B])
    ]
    for side, col in (("A", SEQUENCE_A), ("B", SEQUENCE_B)):
        frame[f"prott5_protein_{side}_id"] = frame[col].map(
            lambda s: hashlib.sha256(prott5_normalize_sequence(s).encode()).hexdigest()
        )
    frame["prott5_pair_id"] = [
        hashlib.sha256("|".join(sorted((a, b))).encode()).hexdigest()
        for a, b in zip(frame.prott5_protein_A_id, frame.prott5_protein_B_id)
    ]
    return frame


def leakage_report(splits):
    """Conservative ESM input-prefix audit; original full-sequence identity is separate."""
    audited = {name: add_pair_ids(frame) for name, frame in splits.items()}
    rows = []
    for name, frame in audited.items():
        rows.append(
            {
                "left": name,
                "right": name,
                "duplicate_pairs": int(frame.pair_id.duplicated().sum()),
                "conflicting_pairs": int(
                    frame.groupby("pair_id")[LABEL].nunique().gt(1).sum()
                ),
                "overlapping_pairs": 0,
                "overlapping_proteins": 0,
            }
        )
    for i, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[i + 1 :]:
            a, b = audited[left], audited[right]
            shared = set(a.pair_id) & set(b.pair_id)
            both = pd.concat([a, b])
            conflicts = both.groupby("pair_id")[LABEL].nunique()
            rows.append(
                {
                    "left": left,
                    "right": right,
                    "duplicate_pairs": 0,
                    "conflicting_pairs": int(
                        conflicts.reindex(list(shared)).gt(1).sum()
                    ),
                    "overlapping_pairs": len(shared),
                    "overlapping_proteins": len(
                        (set(a.protein_A_id) | set(a.protein_B_id))
                        & (set(b.protein_A_id) | set(b.protein_B_id))
                    ),
                }
            )
    return pd.DataFrame(rows)


def assert_model_ready(splits):
    """No silent deduplication or train/test repair in modeling notebooks."""
    report = leakage_report(splits)
    if (
        report[["duplicate_pairs", "conflicting_pairs", "overlapping_pairs"]]
        .to_numpy()
        .any()
    ):
        raise ValueError(
            "Duplicate/conflicting or cross-split normalized pairs detected. Review notebook 00 and select a reviewed split set from notebook 02 before fitting. Original CSVs were not changed."
        )
    if report.overlapping_proteins.sum():
        warnings.warn(
            "Selected split shares proteins across partitions; it cannot support a cold-protein claim.",
            stacklevel=2,
        )
    identified = pd.concat(
        [add_pair_ids(f).assign(partition=n) for n, f in splits.items()],
        ignore_index=True,
    )
    if identified.prott5_pair_id.duplicated().any():
        raise ValueError(
            "Duplicate/cross-split pairs collide under the preserved ProtT5 input rule; review notebook 00/02"
        )
    cluster_path = os.environ.get("SNOOPPI_CLUSTER_FILE")
    if cluster_path:
        from snooppi_protocol import load_reviewed_mapping, mapped_overlap_report

        mapping = load_reviewed_mapping(Path(cluster_path), "cluster_id", identified)
        overlap = mapped_overlap_report(splits, mapping, "cluster_id")
        if overlap.filter(like="overlapping_").to_numpy().any():
            raise ValueError(
                "The explicitly selected MMseqs2 mapping has cross-split cluster overlap; select a reviewed cold-cluster protocol"
            )
    return report


def validate_cache_coverage(cache, splits):
    required = {
        str(s)
        for frame in splits.values()
        for column in (SEQUENCE_A, SEQUENCE_B)
        for s in frame[column]
    }
    missing = required - set(cache)
    if missing:
        raise KeyError(
            f"Cache misses {len(missing)} required original sequence keys; no rows will be dropped or keys renormalized."
        )
    return {
        "required_sequences": len(required),
        "cached_sequences": len(cache),
        "missing_sequences": 0,
    }


def preprocessing_report(splits):
    sequences = {
        s
        for frame in splits.values()
        for col in (SEQUENCE_A, SEQUENCE_B)
        for s in frame[col]
    }
    return {
        "unique_raw_sequences": len(sequences),
        "different_prott5_vs_esm_inputs": sum(
            prott5_normalize_sequence(s) != normalize_sequence(s) for s in sequences
        ),
        "ProtT5": "uppercase_UZOB_to_X_prefix1024",
        "ESM": "canonical_X_spaces_removed_prefix_v1",
        "metadata_identity": "full_uppercase_spaces_removed_sha256_v1",
        "leakage_identity": "canonical_X_spaces_removed_prefix1024_sha256_v1",
    }


def require_comparable_preprocessing(splits, encoders, allow_difference=False):
    report = preprocessing_report(splits)
    if (
        "ProtT5" in encoders
        and len(encoders) > 1
        and report["different_prott5_vs_esm_inputs"]
    ):
        if not allow_difference:
            raise ValueError(
                f"{report['different_prott5_vs_esm_inputs']} raw sequences have different ProtT5/ESM inputs in the existing caches. Set ALLOW_PREPROCESSING_DIFFERENCES=True only for a documented comparison of the cached workflows, or restrict the encoder list."
            )
        warnings.warn(
            "Comparing cached workflows with different preprocessing; encoder identity is not isolated.",
            stacklevel=2,
        )
    return report


def make_run_manifest(paths, splits, configuration, cache_records=None):
    manifest = split_manifest(paths)
    manifest["configuration"] = configuration
    manifest["class_counts"] = {
        name: {
            "n": len(frame),
            "positive": int(frame[LABEL].sum()),
            "negative": int((frame[LABEL] == 0).sum()),
        }
        for name, frame in splits.items()
    }
    manifest["preprocessing"] = preprocessing_report(splits)
    manifest["caches"] = cache_records or {}
    import importlib.metadata

    manifest["packages"] = {
        p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scikit-learn")
    }
    manifest["selection"] = {
        "threshold": "validation macro-F1; 181 points in [0.05,0.95]",
        "checkpoint": "validation macro-AP",
    }
    return manifest


def prediction_identity(frame, manifest):
    """Joinable identity independent of filenames; retains exact evaluated order."""
    identified = add_pair_ids(frame)
    result = (
        identified[["pair_id", "full_pair_id", LABEL]].copy().reset_index(drop=True)
    )
    result["row_index"] = np.arange(len(result))
    result["split_sha256"] = manifest["sha256"]["test"]
    result["split_set_sha256"] = manifest["split_set_sha256"]
    result["protocol"] = Path(manifest["split_directory"]).name
    return result


def validate_prediction_identity(frame, expected):
    required = {
        "row_index",
        "pair_id",
        "full_pair_id",
        "split_sha256",
        "split_set_sha256",
        "label",
        "probability",
        "threshold",
        "quick_mode",
    }
    if required - set(frame):
        raise ValueError(
            f"Prediction schema missing {sorted(required - set(frame))}; regenerate legacy predictions with the revised notebooks"
        )
    if frame.row_index.duplicated().any() or len(frame) != len(expected):
        raise ValueError("Predictions must contain each selected test row exactly once")
    ordered = frame.sort_values("row_index").reset_index(drop=True)
    for col in (
        "row_index",
        "pair_id",
        "full_pair_id",
        "split_sha256",
        "split_set_sha256",
        "label",
    ):
        if not np.array_equal(ordered[col].to_numpy(), expected[col].to_numpy()):
            raise ValueError(f"Predictions disagree with selected test input on {col}")
    for col in ("probability", "threshold"):
        if not pd.to_numeric(ordered[col], errors="raise").between(0, 1).all():
            raise ValueError(f"Invalid {col}")
    if (
        not ordered.quick_mode.isin([True, False]).all()
        or ordered.quick_mode.nunique() != 1
    ):
        raise ValueError("quick_mode must be a constant boolean within a run")
    if ordered.threshold.nunique() != 1:
        raise ValueError("threshold must be fixed per run from validation")
    return ordered


def align_paired_predictions(reference, candidate):
    """Refuse inner-join intersections, different split hashes, or duplicate rows."""
    keys = ["split_set_sha256", "split_sha256", "row_index", "pair_id", "full_pair_id"]
    for frame in (reference, candidate):
        if (
            frame[keys + [LABEL, "probability"]].isna().any().any()
            or frame.duplicated(keys).any()
        ):
            raise ValueError("Invalid or duplicate paired prediction identities")
    a = reference.sort_values(keys).reset_index(drop=True)
    b = candidate.sort_values(keys).reset_index(drop=True)
    if len(a) != len(b) or not a[keys + [LABEL]].equals(b[keys + [LABEL]]):
        raise ValueError(
            "Paired runs must have identical complete split hashes, pair identities, row indices and labels"
        )
    if (
        "quick_mode" in a
        and "quick_mode" in b
        and not a.quick_mode.equals(b.quick_mode)
    ):
        raise ValueError("Cannot pair development and reporting runs")
    return (
        a[LABEL].to_numpy(int),
        a.probability.to_numpy(float),
        b.probability.to_numpy(float),
    )


def check_output_paths(paths):
    existing = [str(p) for p in paths if Path(p).exists()]
    if existing:
        raise FileExistsError(
            f"Refusing to overwrite outputs: {existing}; change RUN_TAG"
        )


def prepare_outputs(paths):
    """Called only inside enabled write blocks."""
    check_output_paths(paths)
    for path in paths:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
