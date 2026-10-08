"""Inference-only unknown-pair mining. No fitting, relabeling, or split writes."""

from contextlib import contextmanager
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile

import numpy as np
import pandas as pd
import torch
from torch import nn

from snooppi_utils import (
    SEQUENCE_A as A,
    SEQUENCE_B as B,
    _load_trusted_torch_file,
    build_pair_features,
    file_sha256,
    full_pair_id,
    full_sequence_id,
    load_embedding_cache,
    normalize_sequence,
    prott5_normalize_sequence,
)

SCORE_COLUMNS = ["pair_id", "Partner_A_sequence", "Partner_B_sequence", "score"]
IDENTITY_SCHEMES = ("full", "prott5_input", "esm_input")
NORMALIZATION = "uppercase_UZOB_to_X_first_1024_characters_v1"
MODEL_ID = "Rostlab/prot_t5_xl_half_uniref50-enc"
FEATURE_OPERATIONS = ("sum", "absdiff", "product")
CANDIDATE_COLUMNS = SCORE_COLUMNS + [
    "candidate_status",
    "label_evidence_source",
    "negative_evidence",
    "protein_overlap_any_labeled",
    "selection_threshold",
]


@lru_cache(maxsize=200_000)
def protein_ids(sequence):
    return (
        full_sequence_id(sequence),
        hashlib.sha256(prott5_normalize_sequence(sequence).encode()).hexdigest(),
        hashlib.sha256(normalize_sequence(sequence).encode()).hexdigest(),
    )


def labeled_protein_index(splits):
    """Index both roles in all labeled partitions; never just training targets."""
    result = {}
    for name in ("train", "validation", "test"):
        members = [set() for _ in IDENTITY_SCHEMES]
        for sequence in set(splits[name][A]) | set(splits[name][B]):
            for identities, identity in zip(members, protein_ids(sequence)):
                identities.add(identity)
        result[name] = members
    result["any_labeled"] = [
        set().union(*(result[name][i] for name in splits))
        for i in range(len(IDENTITY_SCHEMES))
    ]
    return result


def overlaps(sequence_a, sequence_b, members):
    left, right = protein_ids(sequence_a), protein_ids(sequence_b)
    return tuple(a in ids or b in ids for a, b, ids in zip(left, right, members))


def validity_reason(sequence):
    """Do not turn PTM markup, molecules, whitespace, or missing data into proteins.

    Validity uses the checkpoint's preserved ProtT5 normalization. All raw
    characters must be residue letters, and at least one canonical residue must
    remain in the actual truncated input. U/Z/O/B are converted to X as before.
    """
    if not isinstance(sequence, str) or not sequence:
        return "missing_sequence"
    if not re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWYXUZOB]+", sequence.upper()):
        return "unsupported_sequence_characters"
    normalized = prott5_normalize_sequence(sequence)
    if not re.search(r"[ACDEFGHIKLMNPQRSTVWY]", normalized):
        return "no_canonical_residue_in_model_input"
    return None


def unknown_chunks(path, chunk_size=5_000):
    """CSV-only, bounded row chunks; explicit labels must be unknown or absent."""
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    for frame in pd.read_csv(
        path, dtype=str, keep_default_na=False, chunksize=chunk_size
    ):
        aliases = {
            "Partner_A_sequence": A,
            "partner_a_sequence": A,
            "Partner_B_sequence": B,
            "partner_b_sequence": B,
        }
        for old, new in aliases.items():
            if old in frame and old != new:
                if new in frame:
                    raise ValueError(f"Ambiguous sequence columns: {old}, {new}")
                frame = frame.rename(columns={old: new})
        if not {A, B}.issubset(frame):
            raise ValueError(f"Unknown CSV requires {A} and {B}")
        if (
            "label" in frame
            and not frame.label.str.strip().str.lower().isin(["", "unknown"]).all()
        ):
            raise ValueError("Unknown input must not carry binary/numeric labels")
        for column in ("SNOOPPI_final_label", "source_label"):
            if (
                column in frame
                and not frame[column]
                .str.strip()
                .str.lower()
                .isin(["", "unknown"])
                .all()
            ):
                raise ValueError(f"{column} must be unknown or absent")
        reasons = []
        for _, row in frame.iterrows():
            reason = validity_reason(row[A]) or validity_reason(row[B])
            for side in ("A", "B"):
                for column in (
                    f"Partner_{side}_molecule_type",
                    f"partner_{side}_molecule_type",
                    f"Partner_{side}_mol_type",
                    f"partner_{side}_mol_type",
                ):
                    if column in frame and row[column].strip().lower() not in (
                        "",
                        "protein",
                        "polypeptide",
                        "polypeptide(l)",
                    ):
                        reason = reason or "non_protein_molecule_type"
            reasons.append(reason or "valid")
        frame["validity_reason"] = reasons
        yield frame


def audit_unknown(path, splits, cache=None, chunk_size=5_000):
    """Count observed rows, unique full pairs, overlap, and optional cache coverage."""
    index = labeled_protein_index(splits)
    counters = {"total_rows": 0, "valid_rows": 0, "invalid_rows": 0}
    counts = {(part, scheme): 0 for part in index for scheme in IDENTITY_SCHEMES}
    overlap_any = {part: 0 for part in index}
    seen_pairs, valid_proteins, missing = set(), set(), set()
    reasons = {}
    for chunk in unknown_chunks(path, chunk_size):
        counters["total_rows"] += len(chunk)
        for reason, count in chunk.validity_reason.value_counts().items():
            if reason != "valid":
                reasons[reason] = reasons.get(reason, 0) + int(count)
        valid = chunk[chunk.validity_reason.eq("valid")]
        counters["valid_rows"] += len(valid)
        counters["invalid_rows"] += len(chunk) - len(valid)
        for a, b in zip(valid[A], valid[B]):
            seen_pairs.add(full_pair_id(a, b))
            valid_proteins.update((a, b))
            if cache is not None:
                missing.update(s for s in (a, b) if s not in cache)
            for part, members in index.items():
                flags = overlaps(a, b, members)
                overlap_any[part] += int(any(flags))
                for scheme, flag in zip(IDENTITY_SCHEMES, flags):
                    counts[part, scheme] += int(flag)
    rows = [
        dict(
            metric=k, count=v, population="all_unknown_rows", note="observed CSV count"
        )
        for k, v in counters.items()
    ]
    rows += [
        dict(
            metric="unique_valid_full_pairs",
            count=len(seen_pairs),
            population="valid_rows",
            note="unordered full-sequence identity",
        ),
        dict(
            metric="duplicate_valid_full_pair_rows",
            count=counters["valid_rows"] - len(seen_pairs),
            population="valid_rows",
            note="duplicates retained during scoring; deduplicated during selection",
        ),
        dict(
            metric="unique_valid_raw_proteins",
            count=len(valid_proteins),
            population="valid_rows",
            note="original raw cache keys",
        ),
    ]
    rows += [
        dict(
            metric=f"invalid_{k}",
            count=v,
            population="all_unknown_rows",
            note="first rejection reason per row",
        )
        for k, v in sorted(reasons.items())
    ]
    rows += [
        dict(
            metric=f"overlap_{part}_{scheme}",
            count=v,
            population="valid_rows",
            note="either partner, either labeled role; row count",
        )
        for (part, scheme), v in counts.items()
    ]
    rows += [
        dict(
            metric=f"overlap_{part}_any_identity",
            count=v,
            population="valid_rows",
            note="union across three identity rules; row count",
        )
        for part, v in overlap_any.items()
    ]
    rows.append(
        dict(
            metric="valid_rows_without_any_labeled_protein_overlap",
            count=counters["valid_rows"] - overlap_any["any_labeled"],
            population="valid_rows",
            note="identity exclusion does not prove homology/cluster isolation",
        )
    )
    if cache is not None:
        rows.append(
            dict(
                metric="missing_raw_embedding_keys",
                count=len(missing),
                population="unique_valid_raw_proteins",
                note="must be zero before scoring",
            )
        )
    missing_frame = pd.DataFrame({"raw_sequence": sorted(missing)})
    return pd.DataFrame(rows), missing_frame


class ProtT5Predictor(nn.Module):
    """Exact original PPIClassifier architecture and state-dictionary key names."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(3072, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

    def forward(self, features):
        return self.network(features).squeeze(-1)


def load_predictor(path, split_record, provenance_reviewed=False, device="cpu"):
    """Accept original wrapped/bare MLP weights; never guess another architecture."""
    # Own trusted files only; legacy exports contain non-tensor metadata.
    payload = _load_trusted_torch_file(Path(path))
    wrapped = isinstance(payload, dict) and "model_state_dict" in payload
    state = payload["model_state_dict"] if wrapped else payload
    if wrapped:
        if payload.get("model_class", "PPIClassifier") not in (
            "PPIClassifier",
            "ProtT5Predictor",
        ):
            raise ValueError(
                "Checkpoint model_class is incompatible with original ProtT5 MLP"
            )
        if (
            payload.get("model_id", MODEL_ID) != MODEL_ID
            or payload.get("encoder", "ProtT5") != "ProtT5"
        ):
            raise ValueError("Checkpoint is not a ProtT5 predictor")
        if payload.get("normalization", NORMALIZATION) != NORMALIZATION:
            raise ValueError(
                "Checkpoint preprocessing differs from this scoring adapter"
            )
        operations = payload.get("feature_operations", FEATURE_OPERATIONS)
        if (
            not isinstance(operations, (list, tuple))
            or tuple(operations) != FEATURE_OPERATIONS
        ):
            raise ValueError(
                "Checkpoint pair-feature order differs from the original ProtT5 predictor"
            )
    model = ProtT5Predictor()
    expected = model.state_dict()
    if not isinstance(state, dict) or set(state) != set(expected):
        raise ValueError("Checkpoint keys do not match the original ProtT5 MLP")
    for key, tensor in state.items():
        if (
            not torch.is_tensor(tensor)
            or not torch.is_floating_point(tensor)
            or tensor.shape != expected[key].shape
            or not torch.isfinite(tensor).all()
        ):
            raise ValueError(f"Checkpoint tensor incompatible/nonfinite: {key}")
    recorded_split = payload.get("split_set_sha256") if wrapped else None
    if wrapped and "split_manifest" in payload:
        if not isinstance(payload["split_manifest"], dict):
            raise ValueError("Checkpoint split_manifest must be a dictionary")
        nested_split = payload["split_manifest"].get("split_set_sha256")
        if (
            recorded_split is not None
            and nested_split is not None
            and recorded_split != nested_split
        ):
            raise ValueError("Checkpoint split fingerprints contradict each other")
        recorded_split = recorded_split if recorded_split is not None else nested_split
    if (
        recorded_split is not None
        and recorded_split != split_record["split_set_sha256"]
    ):
        raise ValueError(
            "Checkpoint training split fingerprint differs from selected labeled CSVs"
        )
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False)
    model.to(device).eval()
    record = dict(
        path=str(Path(path).resolve()),
        sha256=file_sha256(path),
        architecture="original_symmetric_3072_512_512_256_1_MLP",
        training_split_fingerprint=recorded_split,
        training_provenance=(
            "fingerprint_matched"
            if recorded_split
            else "legacy_missing_split_fingerprint"
        ),
        provenance_reviewed=bool(provenance_reviewed),
        normalization=NORMALIZATION,
        model_id=MODEL_ID,
        feature_operations=list(FEATURE_OPERATIONS),
        checkpoint_format="model_state_dict_wrapper" if wrapped else "bare_state_dict",
    )
    return model, record


def load_mining_caches(paths):
    """Merge separately supplied existing caches, rejecting inconsistent vectors."""
    merged, records = {}, []
    for path in paths:
        cache, metadata = load_embedding_cache(Path(path), "ProtT5")
        if metadata.get("normalization", NORMALIZATION) != NORMALIZATION:
            raise ValueError(
                "Embedding cache preprocessing differs from the original ProtT5 predictor"
            )
        for sequence, vector in cache.items():
            if sequence in merged and not np.array_equal(merged[sequence], vector):
                raise ValueError(
                    "Different embeddings for the same raw key across caches"
                )
            merged[sequence] = vector
        records.append(
            dict(
                path=str(Path(path).resolve()),
                sha256=file_sha256(path),
                metadata=metadata,
            )
        )
    if not records:
        raise ValueError("At least one existing ProtT5 embedding cache is required")
    return merged, records


def input_record(unknown_path, split_record, checkpoint_record, cache_records):
    return dict(
        schema_version=1,
        unknown_csv=dict(
            path=str(Path(unknown_path).resolve()), sha256=file_sha256(unknown_path)
        ),
        splits=split_record,
        checkpoint=checkpoint_record,
        caches=cache_records,
        pair_id_scheme="unordered_full_uppercase_spaces_removed_sha256_v1",
        score_definition="sigmoid_of_frozen_ProtT5_logit_uncalibrated",
    )


def verify_current_inputs(record):
    """Refuse reuse when the unknown source, checkpoint, caches, or splits change."""
    entries = [record["unknown_csv"], record["checkpoint"], *record["caches"]]
    entries += [
        dict(path=path, sha256=record["splits"]["sha256"][name])
        for name, path in record["splits"]["files"].items()
    ]
    for entry in entries:
        if file_sha256(Path(entry["path"])) != entry["sha256"]:
            raise ValueError(
                f"Input changed; rerun mining with a new destination: {entry['path']}"
            )


@contextmanager
def new_artifacts(paths):
    """Stage files beside destinations, refuse overwrite, clean up failed writes."""
    paths = [Path(p) for p in paths]
    if len(set(paths)) != len(paths) or any(p.exists() for p in paths):
        raise FileExistsError(
            "Mining outputs already exist or destinations collide; use a new MINED_DIR"
        )
    staged, published = [], []
    try:
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                prefix=".mining-", dir=path.parent, delete=False
            ) as file:
                staged.append(Path(file.name))
        yield staged
        # Exclusive creation protects existing outputs, including races. Drive
        # does not reliably support hard links, so use exclusive byte copies.
        for source, target in zip(staged, paths):
            with target.open("xb") as output, source.open("rb") as input_stream:
                published.append(target)
                while block := input_stream.read(1 << 20):
                    output.write(block)
    except BaseException:
        for path in published:
            path.unlink(missing_ok=True)
        raise
    finally:
        for path in staged:
            path.unlink(missing_ok=True)


def write_audit(summary, missing, summary_path, missing_path=None):
    paths = [summary_path] + ([] if missing_path is None else [missing_path])
    with new_artifacts(paths) as staged:
        summary.to_csv(staged[0], index=False)
        if missing_path is not None:
            missing.to_csv(staged[1], index=False)


def score_unknown(
    unknown_path,
    model,
    cache,
    record,
    output_path,
    chunk_size=5_000,
    batch_size=512,
    device="cpu",
):
    """Write every valid source row once, including labeled-protein overlappers."""
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    verify_current_inputs(record)
    if Path(unknown_path).resolve() != Path(record["unknown_csv"]["path"]).resolve():
        raise ValueError("Unknown scoring source differs from its input record")
    if model.training or any(
        parameter.requires_grad for parameter in model.parameters()
    ):
        raise ValueError("Scorer must be frozen and in eval mode")
    output_path = Path(output_path)
    manifest_path = output_path.with_suffix(".manifest.json")
    total, valid_count = 0, 0
    with new_artifacts([output_path, manifest_path]) as staged:
        pd.DataFrame(columns=SCORE_COLUMNS).to_csv(staged[0], index=False)
        with torch.inference_mode():
            for chunk in unknown_chunks(unknown_path, chunk_size):
                total += len(chunk)
                valid = chunk[chunk.validity_reason.eq("valid")]
                for start in range(0, len(valid), batch_size):
                    batch = valid.iloc[start : start + batch_size]
                    features = torch.from_numpy(build_pair_features(batch, cache)).to(
                        device
                    )
                    if not torch.isfinite(features).all():
                        raise ValueError(
                            "Nonfinite pair features; review embedding values"
                        )
                    logits = model(features)
                    if not torch.isfinite(logits).all():
                        raise ValueError(
                            "Nonfinite predictor logits; scoring stopped before sigmoid"
                        )
                    scores = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
                    if len(scores) != len(batch) or not np.isfinite(scores).all():
                        raise ValueError("Invalid/nonfinite scorer output")
                    scored = pd.DataFrame(
                        {
                            "pair_id": [
                                full_pair_id(a, b) for a, b in zip(batch[A], batch[B])
                            ],
                            "Partner_A_sequence": batch[A].to_numpy(),
                            "Partner_B_sequence": batch[B].to_numpy(),
                            "score": scores,
                        }
                    )
                    scored.to_csv(staged[0], mode="a", header=False, index=False)
                    valid_count += len(batch)
        verify_current_inputs(record)
        manifest = dict(
            record,
            total_source_rows=total,
            valid_scored_rows=valid_count,
            invalid_source_rows=total - valid_count,
            score_file_sha256=file_sha256(staged[0]),
            selection_performed=False,
            training_performed=False,
        )
        staged[1].write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def read_verified_score_manifest(score_path, split_record):
    score_path = Path(score_path)
    record = json.loads(score_path.with_suffix(".manifest.json").read_text())
    verify_current_inputs(record)
    if record["splits"]["split_set_sha256"] != split_record["split_set_sha256"]:
        raise ValueError("Scores use different labeled partitions")
    if file_sha256(score_path) != record["score_file_sha256"]:
        raise ValueError("Score CSV has changed since scoring")
    return record


def select_candidates(
    score_path,
    splits,
    split_record,
    threshold=0.1,
    K=None,
    seed=44,
    provenance_reviewed=False,
    chunk_size=5_000,
):
    """Strict score cutoff AND exclusion of either labeled protein, in either role.

    K is an optional cap: lowest score first, deterministic pair-id tie breaking.
    Seed is retained for future training; this selection has no random sampling.
    """
    if (
        isinstance(threshold, bool)
        or not math.isfinite(threshold)
        or not 0 <= threshold <= 1
    ):
        raise ValueError("threshold must be finite and in [0, 1]")
    if K is not None and (type(K) is not int or K < 1):
        raise ValueError("K must be None (all eligible) or a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    record = read_verified_score_manifest(score_path, split_record)
    if not (provenance_reviewed or record["checkpoint"]["provenance_reviewed"]):
        raise ValueError(
            "Review checkpoint/cache identity and training split provenance before selecting candidates"
        )
    index = labeled_protein_index(splits)["any_labeled"]
    collected = []
    pair_scores = {}
    low_count, overlap_count, row_count = 0, 0, 0
    for chunk in pd.read_csv(score_path, chunksize=chunk_size, keep_default_na=False):
        if list(chunk.columns) != SCORE_COLUMNS:
            raise ValueError("Score CSV does not have the four expected columns")
        values = pd.to_numeric(chunk.score, errors="coerce")
        if not np.isfinite(values).all() or not values.between(0, 1).all():
            raise ValueError("Scores must be finite and in [0, 1]")
        chunk["score"] = values
        row_count += len(chunk)
        selected = []
        for row in chunk.itertuples(index=False):
            a, b = row.Partner_A_sequence, row.Partner_B_sequence
            if (
                validity_reason(a)
                or validity_reason(b)
                or full_pair_id(a, b) != row.pair_id
            ):
                raise ValueError("Score row sequence/identity is invalid")
            if (
                row.pair_id in pair_scores
                and abs(row.score - pair_scores[row.pair_id]) > 1e-6
            ):
                raise ValueError(
                    "Duplicate full pairs have inconsistent scores; review cache/preprocessing"
                )
            pair_scores.setdefault(row.pair_id, row.score)
            if row.score < threshold:
                low_count += 1
                if any(overlaps(a, b, index)):
                    overlap_count += 1
                else:
                    selected.append(row._asdict())
        if selected:
            collected.append(pd.DataFrame(selected))
    if row_count != record["valid_scored_rows"]:
        raise ValueError("Score row count differs from its manifest")
    eligible = (
        pd.concat(collected, ignore_index=True)
        if collected
        else pd.DataFrame(columns=SCORE_COLUMNS)
    )
    eligible = eligible.sort_values(
        ["score", "pair_id", "Partner_A_sequence", "Partner_B_sequence"], kind="stable"
    )
    eligible = eligible.drop_duplicates("pair_id", keep="first")
    unique_count = len(eligible)
    candidates = eligible.head(K).copy() if K is not None else eligible.copy()
    candidates["candidate_status"] = "unknown_negative_candidate_requires_review"
    candidates["label_evidence_source"] = "SNOOPPI_unknown_split_model_screen"
    candidates["negative_evidence"] = "none; unvalidated model score only"
    candidates["protein_overlap_any_labeled"] = False
    candidates["selection_threshold"] = threshold
    configuration = dict(
        schema_version=1,
        threshold=float(threshold),
        K=K,
        seed=seed,
        checkpoint_path=record["checkpoint"]["path"],
        checkpoint_sha256=record["checkpoint"]["sha256"],
        checkpoint_provenance_reviewed=True,
        split_set_sha256=split_record["split_set_sha256"],
        unknown_csv_sha256=record["unknown_csv"]["sha256"],
        score_csv_path=str(Path(score_path).resolve()),
        score_csv_sha256=record["score_file_sha256"],
        selection_rule="score < threshold AND no labeled protein overlap",
        overlap_identity_schemes=list(IDENTITY_SCHEMES),
        ranking="ascending score then full pair_id; no random sampling",
        low_score_rows=low_count,
        excluded_low_score_overlap_rows=overlap_count,
        eligible_unique_pairs=unique_count,
        selected_pairs=len(candidates),
        confidence_interval=None,
        calibration="not established",
        exponential_tilting="not applied; no fitted density ratio or target distribution",
        training_performed=False,
        binary_labels_assigned=False,
    )
    # Verify again after streaming, before allowing a candidate CSV to be saved.
    read_verified_score_manifest(score_path, split_record)
    return candidates[CANDIDATE_COLUMNS].reset_index(drop=True), configuration


def write_candidates(candidates, configuration, candidate_path, config_path):
    # JSON is a valid YAML 1.2 document: portable without a PyYAML dependency.
    # Writing JSON syntax intentionally preserves null K, booleans and exact paths.
    if len(candidates) != configuration["selected_pairs"] or "label" in candidates:
        raise ValueError("Candidate/config counts or unknown label contract differ")
    with new_artifacts([candidate_path, config_path]) as staged:
        candidates.to_csv(staged[0], index=False)
        payload = dict(
            configuration,
            candidate_csv_path=str(Path(candidate_path).resolve()),
            candidate_csv_sha256=file_sha256(staged[0]),
        )
        staged[1].write_text(json.dumps(payload, indent=2, sort_keys=True))


def cache_verdict(candidate_path, config_path, verdict_path, split_record):
    """Cache a review queue verdict, never a biological negative classification."""
    config = json.loads(Path(config_path).read_text())
    record = read_verified_score_manifest(config["score_csv_path"], split_record)
    if config["score_csv_sha256"] != record["score_file_sha256"]:
        raise ValueError("Selection config references stale scores")
    if file_sha256(candidate_path) != config["candidate_csv_sha256"]:
        raise ValueError("Candidate CSV changed after selection")
    candidates = pd.read_csv(candidate_path, keep_default_na=False)
    if (
        len(candidates) != config["selected_pairs"]
        or list(candidates.columns) != CANDIDATE_COLUMNS
    ):
        raise ValueError("Candidate cache schema/count differs from its config")
    if (
        config["split_set_sha256"] != split_record["split_set_sha256"]
        or config["checkpoint_sha256"] != record["checkpoint"]["sha256"]
    ):
        raise ValueError("Candidate config references different inputs")
    if (
        candidates.pair_id.duplicated().any()
        or not candidates.candidate_status.eq(
            "unknown_negative_candidate_requires_review"
        ).all()
    ):
        raise ValueError("Candidate cache must retain unique unknown review statuses")
    if not candidates.negative_evidence.eq("none; unvalidated model score only").all():
        raise ValueError(
            "No biological negative verdict can be inferred from the score"
        )
    if (
        not pd.to_numeric(candidates.score, errors="coerce")
        .lt(config["threshold"])
        .all()
    ):
        raise ValueError("Cached candidates do not pass the configured strict cutoff")
    payload = dict(
        schema_version=1,
        status="review_queue_ready" if len(candidates) else "no_eligible_candidates",
        selected_pairs=len(candidates),
        verdict="unknown; negative evidence not established",
        future_training_authorized=False,
        original_splits_modified=False,
        training_performed=False,
        binary_labels_assigned=False,
        inputs=record,
        configuration=config,
        candidate_csv_sha256=file_sha256(candidate_path),
        config_sha256=file_sha256(config_path),
        limitations=[
            "scores are uncalibrated",
            "identity exclusion does not ensure homology isolation",
            "legacy checkpoint/cache provenance requires external review",
            "no confidence interval from a single checkpoint",
        ],
    )
    with new_artifacts([verdict_path]) as staged:
        staged[0].write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload
