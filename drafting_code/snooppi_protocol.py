"""Split/candidate design helpers. Never train on unknown candidates or overwrite inputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from snooppi_utils import (
    LABEL,
    SEQUENCE_A,
    SEQUENCE_B,
    SPLIT_NAMES,
    add_pair_ids,
    assert_model_ready,
    full_sequence_id,
    sequence_id,
    canonical_pair_id,
    full_pair_id,
    file_sha256,
    split_manifest,
    save_json,
)

IDENTITY_SCHEME = "full_uppercase_spaces_removed_sha256_v1"
MATCH_COLUMNS = ["species", "localization", "family", "expression_bin"]


class UnionFind:
    """Iterative path compression avoids recursion failure on large graphs."""

    def __init__(self):
        self.parent = {}
        self.size = {}

    def find(self, value):
        self.parent.setdefault(value, value)
        self.size.setdefault(value, 1)
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b:
            return
        if self.size[a] < self.size[b]:
            a, b = b, a
        self.parent[b] = a
        self.size[a] += self.size[b]


def prepare_source_pairs(splits):
    pairs = add_pair_ids(
        pd.concat(
            [frame.assign(original_split=name) for name, frame in splits.items()],
            ignore_index=True,
        )
    )
    duplicate_mask = pd.Series(False, index=pairs.index)
    for id_col in ("pair_id", "prott5_pair_id"):
        conflicts = pairs.groupby(id_col)[LABEL].nunique().gt(1)
        if conflicts.any():
            raise ValueError(
                f"{int(conflicts.sum())} {id_col} pairs have conflicting labels; review evidence before re-splitting"
            )
        duplicate_mask |= pairs[id_col].duplicated(keep=False)
    duplicated = pairs.loc[
        duplicate_mask,
        [
            "pair_id",
            "full_pair_id",
            "prott5_pair_id",
            "original_split",
            LABEL,
            "label_evidence_source",
        ],
    ].copy()
    # Conservative duplicate/collision quarantine: do not choose an arbitrary raw variant.
    pairs = pairs.loc[~duplicate_mask].reset_index(drop=True)
    if pairs.empty:
        raise ValueError("No unique unconflicted pairs remain")
    return pairs, duplicated


def grouped_three_way_split(frame, group_column, seed=44, fractions=None):
    fractions = fractions or {"train": 0.8, "validation": 0.1, "test": 0.1}
    if (
        set(fractions) != set(SPLIT_NAMES)
        or any(v <= 0 for v in fractions.values())
        or not np.isclose(sum(fractions.values()), 1)
    ):
        raise ValueError(
            "Fractions must be positive train/validation/test values summing to 1"
        )
    if frame[group_column].isna().any() or frame[group_column].nunique() < 3:
        raise ValueError(
            f"{group_column} needs at least three non-missing groups; giant components can make strict splitting impossible"
        )
    outer = GroupShuffleSplit(
        n_splits=1,
        test_size=fractions["validation"] + fractions["test"],
        random_state=seed,
    )
    train_idx, held_idx = next(outer.split(frame, groups=frame[group_column]))
    held = frame.iloc[held_idx]
    if held[group_column].nunique() < 2:
        raise ValueError(
            "Held-out set has fewer than two groups; review group sizes and prespecified split fractions"
        )
    inner = GroupShuffleSplit(
        n_splits=1,
        test_size=fractions["test"] / (fractions["validation"] + fractions["test"]),
        random_state=seed,
    )
    val_idx, test_idx = next(inner.split(held, groups=held[group_column]))
    parts = {
        "train": frame.iloc[train_idx].copy(),
        "validation": held.iloc[val_idx].copy(),
        "test": held.iloc[test_idx].copy(),
    }
    for name, part in parts.items():
        if set(part[LABEL].unique()) != {0, 1}:
            raise ValueError(
                f"{name} lacks both classes for {group_column}; review design before modeling"
            )
        parts[name] = part.reset_index(drop=True)
    groups = {n: set(p[group_column]) for n, p in parts.items()}
    if any(
        groups[a] & groups[b]
        for i, a in enumerate(SPLIT_NAMES)
        for b in SPLIT_NAMES[i + 1 :]
    ):
        raise ValueError("Grouping unit crosses split boundaries")
    return parts


def interaction_components(frame, left="protein_A_id", right="protein_B_id"):
    components = UnionFind()
    for a, b in zip(frame[left], frame[right]):
        components.union(a, b)
    if left == "protein_A_id":
        for side in ("A", "B"):
            for esm, t5 in zip(
                frame[f"protein_{side}_id"], frame[f"prott5_protein_{side}_id"]
            ):
                components.union(esm, ("prott5", t5))
    return frame[left].map(components.find)


def cold_target_split(frame, target_side=SEQUENCE_B, seed=44, fractions=None):
    if target_side not in (SEQUENCE_A, SEQUENCE_B):
        raise ValueError("TARGET_SIDE must be a partner sequence column")
    target_col = "protein_A_id" if target_side == SEQUENCE_A else "protein_B_id"
    parts = grouped_three_way_split(frame, target_col, seed, fractions)
    # A held-out target must also be absent in the other partner role of earlier partitions.
    held_test = set(parts["test"][target_col])
    held_val = set(parts["validation"][target_col])
    t5_target_col = (
        "prott5_protein_A_id" if target_side == SEQUENCE_A else "prott5_protein_B_id"
    )
    t5_test = set(parts["test"][t5_target_col])
    t5_val = set(parts["validation"][t5_target_col])
    rejected = []
    for name, forbidden, t5_forbidden in (
        ("train", held_test | held_val, t5_test | t5_val),
        ("validation", held_test, t5_test),
    ):
        bad = parts[name].protein_A_id.isin(forbidden) | parts[name].protein_B_id.isin(
            forbidden
        )
        bad |= parts[name].prott5_protein_A_id.isin(t5_forbidden) | parts[
            name
        ].prott5_protein_B_id.isin(t5_forbidden)
        rejected.append(
            parts[name]
            .loc[bad]
            .assign(exclusion_reason="held_out_target_seen_in_other_role")
        )
        parts[name] = parts[name].loc[~bad].reset_index(drop=True)
        if set(parts[name][LABEL].unique()) != {0, 1}:
            raise ValueError(
                f"{name} lacks both classes after cross-role target exclusions"
            )
    return parts, pd.concat(rejected, ignore_index=True)


def load_reviewed_mapping(path, field, frame):
    """Full-sequence hashes, explicit identity scheme and source provenance are mandatory."""
    path = Path(path)
    table = pd.read_csv(path)
    required = {"sequence_sha256", field, "identity_scheme", "evidence_source"}
    if required - set(table):
        raise ValueError(
            f"{path} requires {sorted(required)}. MMseqs2 TSV IDs must first be mapped to full sequences."
        )
    if (
        table[list(required)].isna().any().any()
        or table[list(required)]
        .astype(str)
        .apply(lambda c: c.str.strip().eq("").any())
        .any()
    ):
        raise ValueError(f"{path} has missing mapping values/provenance")
    if not table.identity_scheme.eq(IDENTITY_SCHEME).all():
        raise ValueError(
            f"{path} identity_scheme must be {IDENTITY_SCHEME}; normalized/truncated hashes are ambiguous"
        )
    if table.sequence_sha256.duplicated().any():
        raise ValueError(f"{path} must have one reviewed row per full sequence hash")
    if not table.sequence_sha256.astype(str).str.fullmatch(r"[0-9a-f]{64}").all():
        raise ValueError(f"{path} has invalid SHA-256 identifiers")
    required_ids = {
        full_sequence_id(s) for c in (SEQUENCE_A, SEQUENCE_B) for s in frame[c]
    }
    if required_ids - set(table.sequence_sha256):
        raise ValueError(
            f"{path} misses {len(required_ids - set(table.sequence_sha256))} protein mappings"
        )
    return table


def mapped_component_frame(frame, mapping, field):
    result = frame.copy()
    lookup = mapping.set_index("sequence_sha256")[field]
    for side, col in (("A", SEQUENCE_A), ("B", SEQUENCE_B)):
        result[f"{side}_{field}"] = result[col].map(full_sequence_id).map(lookup)
    # Also link normalized/truncated identical inputs, even when the reviewed full sequences
    # have different families/clusters, to avoid a prefix collision leaking across partitions.
    graph = UnionFind()
    for row in result.itertuples(index=False):
        a, b = getattr(row, f"A_{field}"), getattr(row, f"B_{field}")
        pa, pb = row.protein_A_id, row.protein_B_id
        graph.union(("group", str(a)), ("group", str(b)))
        graph.union(("protein", pa), ("group", str(a)))
        graph.union(("protein", pb), ("group", str(b)))
        graph.union(("prott5", row.prott5_protein_A_id), ("group", str(a)))
        graph.union(("prott5", row.prott5_protein_B_id), ("group", str(b)))
    result["mapped_component"] = result[f"A_{field}"].map(
        lambda g: str(graph.find(("group", str(g))))
    )
    return result


def mapped_overlap_report(splits, mapping, field):
    lookup = mapping.set_index("sequence_sha256")[field]
    groups = {
        name: {
            lookup[full_sequence_id(s)]
            for col in (SEQUENCE_A, SEQUENCE_B)
            for s in frame[col]
        }
        for name, frame in splits.items()
    }
    return pd.DataFrame(
        [
            {"left": a, "right": b, f"overlapping_{field}": len(groups[a] & groups[b])}
            for i, a in enumerate(SPLIT_NAMES)
            for b in SPLIT_NAMES[i + 1 :]
        ]
    )


def summarize_protocol(name, parts, group_column):
    return pd.DataFrame(
        [
            {
                "protocol": name,
                "split": n,
                "pairs": len(p),
                "groups": p[group_column].nunique(),
                "positives": int(p[LABEL].sum()),
                "negatives": int((p[LABEL] == 0).sum()),
                "positive_prevalence": p[LABEL].mean(),
                "actual_row_fraction": len(p) / sum(map(len, parts.values())),
            }
            for n, p in parts.items()
        ]
    )


def write_protocol(destination, parts, source_manifest, config, review_tables=None):
    destination = Path(destination).resolve()
    source_dir = Path(source_manifest["split_directory"]).resolve()
    if (
        destination == source_dir
        or source_dir in destination.parents
        or destination.name == "splits"
        or destination.parent.name != "splits_ablations"
    ):
        raise ValueError(
            "Strict split destinations must be a new direct child of splits_ablations, separate from source splits"
        )
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    for name in review_tables or {}:
        if Path(name).name != name or not name.endswith(".csv"):
            raise ValueError("Review table filenames must be plain CSV basenames")
    assert_model_ready(parts)
    destination.mkdir(parents=True)
    for name, part in parts.items():
        # Preserve identities, original partition and label evidence for inspection.
        part.to_csv(destination / f"{name}.csv", index=False)
    target_paths = {"splits": destination}
    manifest = split_manifest(target_paths)
    manifest.update(
        source_manifest=source_manifest,
        protocol=config,
        summaries=summarize_protocol(
            config["name"], parts, config["group_column"]
        ).to_dict("records"),
    )
    for name, table in (review_tables or {}).items():
        table.to_csv(destination / name, index=False)
    save_json(manifest, destination / "manifest.json")
    return manifest


def matched_candidate_review(
    candidates, metadata, training_pairs, all_labeled_pairs, evidence_source
):
    """Training-only strata, either partner orientation, unknown queue without binary labels."""
    candidates = candidates.rename(
        columns={
            "Partner_A_sequence": SEQUENCE_A,
            "Partner_B_sequence": SEQUENCE_B,
            "partner_a_sequence": SEQUENCE_A,
            "partner_b_sequence": SEQUENCE_B,
        }
    ).copy()
    if candidates.columns.duplicated().any() or not {SEQUENCE_A, SEQUENCE_B}.issubset(
        candidates
    ):
        raise ValueError("Candidate sequences missing or ambiguous")
    for col in (SEQUENCE_A, SEQUENCE_B):
        if (
            not candidates[col]
            .map(lambda s: isinstance(s, str) and bool(s.strip()))
            .all()
        ):
            raise ValueError("Candidate sequences must be nonempty strings")
    for col in (LABEL, "SNOOPPI_final_label"):
        if col in candidates and candidates[col].notna().any():
            if (
                not candidates[col]
                .dropna()
                .astype(str)
                .str.strip()
                .str.lower()
                .eq("unknown")
                .all()
            ):
                raise ValueError(
                    "Candidate input includes confirmed/numeric labels; use the labeled dataset separately"
                )
    candidates = candidates.drop(
        columns=[c for c in (LABEL, "SNOOPPI_final_label") if c in candidates]
    )
    candidates = add_pair_ids(candidates)
    known_ids = {
        canonical_pair_id(a, b)
        for a, b in zip(all_labeled_pairs[SEQUENCE_A], all_labeled_pairs[SEQUENCE_B])
    }
    candidate_input_count = len(candidates)
    known_mask = candidates.pair_id.isin(known_ids)
    candidates = candidates.loc[~known_mask].drop_duplicates("pair_id").copy()
    excluded_count = candidate_input_count - len(candidates)
    available = [c for c in MATCH_COLUMNS if c in metadata]
    if not available:
        raise ValueError("No reviewed biological matching strata available")
    if not {"sequence_sha256", "identity_scheme", "evidence_source"}.issubset(metadata):
        raise ValueError(
            "Matching metadata requires full sequence hashes, identity_scheme and evidence_source"
        )
    if (
        metadata.sequence_sha256.duplicated().any()
        or not metadata.identity_scheme.eq(IDENTITY_SCHEME).all()
    ):
        raise ValueError("Ambiguous matching metadata identity")
    if (
        metadata[["sequence_sha256", "evidence_source"]].isna().any().any()
        or metadata.evidence_source.astype(str).str.strip().eq("").any()
    ):
        raise ValueError("Metadata provenance is missing")
    lookup = metadata.set_index("sequence_sha256")

    def strata(frame):
        values = []
        for a, b in zip(frame[SEQUENCE_A], frame[SEQUENCE_B]):
            ids = [full_sequence_id(a), full_sequence_id(b)]
            if any(s not in lookup.index for s in ids):
                values.append(None)
                continue
            sides = [lookup.loc[s, available] for s in ids]
            if any(
                side.isna().any() or side.astype(str).str.strip().eq("").any()
                for side in sides
            ):
                values.append(None)
                continue
            values.append(json.dumps(sorted([list(map(str, side)) for side in sides])))
        return pd.Series(values, index=frame.index, dtype=object)

    positives = training_pairs.loc[training_pairs[LABEL] == 1].copy()
    positives["match_stratum"] = strata(positives)
    reference = set(positives.match_stratum.dropna())
    candidates["match_stratum"] = strata(candidates)
    matched = candidates.loc[candidates.match_stratum.isin(reference)].copy()
    matched["candidate_status"] = "unknown_requires_review"
    matched["candidate_evidence_source"] = evidence_source
    matched["matched_on"] = ",".join(available)
    matched["matching_reference"] = "selected_training_positives_only"
    matched["negative_evidence_source"] = "unreviewed_no_negative_evidence"
    return matched.reset_index(drop=True), {
        "input_candidates": candidate_input_count,
        "excluded_known_or_duplicate": excluded_count,
        "incomplete_metadata": int(candidates.match_stratum.isna().sum()),
        "eligible_unlabeled_candidates": len(candidates),
        "matched_candidates": len(matched),
    }
