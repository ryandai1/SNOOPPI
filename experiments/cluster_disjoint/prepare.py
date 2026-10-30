"""Audit source CSVs, cluster full sequences, and freeze component-disjoint splits."""

import argparse, hashlib, json, re, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "drafting_code"))
from snooppi_utils import (
    normalize_split_columns,
    add_pair_ids,
    full_sequence_id,
    file_sha256,
    assert_model_ready,
    preprocessing_report,
)
from snooppi_protocol import mapped_component_frame, mapped_overlap_report

A, B = "partner_A_sequence", "partner_B_sequence"


def dump(obj, path):
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n")


def audit(inputs, out):
    frames = []
    for name in ("train", "validation", "test"):
        p = inputs / f"{name}.csv"
        f = normalize_split_columns(pd.read_csv(p), p)
        f["source_row"] = np.arange(len(f))
        f["original_split"] = name
        frames.append(f)
    f = add_pair_ids(pd.concat(frames, ignore_index=True))
    invalid = pd.Series(False, index=f.index)
    for c in (A, B):
        invalid |= (
            ~f[c]
            .str.upper()
            .str.replace(" ", "", regex=False)
            .str.fullmatch("[ACDEFGHIKLMNPQRSTVWYXBZOUJ]+")
        )
    bad = f[invalid].copy()
    bad["reason"] = "invalid_protein_sequence"
    f = f[~invalid].copy()
    conflict = pd.Series(False, index=f.index)
    for c in ("full_pair_id", "pair_id", "prott5_pair_id"):
        conflict |= f[c].isin(f.groupby(c).label.nunique().loc[lambda x: x > 1].index)
    conflicts = f[conflict].copy()
    conflicts["reason"] = "conflicting_labels_quarantined_without_relabeling"
    f = f[~conflict].copy()
    dup = f[f.full_pair_id.duplicated()].copy()
    dup["reason"] = "consistent_full_pair_duplicate"
    f = f.drop_duplicates("full_pair_id").copy()
    collision = f.pair_id.duplicated(False) | f.prott5_pair_id.duplicated(False)
    coll = f[collision].copy()
    coll["reason"] = "distinct_full_pairs_collide_after_preprocessing"
    f = f[~collision].sort_values("full_pair_id").reset_index(drop=True)
    pd.concat([bad, conflicts, dup, coll]).to_csv(out / "exclusions.csv", index=False)
    f.to_csv(out / "clean_pairs.csv", index=False)
    seqs = {
        full_sequence_id(s): s.upper().replace(" ", "") for c in (A, B) for s in f[c]
    }
    with (out / "proteins.fasta").open("w") as h:
        for k, s in sorted(seqs.items()):
            h.write(f">{k}\n{s}\n")
    record = {
        "source_rows": sum(map(len, frames)),
        "invalid_rows": len(bad),
        "conflict_rows": len(conflicts),
        "duplicate_rows": len(dup),
        "normalized_collision_rows": len(coll),
        "retained_rows": len(f),
        "unique_full_sequences": len(seqs),
        "class_counts": f.label.value_counts().to_dict(),
        "sources": {
            n: file_sha256(inputs / f"{n}.csv") for n in ("train", "validation", "test")
        },
    }
    dump(record, out / "audit.json")
    preprocessing = preprocessing_report({"all": f})
    preprocessing["max_full_sequence_length"] = max(map(len, seqs.values()))
    preprocessing["sequences_over_1024"] = sum(len(s) > 1024 for s in seqs.values())
    dump(preprocessing, out / "preprocessing.json")
    return f, seqs


def assign(f):
    # Fixed deterministic size/class-aware assignment; no model performance is consulted.
    counts = f.groupby("mapped_component").label.agg(["size", "sum"])
    counts["negative"] = counts["size"] - counts["sum"]
    target = (
        np.array([0.8, 0.1, 0.1])[:, None]
        * np.array([len(f), f.label.sum(), (f.label == 0).sum()])[None, :]
    )
    used = np.zeros((3, 3))
    allocation = {}
    for k, r in counts.sort_values(
        ["size", "negative"], ascending=False, kind="stable"
    ).iterrows():
        v = np.array([r["size"], r["sum"], r["negative"]])
        costs = []
        for j in range(3):
            trial = used.copy()
            trial[j] += v
            costs.append(np.sum((trial - target) ** 2 / np.maximum(target, 1)))
        j = int(np.argmin(costs))
        used[j] += v
        allocation[k] = ("train", "validation", "test")[j]
    parts = {
        n: f[f.mapped_component.map(allocation) == n].copy()
        for n in ("train", "validation", "test")
    }
    return parts, counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--mmseqs", required=True)
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    f, seqs = audit(args.inputs, args.out)
    cmd = [
        args.mmseqs,
        "easy-cluster",
        str(args.out / "proteins.fasta"),
        str(args.out / "mmseqs"),
        str(args.out / "tmp"),
        "--min-seq-id",
        "0.3",
        "-c",
        "0.8",
        "--cov-mode",
        "0",
        "--cluster-mode",
        "1",
        "--single-step-clustering",
        "1",
        "--max-seqs",
        "1000",
        "--seq-id-mode",
        "0",
        "-s",
        "7.5",
        "--threads",
        str(args.threads),
    ]
    dump(
        {
            "command": cmd,
            "version": subprocess.check_output(
                [args.mmseqs, "version"], text=True
            ).strip(),
            "binary_sha256": file_sha256(Path(args.mmseqs)),
        },
        args.out / "clustering.json",
    )
    with (args.out / "mmseqs.log").open("w") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    m = pd.read_csv(
        args.out / "mmseqs_cluster.tsv",
        sep="\t",
        names=["cluster_id", "sequence_sha256"],
    )
    if m.sequence_sha256.duplicated().any() or set(m.sequence_sha256) != set(seqs):
        raise ValueError("Incomplete or ambiguous cluster mapping")
    m["identity_scheme"] = "full_uppercase_spaces_removed_sha256_v1"
    m["evidence_source"] = "clustering.json; mmseqs_cluster.tsv"
    m.to_csv(args.out / "mmseqs2_clusters.csv", index=False)
    from finish import finish

    finish(args.out)


if __name__ == "__main__":
    main()
