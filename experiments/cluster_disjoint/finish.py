"""Finish a completed clustering; use deterministic edge exclusions if components fail."""

import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from prepare import assign, dump, A, B
from snooppi_utils import file_sha256, full_sequence_id, assert_model_ready
from snooppi_protocol import UnionFind, mapped_component_frame, mapped_overlap_report


def edge_partition(f, m):
    # Merge clusters sharing an actual model input, but do not merge interaction edges.
    uf = UnionFind()
    lookup = m.set_index("sequence_sha256").cluster_id.to_dict()
    for row in f.itertuples():
        for side in ("A", "B"):
            group = (
                "cluster",
                lookup[full_sequence_id(getattr(row, "partner_" + side + "_sequence"))],
            )
            uf.union(group, ("esm", getattr(row, "protein_" + side + "_id")))
            uf.union(group, ("t5", getattr(row, "prott5_protein_" + side + "_id")))
    roots = sorted({str(uf.find(("cluster", v))) for v in lookup.values()})
    index = {v: i for i, v in enumerate(roots)}
    ids = {s: index[str(uf.find(("cluster", v)))] for s, v in lookup.items()}
    a = f[A].map(full_sequence_id).map(ids).to_numpy()
    b = f[B].map(full_sequence_id).map(ids).to_numpy()
    y = f.label.to_numpy()
    rng = np.random.default_rng(44)
    probs = np.sqrt([0.8, 0.1, 0.1])
    probs /= probs.sum()
    best = None
    for trial in range(256):
        groups = rng.choice(3, size=len(roots), p=probs)
        keep = groups[a] == groups[b]
        spl = groups[a]
        counts = np.array(
            [
                [(keep & (spl == j) & (y == label)).sum() for label in (0, 1)]
                for j in range(3)
            ]
        )
        if counts.min() < 50:
            continue
        fractions = counts.sum(axis=1) / counts.sum()
        if np.max(abs(fractions - [0.8, 0.1, 0.1])) > 0.05:
            continue
        target = np.array([0.8, 0.1, 0.1])[:, None] * counts.sum(axis=0)
        score = np.sum(((counts - target) / np.maximum(target, 1)) ** 2) + 0.1 * (
            1 - keep.mean()
        )
        if best is None or score < best[0]:
            best = (score, trial, keep, spl, groups)
    if best is None:
        raise ValueError(
            "No feasible cluster partition in 256 prespecified assignments"
        )
    _, trial, keep, spl, groups = best
    parts = {
        n: f[keep & (spl == j)].copy()
        for j, n in enumerate(("train", "validation", "test"))
    }
    excluded = f[~keep].copy()
    excluded["reason"] = "partners_assigned_to_different_partitions"
    assignment = pd.DataFrame(
        {
            "sequence_sha256": list(ids),
            "partition": [
                ("train", "validation", "test")[groups[v]] for v in ids.values()
            ],
        }
    )
    return parts, excluded, assignment, trial


def input_isolation(parts):
    from snooppi_utils import sequence_id, prott5_normalize_sequence

    rows = []
    for rule, fn in [
        ("full", full_sequence_id),
        ("esm", sequence_id),
        ("prott5", prott5_normalize_sequence),
    ]:
        keys = {n: {fn(s) for c in (A, B) for s in f[c]} for n, f in parts.items()}
        for i, left in enumerate(keys):
            for right in list(keys)[i + 1 :]:
                rows.append(
                    {
                        "rule": rule,
                        "left": left,
                        "right": right,
                        "overlap": len(keys[left] & keys[right]),
                    }
                )
    result = pd.DataFrame(rows)
    if result.overlap.any():
        raise ValueError("Cross-partition full or model-input protein identity")
    return result


def finish(out):
    if (out / "manifest.json").exists():
        raise FileExistsError("Frozen manifest already exists")
    f = pd.read_csv(out / "clean_pairs.csv")
    m = pd.read_csv(out / "mmseqs2_clusters.csv")
    expected = {full_sequence_id(s) for c in (A, B) for s in f[c]}
    if (
        m.sequence_sha256.duplicated().any()
        or set(m.sequence_sha256) != expected
        or m.cluster_id.isna().any()
    ):
        raise ValueError("Invalid mapping coverage")
    f = mapped_component_frame(f, m, "cluster_id")
    parts, counts = assign(f)
    counts.sort_values("size", ascending=False).to_csv(out / "components.csv")
    component_summary = {
        n: {"rows": len(v), "negative": int((v.label == 0).sum())}
        for n, v in parts.items()
    }
    dump(component_summary, out / "component_attempt.json")
    feasible = all(
        set(v.label) == {0, 1} and abs(len(v) / len(f) - t) <= 0.05
        for v, t in zip(parts.values(), [0.8, 0.1, 0.1])
    )
    protocol = "interaction_components"
    trial = None
    excluded = pd.DataFrame()
    if not feasible:
        parts, excluded, assignment, trial = edge_partition(f, m)
        protocol = "cluster_assignment_with_cross_partition_edges_excluded"
        excluded.to_csv(out / "cross_partition_exclusions.csv", index=False)
        assignment.to_csv(out / "protein_partition.csv", index=False)
    assert_model_ready(parts)
    input_isolation(parts).to_csv(out / "input_overlap.csv", index=False)
    overlap = mapped_overlap_report(parts, m, "cluster_id")
    if overlap.overlapping_cluster_id.any():
        raise ValueError("Cluster leakage")
    overlap.to_csv(out / "overlap.csv", index=False)
    summary = {
        "protocol": protocol,
        "assignment_trial": trial,
        "assignment_seed": 44,
        "candidate_assignments": 256 if trial is not None else 0,
        "clean_rows": len(f),
        "retained_rows": sum(map(len, parts.values())),
        "cross_partition_exclusions": len(excluded),
        "clusters": m.cluster_id.nunique(),
        "interaction_components": len(counts),
        "largest_component": int(counts["size"].max()),
        "splits": {
            n: {
                "rows": len(v),
                "positive": int(v.label.sum()),
                "negative": int((v.label == 0).sum()),
            }
            for n, v in parts.items()
        },
    }
    dump(summary, out / "feasibility.json")
    for n, v in parts.items():
        v.to_csv(out / f"{n}.csv", index=False)
    dump(
        {
            "status": "frozen",
            "protocol": protocol,
            "files": {
                p.name: file_sha256(p)
                for p in out.iterdir()
                if p.is_file() and p.suffix in (".csv", ".fasta", ".json", ".tsv")
            },
        },
        out / "manifest.json",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    finish(p.parse_args().out)
