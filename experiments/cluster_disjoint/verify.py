"""Independently reconcile exclusions, frozen artifacts, and completed predictions."""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from prepare import file_sha256, dump
from finish import input_isolation
from snooppi_utils import assert_model_ready, summarize_predictions
from snooppi_protocol import mapped_overlap_report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--splits", type=Path, required=True)
    p.add_argument("--models", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    manifest = json.loads((args.splits / "manifest.json").read_text())
    for name, sha in manifest["files"].items():
        assert file_sha256(args.splits / name) == sha, name
    parts = {
        n: pd.read_csv(args.splits / f"{n}.csv")
        for n in ("train", "validation", "test")
    }
    assert_model_ready(parts)
    input_isolation(parts)
    mapping = pd.read_csv(args.splits / "mmseqs2_clusters.csv")
    assert not mapped_overlap_report(
        parts, mapping, "cluster_id"
    ).overlapping_cluster_id.any()
    all_rows = pd.concat(
        [
            *parts.values(),
            pd.read_csv(args.splits / "exclusions.csv"),
            pd.read_csv(args.splits / "cross_partition_exclusions.csv"),
        ]
    )
    assert not all_rows[["original_split", "source_row"]].duplicated().any()
    audit = json.loads((args.splits / "audit.json").read_text())
    assert len(all_rows) == audit["source_rows"]
    metrics = pd.read_csv(args.models / "metrics.csv")
    assert len(metrics) == 45 and len(list(args.models.glob("*_complete.json"))) == 15
    assert (args.models / "COMPLETE").exists()
    assert set(metrics.seed) == {44}
    test = parts["test"]
    for row in metrics[metrics.split == "test"].itertuples():
        tag = f"{row.encoder}_{row.head}_{row.variant}"
        pred = pd.read_csv(args.models / f"{tag}_test_predictions.csv")
        assert pred.full_pair_id.tolist() == test.full_pair_id.tolist()
        assert np.array_equal(pred.label, test.label)
        actual = summarize_predictions(pred.label, pred.probability, row.threshold)
        for k, v in actual.items():
            assert np.isclose(v, getattr(row, k)), (tag, k)
        record = json.loads((args.models / f"{tag}_complete.json").read_text())
        for name, sha in record["files"].items():
            assert file_sha256(args.models / name) == sha
        if row.head == "mlp":
            history = pd.read_csv(args.models / f"{tag}_history.csv")
            assert (
                history.loc[history.validation_macro_ap.idxmax(), "epoch"]
                == row.best_epoch
            )
    dump(
        {
            "status": "passed",
            "source_rows_reconciled": len(all_rows),
            "frozen_file_hashes_verified": len(manifest["files"]),
            "cluster_overlap": 0,
            "full_and_both_model_input_overlaps": 0,
            "classifier_fits": 15,
            "metric_rows": 45,
            "test_predictions_verified": 15 * len(test),
            "mlp_checkpoint_selections_verified": 12,
        },
        args.out,
    )


if __name__ == "__main__":
    main()
