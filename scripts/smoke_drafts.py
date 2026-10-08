"""Execute all drafts on synthetic data only; writes only under work/.

Run from the repo root. This does not test Google Drive or research results.
"""

import json
import os
import runpy
import sys
import tempfile
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "drafting_code"))
from snooppi_utils import full_pair_id
from snooppi_utils import (
    load_splits,
    assert_model_ready,
    cache_paths,
    load_embedding_cache,
    validate_cache_coverage,
    prediction_identity,
    split_manifest,
    validate_prediction_identity,
)

work = ROOT / "work"
work.mkdir(exist_ok=True)
smoke_root = Path(tempfile.mkdtemp(prefix="draft-smoke-", dir=work))
fixture = runpy.run_path(str(ROOT / "tests" / "test_draft_protocol.py"))
paths = fixture["make_fixture"](smoke_root)
source_pairs = fixture["synthetic_pairs"]()
sequence = fixture["sequence"]
metadata = pd.read_csv(paths["metadata"])
for i in (400, 401):
    row = metadata.iloc[0].copy()
    row["sequence_sha256"] = fixture["full_sequence_id"](sequence(i))
    metadata = pd.concat([metadata, pd.DataFrame([row])], ignore_index=True)
metadata.to_csv(paths["metadata"], index=False)
pd.DataFrame(
    {
        "partner_A_sequence": [sequence(400)],
        "partner_B_sequence": [sequence(401)],
        "SNOOPPI_final_label": ["unknown"],
    }
).to_csv(paths["candidate_negatives"], index=False)

mutations, interfaces = [], []
for row in source_pairs.iloc[180:184].itertuples(index=False):
    a, b = row.partner_A_sequence, row.partner_B_sequence
    pair = full_pair_id(a, b)
    for partner, seq in [("A", a), ("B", b)]:
        for position in (1, 6):
            wt = seq[position - 1]
            mutations.append(
                {
                    "pair_id": pair,
                    "partner": partner,
                    "position_1based": position,
                    "wild_type": wt,
                    "mutant": "C" if wt != "C" else "A",
                    "sequence_A": a,
                    "sequence_B": b,
                    "evidence_source": "synthetic_test_only",
                }
            )
            interfaces.append(
                {
                    "pair_id": pair,
                    "partner": partner,
                    "position_1based": position,
                    "is_interface": int(position == 1),
                    "evidence_source": "synthetic_test_only",
                }
            )
pd.DataFrame(mutations).to_csv(
    smoke_root / "raw" / "mutation_benchmark.csv", index=False
)
pd.DataFrame(interfaces).to_csv(
    smoke_root / "raw" / "interface_annotations.csv", index=False
)
mock_checkpoint = smoke_root / "synthetic_scorer.json"
mock_checkpoint.write_text(
    json.dumps({"scorer": "synthetic amino acid count; no scientific interpretation"})
)

# These variables affect only this process and its synthetic notebook kernels.
os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
os.environ["SNOOPPI_ROOT"] = str(smoke_root)
os.environ["SNOOPPI_REPO_ROOT"] = str(ROOT)
os.environ["SNOOPPI_SPLIT_DIR"] = "splits"
os.environ.pop("SNOOPPI_CLUSTER_FILE", None)
os.environ["MPLCONFIGDIR"] = str(smoke_root / "mplconfig")
os.environ["IPYTHONDIR"] = str(smoke_root / "ipython")
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
executed_dir = smoke_root / "executed"
executed_dir.mkdir()
records = []
for path in sorted((ROOT / "drafting_code").glob("*.ipynb")):
    if path.name == "07_mine_unknown_negatives.ipynb":
        # The inference-only mining workflow has independent source/checkpoint
        # fixtures and is executed by scripts/smoke_mining.py.
        continue
    nb = nbformat.read(path, 4)
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        s = cell.source.replace("WRITE_OUTPUTS = False", "WRITE_OUTPUTS = True")
        s = s.replace("WRITE_SPLITS = False", "WRITE_SPLITS = True")
        s = s.replace(
            "WRITE_NEGATIVE_CANDIDATES = False", "WRITE_NEGATIVE_CANDIDATES = True"
        )
        s = s.replace(
            'PROTOCOLS_TO_PREVIEW = ["cold_target", "cold_protein"]',
            'PROTOCOLS_TO_PREVIEW = ["cold_target", "cold_protein", "cold_cluster", "cold_family"]',
        )
        s = s.replace(
            "INCLUDE_DEVELOPMENT_RUNS = False", "INCLUDE_DEVELOPMENT_RUNS = True"
        )
        s = s.replace(
            "BOOTSTRAP_REPLICATES = 500 if QUICK_MODE else 2_000",
            "BOOTSTRAP_REPLICATES = 20  # Synthetic smoke only",
        )
        # Exercise paired comparisons as well as intervals.
        s = s.replace(
            'available_runs = sorted(predictions["run_id"].unique())',
            'available_runs = sorted(predictions["run_id"].unique())\nREFERENCE_RUN = available_runs[0]',
        )
        s = s.replace("TRAIN_MODEL = False", "TRAIN_MODEL = True")
        s = s.replace(
            "CHECKPOINT_PATH = None", f"CHECKPOINT_PATH = {str(mock_checkpoint)!r}"
        )
        s = s.replace("RUN_SCORING = False", "RUN_SCORING = True")
        s = s.replace(
            'raise NotImplementedError("Connect the frozen model checkpoint before running mutation scoring")',
            'return np.array([1 / (1 + np.exp(-(a.count("C") + b.count("C") - 4))) for a, b in sequence_pairs])',
        )
        cell.source = s
    print("EXECUTE", path.name, flush=True)
    client = NotebookClient(
        nb,
        timeout=180,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
        record_timing=True,
    )
    client.execute()
    nbformat.write(nb, executed_dir / path.name)
    # Keep a portable HTML view for manual review, without changing source notebooks.
    from nbconvert import HTMLExporter

    html, _ = HTMLExporter().from_notebook_node(nb)
    (executed_dir / path.with_suffix(".html").name).write_text(html)
    figure_count = 0
    for i, cell in enumerate(nb.cells):
        for output in cell.get("outputs", []):
            data = output.get("data", {})
            if "image/png" in data:
                import base64

                (executed_dir / f"{path.stem}_cell{i}.png").write_bytes(
                    base64.b64decode(data["image/png"])
                )
                figure_count += 1
    records.append(
        {
            "notebook": path.name,
            "executed": True,
            "figures": figure_count,
            "data": "synthetic fixtures only; not scientific results",
        }
    )
    print("PASS", path.name, flush=True)
for manifest_path in sorted((smoke_root / "splits_ablations").glob("*/manifest.json")):
    selected_paths = dict(paths, splits=manifest_path.parent)
    selected_splits = load_splits(selected_paths)
    assert_model_ready(selected_splits)
    for encoder, cache_path in cache_paths(paths).items():
        cache, _ = load_embedding_cache(cache_path, encoder)
        validate_cache_coverage(cache, selected_splits)
    records.append(
        {"strict_export": manifest_path.parent.name, "reload_and_cache_coverage": True}
    )

expected = prediction_identity(load_splits(paths)["test"], split_manifest(paths))
for prediction_path in sorted(paths["tables"].glob("*_test_predictions.csv")):
    predictions = pd.read_csv(prediction_path)
    identity_columns = [
        c
        for c in ("encoder", "model", "ablation", "experiment", "seed")
        if c in predictions
    ]
    for _, run in predictions.groupby(identity_columns):
        validate_prediction_identity(run, expected)
candidate_path = smoke_root / "splits_ablations" / "matched_candidates_seed44_v2.csv"
candidates = pd.read_csv(candidate_path)
assert "label" not in candidates and "SNOOPPI_final_label" not in candidates
assert candidates.candidate_status.eq("unknown_requires_review").all()
records.append(
    {"common_prediction_export_schema": True, "unknown_candidate_label_isolation": True}
)
(smoke_root / "validation.json").write_text(json.dumps(records, indent=2))
print("Synthetic smoke artifacts:", smoke_root, flush=True)
