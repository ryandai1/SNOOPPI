# Draft experiments

[Repository overview](../README.md) · [Complete experiment guide](../docs/guides/draft-experiments.md) · [Revision notes](../docs/reference/draft-revision-notes.md)

This folder contains the draft notebooks and their shared Python helpers. These are source-only review drafts; no new research result or model ranking is claimed.

| Notebook | Purpose |
| --- | --- |
| [00 — Data audit](00_data_audit_and_protocol.ipynb) | Inspect existing split CSVs, overlap, and preprocessing. |
| [01 — Classical baselines](01_classical_embedding_baselines.ipynb) | Compare classical classifiers using cached frozen embeddings. |
| [02 — Strict splits](02_strict_splits_and_negative_sampling.ipynb) | Preview stricter partitions and candidate sampling protocols. |
| [03 — ProtT5 ablations](03_prott5_ablation_matrix.ipynb) | Compare pair-feature operators and dimensions. |
| [04 — Calibration and statistics](04_calibration_and_statistics.ipynb) | Review saved predictions and matched comparisons. |
| [05 — Neural architecture drafts](05_neural_architecture_drafts.ipynb) | Explore pooled-cache neural heads and separate architecture scaffolds. |
| [06 — Interpretability](06_mutation_and_interface_interpretability.ipynb) | Review mutation and interface annotations with a frozen scorer. |
| [07 — Unknown-pair mining](07_mine_unknown_negatives.ipynb) | Score unknown pairs and prepare a candidate review queue. |

The filenames give notebook identifiers, not a top-to-bottom execution order. Follow the [guide's run order](../docs/guides/draft-experiments.md#run-order), setup, and validation instructions before executing. Action switches are disabled by default; unknown candidates are not confirmed negatives.

Shared helpers are `snooppi_utils.py`, `snooppi_protocol.py`, and `snooppi_mining.py`. Dependencies remain in [`requirements.txt`](requirements.txt). Keep this folder path when cloning: notebooks and validation tools use it to locate helpers and drafts.
