# Main experiment notebooks

[Repository overview](../README.md) · [Running instructions](../README.md#running-the-experiments) · [Reproducibility notes](../docs/reference/reproducibility.md)

| Notebook | Role |
| --- | --- |
| [01 — ProtT5 predictor](01_prott5_predictor.ipynb) | Create shared partitions when needed, compute embeddings, and train/evaluate the classifier. |
| [02 — ESM-2 benchmark](02_esm2_benchmark.ipynb) | Evaluate ESM-2 using the existing shared partition CSVs. |
| [03 — ESM-C benchmark](03_esmc_benchmark.ipynb) | Evaluate ESM-C using the same CSVs, with overlap audits and recovery support. |

Run in separate Google Colab GPU runtimes with mounted Drive and each encoder's own dependency setup. If original split CSVs already exist, preserve them and start with 02 or 03. Follow the root README's specific section order for the single seeded ProtT5 run.

These notebooks preserve their original code cells. Saved outputs are cleared; this edition has not been rerun to regenerate results. For split audits, cached-embedding baselines, and additional workflows, see the [draft experiments](../drafting_code/README.md).
