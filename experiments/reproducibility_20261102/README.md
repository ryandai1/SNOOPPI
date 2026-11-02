# Three-seed reproducibility review — November 2, 2026

Start with **[01_results.ipynb](01_results.ipynb)**. It runs from the checked-in aggregate files and shows the measured comparisons, dependence sensitivity, and saved verification receipts. No Drive mount or private data is needed for this aggregate readout.

| File | Purpose | What a rerun does |
|---|---|---|
| `01_results.ipynb` | Portable results and limitations | Recomputes seed summaries from published metrics and displays saved receipts |
| `02_retrain.ipynb` | Optional real-data fitting | Does nothing by default; requires private inputs, verified frozen partitions, and an explicit switch |
| `records/cloud_execution_record.ipynb` | Original executed cloud ledger | References the original private cloud paths; not the public entry point |
| `records/cloud_analysis_record.ipynb` | Original executed independent verification and bootstrap analysis | Requires private predictions and mappings to rerun the raw checks |
| `results/` | Aggregate metrics, 36 MLP histories, provenance and verification receipts | Supports inspection without publishing row-level data |
| `records/` | Test logs and command receipts | Preserves successful checks and the initial collection failure |

## Findings

All 45 configurations completed: RF, full MLP and three feature-removal MLPs for each of ProtT5, ESM-2 and ESM-C, at seeds 44–46. Classifiers reuse frozen caches; encoders were not retrained.

| Full-feature model | Mean test macro-AP | Across-seed standard deviation |
|---|---:|---:|
| ProtT5 RF | 0.6189 | 0.0060 |
| ESM-C MLP | 0.5997 | 0.0018 |
| ProtT5 MLP | 0.5853 | 0.0047 |
| ESM-2 RF | 0.5806 | 0.0139 |
| ESM-C RF | 0.5748 | 0.0033 |
| ESM-2 MLP | 0.5586 | 0.0188 |

Macro-AP is the mean of positive and negative average precision. Both exploratory paired component-bootstrap ranges include zero: ProtT5 RF minus ProtT5 MLP, −0.0062 to 0.1248; ProtT5 RF minus ESM-C MLP, −0.0519 to 0.0486. These ranges do not establish superiority. They are 1,000-draw percentile ranges conditional on the existing test set and three fitted seeds, with no confirmatory or multiplicity-adjusted significance claim.

## Splits, mappings and disjointness

- All six original CSV/cache hashes match the historical sources.
- Reconstructed train/validation/test CSVs match the historical bytes exactly: **13,648 / 1,602 / 1,783 pairs**.
- The regenerated mapping and assignment files have different hashes. The historical private mapping was unavailable for membership comparison; a row-order-only explanation is not established.
- Fresh mapping coverage and cross-partition cluster, full-sequence and normalized/truncated model-input isolation checks pass for either partner role.
- Every source row is reconciled across retained pairs and exclusions (39,003 total). Retained pairs are 46.1% of cleaned data.
- The largest test interaction component contains 1,212 of 1,783 pairs (68.0%). Row-independent confidence intervals are inappropriate.

Raw source rows, split CSVs, membership mappings, embedding caches, exclusions and per-pair predictions stay in the private cloud records. **This PR publishes code and aggregate evidence only.** Hashes cannot restore missing raw data. MMseqs2's heuristic clustering does not establish absence of all remote homologs or shared domains. The historical protocol is documented in [cluster_disjoint/README.md](../cluster_disjoint/README.md).

## Validation and provenance

Independent cloud verification checked 80,235 test predictions, recomputed all 45 test metric sets, verified 36 validation-selected MLP epochs, checked frozen hashes and isolation, and reconciled source rows. `fresh_verification.json` records per-seed counts. The public results notebook displays that receipt; it does not claim to reverify private predictions.

97 tests passed (69 main + 4 current preparation + 16 earlier integration + 8 benchmark). Eight draft workflows completed on synthetic fixtures; both historical record notebooks replayed. Synthetic mutation and mining scores have no biological interpretation. The initial combined pytest command failed module collection; separate suites replaced it successfully. Both the failure and replacement logs remain available.

The cloud kernel stopped after seeds 44 and 45. Seed 45's completed artifacts passed the runner's `--resume` checks before seed 46 ran. Its outer command receipt measures recovery checks, not original training time. Per-fit timings remain in metrics.

Seed-44 scores are not numerically identical to the historical run. PyTorch stayed at 2.15.0+cpu; NumPy changed 2.3.5 → 2.5.4, pandas 2.2.3 → 3.1.0 and scikit-learn 1.8.0 → 1.9.1. The differences are recorded without attributing causality to a specific library. `requirements-frozen.txt` captures the new cloud environment; manifests preserve the source/configuration hashes.

## Run the notebooks

Use Python 3.12 and an isolated environment with pandas, NumPy, Matplotlib, nbformat, nbclient and ipykernel. Run from the repository root or this directory:

```bash
python -m jupyter nbconvert --execute --to notebook --inplace experiments/reproducibility_20261102/01_results.ipynb
```

For optional retraining, install the scientific dependencies in `drafting_code/requirements.txt`. Supply trusted private inputs named `train.csv`, `validation.csv`, `test.csv`, `prott5.pt`, `esm2.pt`, `esmc.pt`; provide the verified split directory with its manifest and mapping. Edit the clearly marked paths in `02_retrain.ipynb`, choose a new destination, and enable `RUN_TRAINING`. Do not overwrite the recorded experiments. Restoring raw artifacts is a separate requirement from opening these notebooks.

## Interpretation limits

This remains an exploratory benchmark built from a previously exposed historical test pool. Three training seeds do not provide independent test sets. The split excludes over half the cleaned pairs; feature-removal ablations also change parameter counts. Legacy ProtT5 cache metadata cannot establish its historical revision/preprocessing provenance. Independent confirmation, real unknown-pair evidence and mutation/interface annotations remain open. Full-page HTML preview was blocked by browser policy; exported figures were inspected directly.
