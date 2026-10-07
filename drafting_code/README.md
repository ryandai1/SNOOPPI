# SNOOPPI drafting notebooks

These notebooks are review drafts for stronger PPI baselines and ablations. They
do not modify the three preserved notebooks in `../notebooks/`.

## Recommended order

1. `00_data_audit_and_protocol.ipynb` — audit duplicate pairs, protein leakage,
   normalization collisions, class balance, and metadata-only label shortcuts.
2. `01_classical_embedding_baselines.ipynb` — run Dummy, logistic-regression,
   random-forest, ExtraTrees, and histogram-gradient-boosting baselines on the
   existing frozen encoder caches.
3. `02_strict_splits_and_negative_sampling.ipynb` — draft cold-protein,
   cold-target, cold-family, cross-species, temporal, and mutation evaluations;
   build a review table for biologically matched negative candidates.
4. `03_prott5_ablation_matrix.ipynb` — test pair operators, common dimensions,
   repeated seeds, and fair ProtT5/ESM comparisons.
5. `04_calibration_and_statistics.ipynb` — aggregate saved predictions, plot
   calibration, bootstrap confidence intervals, and paired model differences.
6. `05_neural_architecture_drafts.ipynb` — reusable Siamese, bilinear, and
   residue-level cross-attention models, plus a ProtT5 LoRA configuration draft.
7. `06_mutation_and_interface_interpretability.ipynb` — in-silico mutation and
   interface-enrichment evaluation for a trained residue-level model.

## Usage

Run in Google Colab with Google Drive mounted. The notebooks default to
`/content/drive/MyDrive/SNOOPPI`, matching the existing encoder notebooks. To use
another location, set `SNOOPPI_ROOT` before importing `snooppi_utils`.

Open the notebooks from a clone of this repository. The setup cells look for the
repository at `/content/SNOOPPI` in Colab. If it lives elsewhere, set
`SNOOPPI_REPO_ROOT` to the clone root before running the first cell. Keep this
code location separate from `SNOOPPI_ROOT`, which points to data and artifacts.

Notebooks 01, 03, and 04 start in `QUICK_MODE=True` for a bounded smoke run.
Switch it off, verify the displayed configuration, choose a unique `RUN_TAG`, and
only then enable `WRITE_OUTPUTS` for a reporting run.

The required shared inputs are:

- `splits/train.csv`, `splits/validation.csv`, `splits/test.csv`
- `embeddings/prot_t5_embedding_cache.pt`
- optionally `embeddings/esm2_t33_650M_snooppi_mean_pool_embeddings.pt`
- optionally `embeddings/esmc_600m_snooppi_mean_pool_embeddings.pt`

Each notebook begins with a **When to run / Inputs / Outputs / Safety** block.
Expensive training and file-writing actions have explicit switches. Outputs go
under `tables/drafting_code`, `figures/drafting_code`, and
`checkpoints/drafting_code`; original splits and encoder artifacts are not
overwritten.

The strict-split notebook may use these optional reviewed files:

- `raw/protein_metadata.csv` with sequence or sequence hash plus family,
  species, localization, expression, and date fields where available.
- `raw/snooppi_unknown_candidates.csv` for candidate-negative review. Unknown
  pairs are never silently relabeled as confirmed negatives.
- mutation or interface annotations configured in notebooks 02 and 06.

## Validation status

The notebooks are generated with `nbformat`, checked for notebook-schema and
Python syntax, and smoke-tested through their dependency-light setup cells.
Full execution requires the external SNOOPPI split/cache files and, for neural
experiments, a GPU runtime.
