# Random-forest baseline and ablation review

This guide covers `drafting_code/01_classical_embedding_baselines.ipynb` and
`drafting_code/03_prott5_ablation_matrix.ipynb`. It defines procedures, not new
scientific results. Existing MLP notebooks and saved input files are unchanged.

## 01: the random-forest comparison

The main comparison is **ProtT5 + random forest versus ProtT5 + MLP**, using the
same cached embeddings, split CSVs, and pair features:

`[h_A + h_B; abs(h_A - h_B); h_A * h_B]`.

| Setting | Protocol |
| --- | --- |
| Primary encoder | Frozen ProtT5; 1,024 values per protein and 3,072 per pair |
| Optional encoders | Explicitly add ESM-2 650M and/or ESM-C 600M |
| Forest | 400 trees, Gini criterion, bootstrap enabled, unrestricted depth |
| Feature sampling | Square root of the input-feature count at each split |
| Minimum samples | 2 per leaf; 2 to split an internal node |
| Class weighting | Negative = 1; positive = training negatives / training positives |
| Resampling/scaling | No class resampling or feature scaling; tree bootstrapping remains enabled |
| Hyperparameter selection | Prespecified fixed configuration; no search or checkpoint selection |
| Threshold selection | Validation macro-F1 on 0.05–0.95 in increments of 0.005 |
| Reporting seeds | 44, 45, 46 on one fixed split |
| Metrics | AUROC, positive AP, negative AP, macro-AP, macro-F1, MCC, balanced accuracy, Brier score, unweighted log loss |

For the recorded training counts, the positive weight is 4,302 / 26,900 =
0.15992565. This matches the MLP's relative class weighting, but the fitting
objectives remain different. The previous forest used `balanced_subsample`, which
recomputed weights separately for each bootstrap sample. The revised weights
are fixed from the entire selected training set.

No tuning advantage is claimed: this is one fixed RF baseline. For reporting,
compare the seed-44 RF with the historical seed-44 MLP. If reporting RF mean ± SD,
also repeat the MLP on the same seeds before claiming a repeated-run comparison.
The standard deviation describes training randomness, not new splits or
interaction-level confidence. RF scores are not demonstrated to be calibrated.

### Run in Colab

1. Mount Google Drive and clone the repository into `/content/SNOOPPI` if needed.
2. Install `drafting_code/requirements.txt` in the runtime. No PLM download or GPU
   is required for the forest; PyTorch is used only to read existing `.pt` caches.
3. Set `SNOOPPI_ROOT=/content/drive/MyDrive/SNOOPPI` and
   `SNOOPPI_REPO_ROOT=/content/SNOOPPI` before running the setup cell. Leave
   `SNOOPPI_SPLIT_DIR=splits` for the historical pair split.
4. Check the three CSV paths and the actual ProtT5 `.pt` filename in section 1.
   The draft default is `embeddings/prot_t5_embedding_cache.pt`; some earlier
   exports used different names. Point to the real frozen cache without renaming
   a CSV or regenerating embeddings.
5. Run top-to-bottom with `QUICK_MODE=True`, `WRITE_OUTPUTS=False` for a development
   check. It uses at most 5,000 training rows, one seed, and validation evaluation;
   test files are audited but no test predictions or metrics are generated.
6. For paper results, use `QUICK_MODE=False`, `WRITE_OUTPUTS=True`, and a unique
   `RUN_TAG`, then rerun top-to-bottom. Leave `SAVE_MODELS=False` to save space;
   enable it if you need serialized forests for later inference. Output filenames
   separate development and reporting runs and refuse overwrites.

The reporting run writes train/validation/test metrics, a summary with sample
standard deviations, identified test predictions, test confusion counts, a PNG,
and a JSON manifest under `tables/drafting_code` and `figures/drafting_code`.
Optional models go under `checkpoints/drafting_code`. Existing split/cache files
are never rewritten. The manifest includes split/cache hashes, RF parameters,
weights, seeds, normalization information, and the actual selection procedure.

Normalized pair collisions stop the run. Do not bypass this check to obtain a
score. If you adopt repaired or cold-protein splits, run the MLP on those same
splits too. Protein overlap alone is allowed for the historical pair-split
comparison and prevents a cold-protein claim. Cross-encoder comparisons also
check normalization differences; ProtT5-only RF versus MLP reuses the same cache.

`log_loss` is unweighted and must not be compared directly to the original MLP's
weighted BCE. The shared metric helper now preserves exact probability rankings
(including 0/1 forest scores); only the log-loss implementation performs its
necessary numerical clipping.

## 03: what the existing notebook actually does

No code cells in 03 were changed by this review; its explanations were corrected.
It is a **logistic-regression feature diagnostic**, with two experiment groups.

1. Load and audit the selected split CSVs and frozen caches. Default selection
   includes all three encoders, even though the first group uses only ProtT5.
2. Construct eight ProtT5 pair representations: each of sum, absolute difference,
   and product alone; each two-block combination; all three; ordered concatenation.
3. For each representation, fit StandardScaler on training data, then a balanced
   logistic regression (`C=1`, default L-BFGS solver, maximum 3,000 iterations).
4. Select a threshold using validation macro-F1; evaluate test metrics. For ordered
   concatenation, swap test partners and measure the mean absolute score change.
5. Separately compress each encoder's complete symmetric pair features with a
   Gaussian random projection: 256 dimensions in quick mode, 256 and 512 in full
   mode. Fit/transform through training inputs and use the same projection for
   validation/test. Fit another scaler + logistic regression after projection.
6. Summarize metrics and optionally export predictions, tables, a plot, and a
   manifest. Quick mode uses seed 44 and at most 5,000 training rows; full mode
   uses seeds 41–45. Its current quick mode still evaluates the test split.

| Comparison | What it can support |
| --- | --- |
| Full features vs one block removed | Contribution of that block under a linear classifier; input dimension also changes |
| One feature block alone | Sufficiency of that block for the logistic probe |
| Ordered concatenation vs swapped inputs | Dependence on arbitrary partner order |
| Projected features at equal width | Performance with compressed inputs to a linear classifier |

Changing `random_state` does not make the L-BFGS operator probes independent
stochastic fits. Their across-seed SD is not a useful training-uncertainty measure.
In the projection group, seeds change the projection matrix. Gaussian projection
is not PCA; it acts on pair features, not each protein. There is no unprojected
ESM-2/ESMC logistic reference here, so the notebook alone cannot quantify their
performance loss due to projection. It also does not harmonize preprocessing.

### Alignment with the paper

Notebook 03 does **not** remove MLP layers, test pooling methods, or directly
ablate the final MLP. Label its results as feature diagnostics if retained.
For the planned architecture study, the next experiment should train the
original ProtT5 MLP reference plus three leave-one-fusion-block-out variants,
with identical splits, training settings, validation checkpoint selection, and
matched seeds. A shallower head can be a separate experiment. Those MLP ablations
are not implemented by this notebook and must not be reported as completed.

You can write data handling, frozen embeddings, the existing MLP, and the RF
protocol now. Add numerical RF/ablation results only after real-data execution.

## Validation of this revision

Run the focused checks from the repository root with:

```bash
python -m pytest tests/test_rf_baseline.py -q
```

Notebook format and Python syntax were checked. All four focused tests passed;
the synthetic reporting figure was visually inspected. The RF protocol is exercised
with synthetic data in `tests/test_rf_baseline.py`, including full-data and
validation-only branches, class weighting, output identity, overwrite protection,
and exact metric behavior. Its `.pt` decoder is replaced by a synthetic fixture
loader because the preparation environment has no PyTorch or Drive caches.
This does not validate real cache decoding, real performance, or runtime/memory
on the full dataset. Run the notebook in Colab with the actual CSVs and caches
using the steps above for that validation. Notebook 03's existing code was
inspected but not run on research data.
