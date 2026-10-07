# SNOOPPI

Research notebooks for binary protein–protein interaction classification using frozen **ProtT5, ESM-2, and ESM-C** representations and a symmetric multilayer perceptron.

This repository documents the supplied experimental workflows. It does not introduce a new protein language model or establish a final model ranking. The cleanup preserves every original code cell and its order, adds numbered explanations, and removes saved outputs and transient notebook metadata. Debugged implementations and historical recovery cells remain intact.

## Notebook order

| Order | Notebook | Role |
| --- | --- | --- |
| 01 | [ProtT5 predictor](notebooks/01_prott5_predictor.ipynb) | Loads labeled SNOOPPI pairs, creates the shared partitions, embeds proteins, and trains/evaluates the classifier. |
| 02 | [ESM-2 benchmark](notebooks/02_esm2_benchmark.ipynb) | Loads the existing partition CSVs and evaluates the ESM-2 representation. |
| 03 | [ESM-C benchmark](notebooks/03_esmc_benchmark.ipynb) | Loads the same partition files, audits raw-sequence/pair overlap, and supports embedding and epoch-level recovery. |

Each notebook separates setup, data, representations, training, evaluation, and exports with numbered sections. Historical, optional, and recovery sections are labeled in place. The notebooks remain separate experiments; they have not been merged into one runtime.

## Running the experiments

1. Open a notebook in **Google Colab with a GPU**. Use a fresh runtime for each encoder and its original dependency setup cells. This ESM-2 notebook uses Hugging Face Transformers; ESM-C uses EvolutionaryScale's `esm` package with its own version constraints.
2. Mount Google Drive and verify that `/content/drive/MyDrive` is actually mounted before writing artifacts. The default experiment directory is `/content/drive/MyDrive/SNOOPPI`.
3. If the original partition CSVs already exist, preserve them and start with notebook 02 or 03. Otherwise use notebook 01 to create them. Recreating partitions from a newer unpinned dataset revision need not reproduce the original experiment.
4. For the documented single seeded ProtT5 run, execute sections **03–08**, skip the historical first training loop in **09**, then execute **10–12**. Section 03 requires the `datasets` package; install it with the original setup cell in section 01 if needed. Sections 01–02 and 13–17 contain exploration, diagnostics, archives, and recovery/export utilities; read their notes before running them.
5. In notebooks 02 and 03, follow the numbered core sections using the same three CSVs. Review existing cache/checkpoint paths before starting or resuming. The preserved cells can overwrite existing experiment artifacts.
6. Retain metrics, split hashes, resolved package/model/dataset versions, and configuration with each run. Compare results only after confirming identical evaluated pairs and metric definitions.

The shared files are `splits/train.csv`, `splits/validation.csv`, and `splits/test.csv`. Their required input columns are `Partner_A_sequence`, `Partner_B_sequence`, and a binary `label`; the benchmark loaders also support deriving labels from `SNOOPPI_final_label`. Inspect each loader's checks before using a new dataset. Existing numeric labels take precedence in the benchmark notebooks.

Data, model weights, embeddings, and experiment outputs are intentionally external to this repository. The original notebook outputs were cleared; this edition has **not been rerun** to regenerate results.

## Methods represented by the code

The supervised task uses SNOOPPI **positive and negative pair labels**. Unknown-label pairs are not added to the ProtT5 supervised dataframe and must not be treated as confirmed negatives. The ProtT5 main path reloads the source positive/negative splits; its earlier exploratory filtered export is not used for training.

ProtT5 creates a stratified random split of pair rows, approximately **80% training / 10% validation / 10% test**, with split seed **44**. ESM-2 and ESM-C reuse the exported files. This procedure does not establish protein-disjoint or sequence-cluster-disjoint evaluation. ESM-C reports raw overlap but does not repair it; a pair-overlap warning permits execution to continue.

| Property | ProtT5 | ESM-2 | ESM-C (default) |
| --- | --- | --- | --- |
| Encoder identifier | `Rostlab/prot_t5_xl_half_uniref50-enc` | `facebook/esm2_t33_650M_UR50D` | `esmc_600m` |
| Per-protein dimension | 1,024 | 1,280 | 1,152 |
| Symmetric pair dimension | 3,072 | 3,840 | 3,456 |
| Sequence limit | First 1,024 characters after normalization | First 1,024 residues after normalization | First 1,024 residues after normalization |
| Pooling | Mean, excluding padding and final EOS | Residue mean, excluding BOS/EOS/padding | Explicit residue-position mean, excluding BOS/EOS/padding and retaining `X` |

All encoders are frozen. For protein embeddings `a` and `b`, pair features concatenate **`a + b`, `abs(a - b)`, and `a * b`**. Swapping the partners leaves this representation unchanged.

The shared head design uses hidden widths **512 → 512 → 256**, ReLU activations, dropout **0.1**, and one output logit. Training uses batch size **1,024**, AdamW with learning rate and weight decay both **1e-4**, and weighted binary cross-entropy with `pos_weight = n_negative / n_positive` from the training split. The run limit is **50 epochs**, with early-stopping patience **10**.

Checkpoint selection maximizes **validation macro average precision**. The classification threshold maximizes **validation macro-F1** over 181 values from 0.05 to 0.95. Test labels are not used in those selection expressions.

## Reading the metrics

- **AUROC** measures ranking discrimination.
- **Positive AP** is average precision for label 1. Its prevalence baseline is the positive-class fraction.
- **Negative AP** applies the same calculation to `1 - label` and `1 - score`.
- **Macro-AP** is the arithmetic mean of positive and negative AP; it is not positive-only AP or trapezoidal area under a precision–recall curve.
- **Macro-F1** averages the two class F1 scores at the validation-selected threshold.

Report class counts and prevalence alongside these metrics. The notebooks' sigmoid scores have not been demonstrated to be calibrated interaction probabilities and are not binding-affinity estimates. No numerical leaderboard is included because this cleanup did not rerun the experiments or verify their input CSVs.

## Reproducibility and limitations

The notebook explanations and [reproducibility notes](docs/reproducibility.md) describe preserved behavior, including differences in normalization, checkpoint recovery, and historical cells. Matching hidden widths does not match total head parameter counts when input dimensions differ. These workflows therefore do not isolate encoder identity under fully identical preprocessing and model capacity.

The [cleanup manifest](docs/cleanup_manifest.json) records source hashes, ordered code-cell hashes, and original-to-cleaned cell mappings. Run the lightweight format/preservation check from the repository root:

```bash
python -m pip install nbformat
python scripts/validate_notebooks.py
```

This check validates notebook structure, unchanged ordered code sources, empty saved outputs, and Python syntax where applicable. It does not download models, mount Drive, train classifiers, audit data leakage, or validate scientific results. Full runtime validation requires the Colab setup, original split files, model access, and the run sequence above. Notebook rendering and generated figures have not been visually revalidated in this preparation environment.

## Upstream resources and attribution

- [SNOOPPI dataset](https://huggingface.co/datasets/ChatterjeeLab/SNOOPPI) and [upstream repository](https://github.com/sophievincoff/snooppi).
- [ProtT5 encoder](https://huggingface.co/Rostlab/prot_t5_xl_half_uniref50-enc).
- [ESM-2 model used here](https://huggingface.co/facebook/esm2_t33_650M_UR50D) and [upstream ESM research code](https://github.com/facebookresearch/esm).
- [ESM-C / EvolutionaryScale ESM](https://github.com/evolutionaryscale/esm).

Consult each upstream resource for its current citation and license/model terms, and record the exact revisions used in a paper. This repository adds no license grant for third-party data or weights. No project-wide code license has been selected for this private research workspace.
