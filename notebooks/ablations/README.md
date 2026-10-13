# Normalization ablations

The three main notebooks now use **uppercase → remove literal spaces → replace all characters outside `ACDEFGHIKLMNPQRSTVWY` with `X` → truncate to 1,024 residues**. To match the existing ESM implementation exactly, “spaces” means ASCII spaces; other whitespace is mapped to `X`.

The historical policy is **uppercase → replace only `U/Z/O/B` with `X` → truncate to 1,024 characters**. It retains spaces, `J`, punctuation, and other symbols at the normalization stage. Subsequent tokenization remains encoder-specific. This ablation compares the full historical preprocessing policy, including space handling.

| Notebook | Role |
| --- | --- |
| [01_prott5_predictor_uzob_original.ipynb](01_prott5_predictor_uzob_original.ipynb) | Byte-for-byte snapshot of the pre-change predictor, including original code, documentation, and output paths. |
| [02_esm2_uzob_ablation.ipynb](02_esm2_uzob_ablation.ipynb) | ESM-2 with original U/Z/O/B-only preprocessing; normalization-tagged cache metadata and isolated outputs. |
| [03_esmc_uzob_ablation.ipynb](03_esmc_uzob_ablation.ipynb) | ESM-C with original U/Z/O/B-only preprocessing; isolated embeddings, outputs, and training-resume checkpoints. |

## Which embeddings must be rebuilt?

All paths below are relative to `/content/drive/MyDrive/SNOOPPI/`.

| Experiment | Cache location | Action |
| --- | --- | --- |
| Historical ProtT5, U/Z/O/B only | `embeddings/prot_t5_embedding_cache.pt` | Preserve for the historical baseline. |
| Main ProtT5, broad rule | `runs/prott5_canonical_X_spaces_removed_prefix_v1/embeddings/prot_t5_embedding_cache.pt` | **Recompute**, then retrain the classifier and regenerate evaluation outputs. |
| Main ESM-2, broad rule | `embeddings/esm2_t33_650M_snooppi_mean_pool_embeddings.pt` | Existing cache can be reused with the same model/settings. |
| Main ESM-C, broad rule | `embeddings/esmc_600m_snooppi_mean_pool_embeddings.pt` | Existing cache can be reused with the same model/settings. |
| ESM-2, U/Z/O/B only | `ablations/uzob_only_prefix_v1/embeddings/esm2_t33_650M_snooppi_mean_pool_embeddings.pt` | **Recompute** with the new ablation notebook, then train/evaluate. |
| ESM-C, U/Z/O/B only | `ablations/uzob_only_prefix_v1/embeddings/esmc_600m_snooppi_mean_pool_embeddings.pt` | **Recompute** with the new ablation notebook, then train/evaluate. |

ESM-C 300M uses the corresponding `esmc_300m` filename. Both ESM ablations reject a cache whose normalization metadata is missing or differs from `uzob_only_prefix_v1`. Main ProtT5 builds a fresh cache and exports a metadata wrapper with the broad rule and its embedding dictionary. Its local pre-export cache also has a new name: `/content/prott5_canonical_X_spaces_removed_prefix_v1_embeddings.pt`.

Do not rename/copy an old cache into a new policy's directory. The caches use raw sequences as keys, so matching keys do not establish matching normalized inputs. Existing caches are not deleted or rewritten by this repository change.

## Run the comparisons

1. Preserve the original `splits/train.csv`, `splits/validation.csv`, and `splits/test.csv`. Main ProtT5 now loads those CSVs unchanged; a partial set is an error. It creates seeded partitions only when none exist. Both ESM ablations require all three existing files.
2. Rebuild the main broad-rule ProtT5 run using sections **03–08**, skip historical loop **09**, then run **10–12**. New artifacts go under `runs/prott5_canonical_X_spaces_removed_prefix_v1/`, including its normalization-tagged checkpoint. Install `datasets` from section 01 if creating splits from scratch.
3. Run each ESM ablation in its own fresh Colab GPU runtime, following the numbered core sections. The first embedding pass starts empty in its isolated directory; later runs can resume compatible caches. Classifier, seed, loss, features, model, truncation limit, and model-selection settings are copied from the corresponding main benchmark. ESM-C pools actual encoded tokens between BOS/EOS because retained spaces/symbols can change token count; canonical inputs retain the baseline mask. ESM-2 retains its original tokenizer and pooling logic.
4. Compare each encoder against its own broad-rule baseline, using identical split files and resolved model/package versions. The ESM-C comparison cell reads that encoder's broad-rule table and the new ablation table. Those table paths alone do not prove that split hashes and settings match.

The archived ProtT5 notebook intentionally retains historical paths and behavior. Use it as a preserved reference with the existing historical results. If rerunning it, use an isolated Drive copy of the project and the same original partitions; running the snapshot unchanged against your live project can overwrite historical results and regenerate splits. Its manual recovery cells also retain their original paths.

The existing draft/mining utilities target historical caches and the historical ProtT5 normalization. They have not been migrated to the new broad-rule ProtT5 checkpoint; use the matching original cache/checkpoint pair there. The main ESM-C notebook's historical cross-model comparison still reads legacy ProtT5 `tables/split_metrics.csv`; use the new ProtT5 run's table explicitly for a standardized comparison.

These notebook edits do not execute GPU inference or produce new scientific results. Embeddings and evaluation must be generated in your Colab/Drive environment. If both policies produce identical tokenized inputs on your data, a null preprocessing effect is possible.
