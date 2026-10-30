# Full-length MMseqs2 cluster-disjoint experiment

Independent workflow using the historical Drive split CSVs as one source pool.
This defines a **new benchmark**: historical test scores are not matched comparisons.
See `RESULTS.md` for the execution status and measured results.

## Jupyter notebooks

- [Executed experiment record](01_experiment_record.ipynb): saved audit outputs, all model comparisons, all 12 complete MLP epoch tables, and learning curves.
- [Training code and optional rerun](02_training_code_and_rerun.ipynb): full original training implementation, preparation/splitting source displays, and an explicit opt-in cell for a fresh fit.

Both notebooks execute top-to-bottom with their saved outputs. They were assembled after the command-line experiment: original histories are loaded from CSV, and training is disabled by default. Validation loss, minibatch logs, and optimizer checkpoints were not saved in the original run. No model was retrained to produce these notebooks. Open them from a repository clone so the adjacent source and aggregate files are available.

## Protocol

- Validate binary labels and agreement with source labels. Exclude invalid protein
  strings (modification markup, punctuation, unsupported whitespace). Keep original
  strings as cache keys; full-sequence identities uppercase and remove literal spaces.
- Canonicalize unordered pairs. Quarantine **all copies** of conflicting full or
  normalized/truncated pairs: no majority vote or invented biological label. Retain
  one copy of consistent full-pair duplicates; quarantine distinct full pairs that
  collide under either legacy ProtT5 or ESM preprocessing. Save row-level reasons.
- Cluster full sequences with MMseqs2 `easy-cluster --min-seq-id 0.3 -c 0.8
  --cov-mode 0 --cluster-mode 1 --single-step-clustering 1 --seq-id-mode 0
  -s 7.5 --max-seqs 1000 --threads 2`. Coverage applies to both partners in an
  alignment; identity uses alignment length. Default E-value is 0.001. Connected
  component clustering is transitive; the threshold does not describe every pair
  within a cluster. MMseqs2 is heuristic, so disjoint clusters do not prove absence
  of every possible homologous cross-split match. Version, binary hash, exact command,
  FASTA hash, mapping, and logs accompany the run. Single-step mode is explicit:
  this MMseqs2 build's cascaded linclust path does not support cluster mode 1.
- Link both interaction partners' clusters and identical model-input hashes under
  both normalization rules. Try intact interaction components, aiming for 80/10/10.
  Require both classes and fractions within five percentage points of the targets.
- If giant interaction components prevent this, merge clusters sharing model inputs,
  assign those groups to partitions, and exclude edges crossing partitions. Examine
  256 fixed seed-44 assignments with group probabilities proportional to
  `sqrt([0.8, 0.1, 0.1])`; retain candidates with >=50 examples of each class per
  partition and fractions within five percentage points. Select the minimum summed
  squared relative per-class allocation error plus `0.1 * excluded_fraction`.
  This uses labels only for split stratification, never model predictions. Archive
  assignments and every excluded pair. Class balance, retention, and selection bias
  remain limitations even after leakage checks pass.
- Freeze hashes before fitting. Verify cluster and input isolation for both partners,
  cache coverage, dimensions, metadata, and equivalent preprocessing. Legacy ProtT5
  provenance cannot be independently established by its metadata-free dictionary.

## Matched classifier comparison

Use every retained training pair, frozen embeddings, and seed 44 for all models.
RF: 400 bootstrap trees, square-root feature sampling, minimum leaf 2, no depth
limit, class weights `{negative: 1, positive: train_negatives/train_positives}`.
MLP: hidden widths 512/512/256, ReLU, dropout 0.1, weighted BCE, AdamW learning rate
and weight decay 1e-4, batch 1024, at most 50 epochs, patience 10, best validation
macro-AP checkpoint. Threshold maximizes validation macro-F1 on 0.05–0.95 in steps
of 0.005. Test metrics never select a split, epoch, threshold, or hyperparameter.

For each of ProtT5, ESM-2 650M, and ESM-C 600M, fit RF on full features and MLP on
full `[sum, absolute difference, product]` features plus three leave-one-block-out
variants. Removing a block also reduces input width/parameter count. This is one
matched seed, not an uncertainty estimate or a definitive encoder ranking. Frozen
encoders are never retrained or downloaded. Save histories, identified predictions,
metrics, provenance and completion hashes; omit large classifier checkpoints.

## Reproduce

From the repository root, install NumPy, pandas, scikit-learn and CPU PyTorch, and
make the recorded MMseqs2 binary available. Supply trusted original Drive artifacts
under `INPUT`: `train.csv`, `validation.csv`, `test.csv`, `prott5.pt`, `esm2.pt`,
`esmc.pt`. Source hashes, filenames and modification times are recorded with the experiment.

```bash
python experiments/cluster_disjoint/prepare.py --inputs INPUT --out RUN --mmseqs /path/to/mmseqs
python experiments/cluster_disjoint/train.py --inputs INPUT --splits RUN --out RESULTS
python -m unittest discover -s experiments/cluster_disjoint -p 'test_*.py' -v
```

Destinations must be new. `train.py --resume` verifies configuration and completed
artifact hashes and resumes unfinished fits. `finish.py RUN` can finish an existing
clustering with a complete `mmseqs2_clusters.csv` when no frozen manifest exists.
Large source data, caches, MMseqs2 scratch files, and local environments stay under
ignored `work/cluster_disjoint/`. No historical notebooks or helper modules are edited.

The cloud `work/cluster_disjoint/published_bundle/` bundle stores frozen split CSVs and cluster assignments
as gzip files. Exclusions are compact indexes into the original source CSVs; source
row numbers are zero-based. Full sequences for excluded pairs and scratch FASTAs
are not duplicated in Git. To reuse the exact published partitions without
reclustering:

```bash
python experiments/cluster_disjoint/unpack.py --bundle work/cluster_disjoint/published_bundle --out RESTORED_RUN
python experiments/cluster_disjoint/train.py --inputs INPUT --splits RESTORED_RUN --out NEW_RESULTS
```

Unpacking verifies every included file against the original frozen hashes, then
writes a compact manifest linked to the original manifest SHA. Its manifest hash
changes because scratch artifacts are omitted; the train/validation/test file
hashes remain exactly those used in the reported experiment.

Only code, aggregate statistics and provenance hashes are published to GitHub.
Row-level data, mappings and predictions stay under ignored `work/cluster_disjoint/`.
