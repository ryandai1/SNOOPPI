# Cluster-disjoint benchmark

Status: implemented and synthetically tested; the complete real-data run remains
blocked by the large-file download host's network allowlist. Read [REPORT.md](REPORT.md).
This is the maintained workflow on top of the preparation already merged in PR #8.
The older PR #7 proposes a different policy (retaining whole interaction components);
it is not needed to run this workflow.

Original notebooks, CSVs and embedding caches are never modified. All generated
row-level data and checkpoints belong under ignored `work/`, not public Git.
Aggregate audit summaries and source hashes may be published with owner authorization.

## Execution

Requirements: Python 3.12, numpy, pandas, scikit-learn, joblib, CPU PyTorch, MMseqs2.
The tested MMseqs2 version is `8cc5ce367b5638c4306c2d7cfc652dd099a4643f`
(release 18-8cc5c); PyTorch is 2.15.0+cpu. Use the repository's existing helper modules.

```bash
# Run from the repository root, with the chosen Python environment and mmseqs on PATH.
python experiments/cluster_disjoint_v1/test_pipeline.py -v
python experiments/cluster_disjoint_v1/test_training.py -v
python experiments/cluster_disjoint_v1/test_mmseqs.py -v

python experiments/cluster_disjoint_v1/pipeline.py \
  --source /path/to/SNOOPPI/splits \
  --output work/cluster_disjoint_v1/full_attempt_001

python experiments/cluster_disjoint_v1/train.py fit \
  --splits work/cluster_disjoint_v1/full_attempt_001 \
  --caches /path/to/SNOOPPI/embeddings \
  --output work/cluster_disjoint_v1/comparison_001

# Only after every fit and checkpoint readback has passed:
python experiments/cluster_disjoint_v1/train.py evaluate \
  --splits work/cluster_disjoint_v1/full_attempt_001 \
  --caches /path/to/SNOOPPI/embeddings \
  --output work/cluster_disjoint_v1/comparison_001
```

Legacy ProtT5 caches lack preprocessing/revision metadata. By default they block
fitting. If external source evidence supports using that cache, explicitly record
that decision and add `--allow-legacy-provenance` to both training commands. This
records the unverified provenance in the frozen configuration; it does not certify
it. Cache inputs must agree across the three encoders, cover every selected raw
sequence key, have the expected dimensions and contain finite vectors.

## Fixed preparation policy

Full-length MMseqs2 connected clustering at 30% identity, 80% bidirectional coverage,
sensitivity 7.5, single-step clustering, maximum 1,000 candidate hits per query,
three threads. Version 2 makes the search cap and single-step policy explicit;
no real clustering result was used to choose them. The heuristic does not guarantee
absence of all homology, shared short domains or hits beyond the cap.

Merge MMseqs2 clusters whose normalized/truncated model inputs coincide. Assign
those groups using label-blind seed-44 hashes (80/10/10), retaining only pairs whose
endpoints share a partition. Report interaction components and excluded pairs;
require 50 examples per class per partition. Never search for a favorable test seed.
Realized pair fractions differ from group fractions. Conflicts and malformed
sequences are quarantined without relabeling/sanitization; same-label duplicates
are deduplicated deterministically, and distinct colliding input pairs are quarantined.

Each preparation attempt needs a new output directory. `--partial-audit` is solely
for source diagnosis and never creates splits. Full output includes raw-source hashes,
quarantine/duplicate tables, clustering command/version, mapping, all assignments,
feasibility diagnostics, split hashes and provenance. Failure preserves evidence.
Never run Python with `-O` because preparation uses assertions as validation gates.

## Fixed comparison policy

Nine fits: RF and MLP on each of ProtT5, ESM-2 650M and ESM-C 600M, plus three
ProtT5 MLP leave-one-fusion-block-out variants. Pair features are sum, absolute
difference and product. Every model uses the same partitions and seed 44; there is
no hyperparameter search. A training-prevalence baseline precedes expensive fitting.

RF: 400 bootstrap trees, Gini, sqrt feature sampling, minimum leaf size 2, unrestricted
depth, fixed train-derived class weights. MLP: 512/512/256 hidden units, ReLU,
dropout .1, AdamW learning rate/weight decay 1e-4, weighted BCE, batch 1024, at most
50 epochs, patience 10, validation macro-AP checkpoint selection. This runs the
original notebook's intended training schedule once; it does not reproduce its
duplicated training-loop cells. No feature scaler is introduced.

Validation macro-F1 chooses the first maximizing threshold on the 181-point grid
from .05 through .95. All validation predictions and selected checkpoints are saved,
and checkpoint reload must reproduce predictions. A one-hour fitting budget and
three CPU threads bound classifier work; MMseqs2 has a separate one-hour timeout.
No encoder is downloaded or retrained. One seed minimizes cost and does not estimate
training-seed uncertainty.

`fit` can skip completed runs only after checking artifact identities. An interrupted
model attempt is retained and requires review before another attempt; this is not
an exact optimizer-state resume. A lock prevents concurrent runs in one output
folder. The final batch freezes all receipts/configuration before test scoring.
After test scoring starts, further fitting in that run is prohibited. An interrupted
final batch may reuse verified predictions. Configuration, cache, source, mapping,
split or code changes require a new versioned experiment, not silent reuse.

`final_metrics.csv` includes ten rows (nine models plus prevalence control).
`uncertainty.json` contains paired macro-AP comparisons to ProtT5 MLP using interaction
components of MMseqs2 clusters. Fewer than ten components, a component holding over
half the test rows, or too many single-class bootstrap samples yields INCONCLUSIVE
uncertainty rather than a row-bootstrap claim. These are exploratory intervals:
within-test dependency and historical test exposure remain limitations.

This command runner is not an installed unattended Codex controller or a protected
external evaluator. Logs, artifacts and code remain in the same writable workspace.
No background jobs are left running after this session.
