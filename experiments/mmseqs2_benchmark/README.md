# Isolated MMseqs2 benchmark preparation

**Status: preparation code tested on synthetic fixtures; real-data execution is blocked.**
See [STATUS.md](STATUS.md) for exactly what was and was not run. No model scores,
cluster counts, or scientific conclusions are claimed. Existing notebooks and
artifacts are unchanged. This folder is an orchestration layer over the existing
`drafting_code/snooppi_protocol.py` and `snooppi_utils.py` helpers.

## Protocol chosen before looking at results

- Use the original three labeled CSVs as the source cohort. Preserve their hashes
  and original partition identities. Repartitioning creates a **new benchmark**;
  historical scores are not matched comparisons on it.
- Preserve raw sequence keys for embedding lookup. Validate numeric and original
  labels. Never assign unknowns a negative label or vote away label conflicts.
- Stop for contradictory labels. Quarantine **all copies** of duplicate or
  normalized-colliding unordered pairs, following the repository's conservative
  policy. Quarantine is not evidence that those pairs are biologically invalid.
- Cluster full-length sequences after uppercase, removal of ASCII spaces, and
  noncanonical-residue substitution with `X`. Model inputs use the same rule and
  truncate to 1,024 residues. Reject markup and non-space whitespace for review
  rather than interpreting them as protein annotations.
- One initial MMseqs2 `easy-cluster` run: `--min-seq-id 0.3 -c 0.8 --cov-mode 0
  --cluster-mode 1 -s 7.5`, two CPU threads. Coverage mode 0 requires coverage of
  both sequences; cluster mode 1 uses connected components. Record actual binary
  version, exact command, FASTA and membership hashes. No parameter sweep.
- These are **operational clustering criteria**, not a universal homology cutoff.
  MMseqs2's heuristic search and coverage definition do not prove absence of every
  related sequence across clusters, particularly shared domains or short fragments.
- Connect clusters appearing together in any interaction, and connect identical
  model inputs (including 1,024-residue prefix collisions). Both partner roles are
  checked. The existing helper additionally checks historical ProtT5 identities.
- Assign entire connected interaction components with split seed 44, nominal
  80/10/10 fractions. The existing GroupShuffleSplit allocates groups, not rows.
  Require both classes in each partition and realized row fractions within five
  percentage points of target. This is a feasibility guard, not a guarantee of
  adequate sample size or class balance: inspect exported class counts/prevalence.
- If components make the split infeasible, stop and report component sizes.
  Do not silently discard cross-partition edges, search many split seeds, or relax
  clustering. Any alternative is a separately documented benchmark design.

## Run when the real files are available

Run from a complete checkout of this repository, with Python, NumPy, pandas,
scikit-learn and MMseqs2 installed. The preparation tests need no torch or GPU.
Choose new artifact directories; existing run/split destinations are refused.
Store data and caches outside the Git checkout.

```bash
python experiments/mmseqs2_benchmark/pipeline.py prepare \
  --source /path/to/SNOOPPI/splits \
  --run /path/to/SNOOPPI/mmseqs2_run_001 \
  --dataset-revision 'record-exact-source-revision-or-historical-provenance'

python experiments/mmseqs2_benchmark/pipeline.py cluster \
  --run /path/to/SNOOPPI/mmseqs2_run_001 --threads 2

python experiments/mmseqs2_benchmark/pipeline.py freeze \
  --run /path/to/SNOOPPI/mmseqs2_run_001 \
  --destination /path/to/SNOOPPI/splits_ablations/mmseqs2_30id_80cov_seed44_v1
```

`prepare` saves source overlap/preprocessing reports, quarantined pair identities,
retained pairs, full-length FASTA, and source/artifact/code hashes. `cluster` saves
the log and versioned run record. `freeze` validates those hashes and membership,
exports component sizes, mapping and split preview, and only writes splits after
feasibility checks. A failed run remains inspectable; use a new run directory for
a fresh attempt. A synthetic mapping is used only in tests, never in production.

The final `manifest.json` is compatible with the existing split loader. CSVs retain
labels, sequence keys, original partition, cluster assignments and components.
Hash checks detect accidental changes; they are not an authenticity signature.

## Training remains a separate, blocked stage

After reviewing a feasible frozen split, all model runs must use its exact three
CSVs and split fingerprint. No training or PLM download is launched by this script.

1. Recompute broad-rule ProtT5 embeddings once; preserve historical caches. Reuse
   main ESM-2/ESM-C embeddings only after checking configuration and full coverage.
2. Run the three frozen encoders with the shared MLP. Use seeds 44–46, early
   stopping/checkpoint selection by validation macro-AP and validation-selected
   macro-F1 thresholds. Never choose a model or protocol using test scores.
3. Run the matched ProtT5 RF comparator on the **same** broad-rule cache and split.
   Existing draft utilities still describe historical ProtT5 preprocessing; adapt
   and validate that cache contract before reusing them for this standardized run.
4. Implement and run the full ProtT5 MLP plus leave-sum-out, leave-absolute-
   difference-out and leave-product-out variants, with the same seeds/settings.
   Notebook 03's logistic probes do not implement these MLP ablations.
5. Save per-pair predictions, manifests, metrics, configurations and seed summaries.
   Across-seed SD represents training randomness; independent-pair uncertainty is
   not established by a row bootstrap when proteins/components repeat in test.

For minimum compute, validate one training seed before repeating seeds; reuse all
compatible frozen embeddings; avoid optional encoder preprocessing ablations,
LoRA, cross-attention or broad hyperparameter searches in this core benchmark.
No GPU resources or paid jobs have been provisioned by this change.

## Validation

```bash
python -m unittest discover -s experiments/mmseqs2_benchmark -v
```

Tests exercise synthetic splitting and reload, partner symmetry, conflicts,
tampered artifacts, incomplete mappings, full-length FASTA, overwrite protection,
giant-component failure and selected command parameters. They do not invoke the
MMseqs2 binary or validate real data, caches, GPU training, or scientific results.
