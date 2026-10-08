# SNOOPPI draft experiments

These are source-only review drafts for frozen-embedding baselines, ablations,
and stricter PPI evaluation. The three historical notebooks in `../notebooks/`
remain unchanged. No new research result or model ranking is claimed.

## SNOOPPI citation

Dataset use should cite the published article:

> Vincoff S, Chatterjee P. **SNOOPPI: A Sequence-Normalized Database of On- and Off-Target Protein-Protein Interactions.** *Bioinformatics*. 2026; btag736. [https://doi.org/10.1093/bioinformatics/btag736](https://doi.org/10.1093/bioinformatics/btag736)

The complete BibTeX entry and links to the dataset, upstream repository, and
earlier OpenReview workshop version are in the [repository README](../README.md#citing-snooppi).

## What the earlier notebooks actually do

The available ProtT5 notebook constructs an approximately **80/10/10 stratified
pair-row split with seed 44** using two `train_test_split` calls. It has **no
MMseqs2 clustering step**. ESM-2 and ESM-C read the exported `train.csv`,
`validation.csv`, and `test.csv`. Cached embeddings do not establish that those
splits are protein- or cluster-disjoint. If a newer notebook or a separate
MMseqs2 export exists, reconcile that evidence before calling a split strict.

## Run order

1. Run **00** on the existing shared CSVs to audit duplicate/conflicting pairs,
   both historical normalization rules, protein overlap, preprocessing differences,
   optional reviewed cluster overlap, and length shortcuts. Unsafe shortcut fitting
   is skipped so audit tables can still be exported.
2. Use **02** if stricter partitions are needed. Review class counts, group sizes,
   realized row fractions, duplicate quarantine and cross-role exclusions. Enable
   `WRITE_SPLITS` only for feasible selected protocols. Original CSVs are not edited.
3. Set the **same `SNOOPPI_SPLIT_DIR` in every notebook**, rerun 00 on the chosen
   set, then run **01** (classical baselines) and **03** (operator/common pair-feature
   dimension ablations). Models stop on duplicate/conflicting or cross-split input
   pairs. Protein overlap is reported and precludes a cold-protein claim.
4. **05** provides optional pooled-cache Siamese/bilinear training with the common
   split/prediction contract. Cross-attention and LoRA remain separate scaffolds;
   pooled mean vectors cannot supply residue representations or encoder gradients.
5. **04** reads saved predictions from 01/03/05. It verifies the whole split-set
   fingerprint and every test pair/label before comparison. **06** requires a
   frozen scorer and independent mutation/interface annotations for test pairs.
6. **07** audits the exported SNOOPPI unknown CSV, scores valid rows with an
   existing frozen ProtT5 MLP, and optionally saves a negative-candidate review
   queue and verdict cache. No semi-supervised training is performed.

**Cross-species, temporal, mutation-family splitting, residue training, and LoRA
training are study-specific contracts/scaffolds, not completed experiments.**

## Shared setup in Colab or a local clone

Mount Google Drive first in Colab. Run from a repository clone (including the
helper modules) and set these variables **before the identical setup cell**:

```python
import os
# In Colab:
# from google.colab import drive
# drive.mount("/content/drive")
os.environ["SNOOPPI_REPO_ROOT"] = "/content/SNOOPPI"  # code clone
os.environ["SNOOPPI_ROOT"] = "/content/drive/MyDrive/SNOOPPI"  # data/artifacts
os.environ["SNOOPPI_SPLIT_DIR"] = "splits"  # relative to SNOOPPI_ROOT, or absolute
# After review/export, for example:
# os.environ["SNOOPPI_SPLIT_DIR"] = "splits_ablations/cold_cluster_seed44_v2"
```

Each notebook's setup now has an **EDIT YOUR GOOGLE DRIVE FILE PATHS HERE**
block. Fill in `SPLIT_CSV_FILES` with your exact cached training, validation and
test CSV locations. You can change filenames and use different folders; the
loader and split fingerprint use those exact files. Copy the same three values
into every notebook. `PLM_CACHE_FILES` lets you fill in the three separate `.pt`
embedding-cache locations too. The uploaded CSV schema contains sequences and
labels, so those files supply split data, not per-protein embedding vectors.
No downloading, embedding computation or split regeneration is needed to load
already cached files. Mount Drive yourself before running the setup.

Without per-file overrides, the selected split directory uses `train.csv`,
`validation.csv`, and `test.csv`. Every split must have both binary classes. Generated directories also include
`manifest.json`, recording file hashes, source hashes, definitions, fractions,
seed, class counts and mapping provenance. The loader verifies the hashes and
strict group isolation. Do not edit generated CSVs or mix partitions from runs.
Changing source dataset versions or re-splitting original test pairs creates a
new benchmark; old metrics cannot be used as matched comparisons.

Column aliases from the earlier notebooks are accepted. Labels are validated
**before integer casting**; fractional/out-of-range/missing values are rejected.
When `SNOOPPI_final_label` is present it must agree with `label` and contain only
positive/negative labels. Unknown rows cannot enter supervised splits. Negative
label provenance is retained in `label_evidence_source` (documented source label
or the binary split CSV), without inventing assay evidence.

## Existing caches for all three PLMs

The suite loads your existing caches; it never downloads encoders, recomputes
embeddings, overwrites caches or silently drops proteins with missing embeddings.
All three are selected by default in 01 and 03. A narrower comparison requires
an explicit `ENCODERS_TO_RUN` list. Paths and specifications are centralized in
`snooppi_utils.py`:

| Encoder | Cache under `embeddings/` | Protein dimension |
| --- | --- | --- |
| ProtT5 | `prot_t5_embedding_cache.pt` | 1024 |
| ESM-2 650M | `esm2_t33_650M_snooppi_mean_pool_embeddings.pt` | 1280 |
| ESM-C 600M | `esmc_600m_snooppi_mean_pool_embeddings.pt` | 1152 |

The direct legacy ProtT5 dictionary and metadata-wrapped ESM payloads match the
original exports. Checks cover dimensions, finite vectors, raw-sequence key
coverage, model/limit/pooling metadata and the ESM-C normalization/package
signature. Cache SHA-256 values and supplied metadata are recorded with runs.
Legacy ProtT5 has no stored revision/preprocessing signature, so those properties
remain externally unverified; a dimension check cannot establish cache provenance.
Only load trusted `.pt` files from your own experiments.

**Keep original raw sequence keys.** ProtT5 uppercases, substitutes U/Z/O/B with X
and truncates at 1024 characters; the ESM notebooks also remove spaces and map
all noncanonical characters to X. The audit reports when those actual inputs
differ. Multi-encoder comparisons stop on differences unless
`ALLOW_PREPROCESSING_DIFFERENCES=True` is explicitly chosen and recorded. Such a
run compares cached workflows; it does not isolate encoder identity under equal
preprocessing. Repartitioning can reuse frozen caches when coverage passes.

## Reviewed metadata and MMseqs2 mappings

Leakage checks hash model inputs after normalization/truncation. Metadata uses a
separate full-sequence identity to avoid confusing different proteins sharing a
prefix:

```python
from snooppi_utils import full_sequence_id, full_pair_id
# SHA-256 of uppercase raw sequence with literal spaces removed;
# no residue substitution and no truncation.
sequence_sha256 = full_sequence_id(raw_sequence)
```

`raw/mmseqs2_clusters.csv` requires one reviewed row per full sequence:

- `sequence_sha256`
- `cluster_id` (consistent representative/group identifier)
- `identity_scheme` = `full_uppercase_spaces_removed_sha256_v1`
- `evidence_source` (source export; retain MMseqs2 version/command, sequence
  identity/coverage parameters and dataset revision with that export)

Map raw MMseqs2 TSV representative/member IDs to their exact input sequences
before making this CSV. The code does not guess identifiers or clustering
thresholds. Missing, duplicate or ambiguous mappings fail. Reconcile your
previous clustering parameters; cluster-disjointness means disjoint groups in
that reviewed mapping, not proof of a universal homology threshold.

00 reads the default mapping if present and displays overlap. Set
`SNOOPPI_CLUSTER_FILE` to use another path. **When explicitly set, this is also a
strict model gate:** a missing mapping or cross-split cluster overlap stops
fitting. Leave it unset for a deliberately documented historical pair-row audit.

02 previews cold-target/cold-protein by default. Add `cold_cluster` or
`cold_family` to `PROTOCOLS_TO_PREVIEW` after mapping review. Cold-family uses
`raw/protein_metadata.csv` with the same hash/scheme/evidence columns and a
reviewed `family` value per protein. Cluster/family connected components link
both partner groups and both historical normalized input identities. Giant
components may make a split infeasible. Cold-target removes earlier rows where
held-out targets appear in **either partner role**, recording exclusions.
Group fractions are not guaranteed row fractions; inspect the displayed balance.

All copies of duplicate/colliding input pairs are quarantined before re-splitting,
not reduced to an arbitrary representative. Conflicting labels stop split design
for evidence review. Exported protocols retain identities, `original_split` and
label evidence, plus quarantine/exclusion tables. New destinations must be direct
children of `splits_ablations/`, and existing destinations cannot be overwritten.

## Unknown candidates and mutation inputs

For model-based mining of the full unknown export, use
[`07_mine_unknown_negatives.ipynb`](07_mine_unknown_negatives.ipynb) and the
[`mined/` artifact guide](../mined/README.md). Fill in `UNKNOWN_CSV`, the source
dataset revision, `CHECKPOINT_PATH`, and your existing ProtT5 cache paths. The
known compatible checkpoint is the original notebook **01** export, not draft
**03**, which does not save this MLP. If another notebook saved a different head,
this adapter rejects it rather than interpreting its weights as this predictor.
Both original export formats are supported: `best_predictor_checkpoint.pt`
(`model_state_dict` wrapper) and `best_predictor_weights.pt` (bare state dict).
The predictor loader uses the same trusted-file loading rule as the cache loader.
It rejects conflicting encoder, normalization, pair-feature-order or training
split metadata, including a nested `split_manifest` fingerprint. Scoring checks
finite features and logits before sigmoid. A regression compares loaded scores
and pair features with the **original notebook definitions** for both formats.

07 reports observed counts, sequence validity, full/input-prefix overlap with
each labeled partition, and optional raw-key embedding coverage. It rejects PTM
markup, whitespace, missing/all-X inputs and supplied non-protein molecule types;
reviewed parsing can be developed separately. It does not change the historical
ProtT5 normalization. Missing embeddings stop scoring without dropping rows;
append a separately saved compatible unknown-protein cache when needed.

Scoring and selection have separate switches. Scoring retains every valid row,
including overlap and duplicates, and writes the requested four columns plus a
hashed manifest. Selection requires `score < 0.1` by default and no identity
overlap for either partner with **any** labeled train/validation/test protein.
The exclusion considers full-sequence, actual ProtT5-input and ESM-input identity,
so long-sequence prefix collisions are also excluded. It does not prove homology
isolation. Candidates are deduplicated full pairs and ranked by score then
identity; `K=None` keeps all eligible pairs. Seed is recorded but no random
sampling is performed.
All scored copies of a full pair must agree, including copies above the cutoff;
inconsistent duplicates stop selection instead of favoring a low-scoring copy.

Review checkpoint/cache training provenance before enabling selection. Legacy
exports lack split/revision signatures; manual acknowledgement does not resolve
a recorded conflicting split fingerprint. Changed source CSVs, partitions,
caches, checkpoints or score files invalidate reuse. Outputs refuse overwrite.
The YAML config uses JSON syntax (valid YAML 1.2), and the verdict retains unknown
status and absence of negative evidence. Sigmoid scores are uncalibrated; no
per-pair confidence interval or exponential-tilting model is claimed. Unknown
candidates are never assigned binary labels or inserted into supervised splits.

The metadata-matched candidate workflow in **02** remains a separate design:

`raw/snooppi_unknown_candidates.csv` provides nonempty partner sequences;
optional source labels must be `unknown` or absent. Candidates match available
reviewed `species`, `localization`, `family`, and/or `expression_bin` strata from
**selected training positives only**, in either partner orientation. Matching
metadata requires the full-sequence hash/scheme and evidence source. Incomplete
metadata is excluded rather than matched as missing values. Labeled pairs from
all original source partitions are excluded, and candidate pairs are deduplicated.

The queue has **no binary label**. It records candidate provenance, training and
metadata hashes, `candidate_status=unknown_requires_review`, matching fields and
lack of negative evidence. It is written under `splits_ablations/` only when
`WRITE_NEGATIVE_CANDIDATES=True`. It is not a confirmed-negative dataset or a
specified positive-to-negative sampling ratio. Rerun candidate design against
the frozen selected split set before using its training strata in a study.

In 06, mutation rows require `pair_id`, `partner`, `position_1based`, `wild_type`,
`mutant`, `sequence_A`, `sequence_B`, `evidence_source`. Annotation `pair_id` uses
`full_pair_id(sequence_A, sequence_B)`; A/B orientation must match the supplied
sequence columns. Wild-type pairs must belong to the selected test split.
Positions are finite integers within the first 1024 residues; substitutions must
use one canonical residue and change the wild type. Interface rows require
`pair_id`, `partner`, `position_1based`, binary `is_interface`, `evidence_source`.
Connect the checkpoint-specific scorer and set `CHECKPOINT_PATH` before enabling
scoring. Review checkpoint training provenance and coordinate conventions too.

## Runs, writes and statistical limits

All write/training/scoring/LoRA switches default to false. Setup creates no output
directories. Use a unique `RUN_TAG`. New outputs refuse overwrite. 01/03/05 use
`QUICK_MODE=True` for development with seed 44 and at most 5000 training rows;
01/03 use the same stratified subset. Reporting uses full training and seeds
41–45. Quick-mode development results are explicitly flagged; 04 excludes them
unless `INCLUDE_DEVELOPMENT_RUNS=True` is deliberately set.

Predictions include full and normalized pair identities, original row order,
**test CSV hash and combined train/validation/test fingerprint**, encoder/model,
seed, quick-mode flag and validation-selected threshold. 04 rejects stale,
partial, duplicated, mislabeled or mixed-protocol predictions. Set `PREDICTION_FILES` in 04 to exact filenames for the selected protocol when
multiple runs share the output folder. Older prediction CSVs must be regenerated
or excluded from the chosen files. Copy exact displayed
`run_id` values to choose a reference; do not select one using test scores.

01/03/05 write JSON run manifests beside tables, preserving selected splits,
configuration, cache hashes/metadata/coverage and dependency versions. Model and
threshold selection use validation only. Gaussian random projection in 03 acts
on full **pair features**, equalizing the logistic head input width. Ordered
concatenation is a diagnostic with an explicit partner-swap score check.

The row/class-stratified bootstrap and row score-swap tests in 04, and the
residue-level Mann–Whitney test in 06, are exploratory if pairs/residues share
proteins or families. Strict partitions do not remove within-test dependence.
Use an appropriate reviewed group-resampling analysis for inferential claims.
Attention, mutation deltas and sigmoid scores are not binding-affinity evidence.

## Validation

```bash
python -m pip install -r drafting_code/requirements.txt
python scripts/validate_notebooks.py  # original code-cell preservation
python scripts/validate_drafts.py     # schema, syntax, identical setup, safe defaults
python -m pytest tests/test_draft_protocol.py tests/test_unknown_mining.py -q
python scripts/smoke_mining.py        # 07 only: synthetic frozen inference; NO training
# Optional separate check of 00–06 (includes synthetic baseline/head training):
python scripts/smoke_drafts.py        # synthetic end-to-end only; writes under work/
```

The smoke script creates synthetic split CSVs, authentic cache payload formats
and dimensions, reviewed fixture metadata, and a mock mutation scorer. It enables
writes/training **only in copied notebooks against its synthetic directory**,
executes all seven drafts, and saves executed notebooks/HTML/plots under `work/`.
It does not load Drive, download PLMs, test LoRA/residue training or establish
scientific performance. `smoke_mining.py` separately executes 07 with artificial
weights, vectors and pairs, without fitting any model or downloading any PLM.
Neither smoke test establishes actual unknown-pair counts, negative evidence or
performance. Review validation details in [REVISION_NOTES.md](REVISION_NOTES.md).
