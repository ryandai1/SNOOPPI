# Draft code review and revisions

Reviewed 2026-10-08 against the three preserved PLM notebooks in this checkout.
This revision changes the seven draft notebooks and their shared helpers;
it preserves all original code cells in `notebooks/`.

## Finding about the earlier split

The available ProtT5 code uses two stratified `train_test_split` calls with
`random_state=44`, producing approximately 80/10/10 train/validation/test pair
rows. The available ESM notebooks load its exported CSVs. No MMseqs2 invocation,
cluster assignment, or cluster-disjoint split construction appears in these
notebooks. This describes the code in this checkout, **not a verification of the
actual CSVs in Google Drive**. A separate/newer MMseqs2 workflow may exist and
must be reconciled using its input sequences, mapping, parameters and exports.

## Changes by notebook

| File | Material revisions |
| --- | --- |
| All drafts | Identical code/data path setup; one `SNOOPPI_SPLIT_DIR`; no implicit output-directory writes; consistent hashes/identities and disabled action defaults. |
| 00 | Audits both ProtT5 and ESM input identities, optional reviewed cluster overlap and preprocessing differences; unsafe shortcut fitting is skipped while audit export stays possible; symmetric length features; overwrite protection; selected-protocol plot title. |
| 01 | All three existing caches selected by default; metadata/dimension/coverage checks; no cache-key normalization or missing-protein row drops; consistent stratified quick subset; predictions with complete split-set fingerprint, full/normalized pair IDs, threshold and quick flag; run manifest. |
| 02 | Reviewed cluster/family component splits link both partner groups and both input normalizations; cold-target exclusion checks both partner roles and both input rules; all duplicate/colliding pair copies quarantined; conflicting labels stop design; requested fractions honored and realized balance displayed; independent feasibility errors; provenance-rich non-overwriting exports under `splits_ablations/`. |
| 02 candidates | Training-positive strata only; orientation-invariant matching; exclude labeled pairs; reject confirmed/numeric candidate labels; exclude incomplete metadata; queue has no binary class and retains candidate/evidence provenance; review CSV/manifest stay under `splits_ablations/`. |
| 03 | Fixed `GaussianRandomProjection` import from `sklearn.random_projection`; consistent caches/subset/seed controls; no silent encoder skipping; threshold/identity/manifests in exports; ordered-concatenation swap diagnostic; clarify that projection equalizes pair-feature width. |
| 04 | Validate exact selected train/validation/test fingerprint and every test identity/label; reject partial/duplicate/mixed-protocol predictions; exclude quick results by default; selectable prediction files; exact paired alignment without inner-join intersections; handle a lone reference run; shorter legends and bounded calibration curves; state dependence assumptions. |
| 05 | Optional working pooled-cache Siamese/bilinear training using common CSV/cache/prediction contracts; configured gradient clipping; device/config/class/empty-loader checks; selected validation checkpoint and threshold export; padded-mask and symmetry checks. Residue/LoRA training remains a scaffold. |
| 06 | Require selected held-out wild-type pairs and independent annotation provenance; validate full pair IDs, integer coordinates, canonical changed residues and input-window limits; checkpoint fingerprint; partner-specific plots and interface legend; flag exploratory residue-test independence limits. |

Shared modules now validate labels before integer conversion and reject numeric
labels contradicting documented source labels. Full-sequence metadata identity
is separate from truncated model-input identity. Cache checks use the earlier
export formats and preserve their raw sequence keys. Legacy ProtT5 cache
revision/preprocessing provenance remains unverified because its dictionary does
not store those fields. Existing cached preprocessing differences are disclosed
and gated, not concealed by pretending the embeddings were recomputed.

New tests and validation scripts exercise labels, duplicates/conflicts,
normalization collisions, target role leakage, cluster grouping, giant components,
custom fractions, mapping coverage, cache compatibility, feature symmetry,
stale/partial prediction identity, different training with identical test rows,
and unknown-candidate isolation. `.gitignore` excludes external datasets,
caches and synthetic validation artifacts from ordinary publication.

## Validation performed

- All seven draft notebook schemas and all Python code cells passed syntax checks.
  The shared setup is identical and all action switches are disabled in saved drafts.
- All three preserved PLM notebooks passed the ordered original-code hash check.
- **18 regression tests passed.** Expected warnings expose legacy ProtT5 metadata
  limitations and protein overlap in a deliberately non-strict fixture.
- All seven drafts executed top-to-bottom on synthetic fixtures, with optional
  writes enabled only in copied notebooks. This exercised authentic cache
  dimensions/formats for all three PLMs, baseline/ablation prediction exports,
  four strict protocol exports, candidate review, paired statistics, pooled
  neural training and mutation scoring with a mock scorer.
- All four synthetic strict exports reloaded with manifest/group checks and
  complete coverage in each PLM cache. Prediction exports from 01/03/05 passed
  the same identity contract; candidate exports had no supervised labels.
- After tightening the ProtT5 cold-target guard, notebook 02's final preview and
  exported-protocol reloading passed again, and the corresponding regression
  test passed. Other computation was unchanged.
- Synthetic plot images were inspected; the cramped calibration legend and
  missing interface legend were fixed. Executed notebooks and HTML previews were
  saved under ignored `work/`. Whole HTML pages were not inspected in a browser.
- `git diff --check` passed.

Validation used Python 3.12 with NumPy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1,
Matplotlib 3.11.2, SciPy 1.18.1 and CPU PyTorch 2.14.1. The minimum dependency
file supports installation but is not a complete lockfile. Record resolved
versions with real Colab runs; cross-runtime bitwise equivalence is not claimed.

## What is still unverified

The initial review had no real split CSVs available. In the follow-up, the
uploaded validation/test CSVs were inspected (see below). The training upload
exceeded the 32 MiB transfer limit. Google Drive caches, a separate MMseqs2
output, assay metadata and historical results remain unavailable.
The synthetic checks validate behavior, **not the scientific credibility of
existing scores**. No PLM embedding regeneration or full real-data benchmark
was performed. Residue/LoRA training and a real frozen mutation scorer were
not executed. Cross-species, temporal and mutation-family splitting remain
explicit study-specific contracts.

Before reporting ablations/baselines, audit the actual existing CSVs in 00,
reconcile any newer MMseqs2 evidence, review/export a feasible protocol if needed,
set the same frozen split directory everywhere, and run the three caches through
the coverage/preprocessing checks. Use full reporting runs, unique run tags and
validation-only selection. Select prediction files for that exact protocol in 04.
Group dependence still needs an appropriate group-resampling analysis for
inferential claims; row/residue tests are exploratory.

## Publication handoff

The revised `drafting_code/` files, dependency instructions, regression tests,
validation/smoke scripts and `.gitignore` are saved in the existing checkout.
Review the working-tree changes and run the checks in the draft README before
publishing your branch. Original PLM notebooks and real artifacts are unchanged.
No GitHub push or publication was performed. A separate download bundle could
not be placed in the designated output directory because its parent filesystem
is read-only in this environment.


## Follow-up: user-editable Colab paths and uploaded split check

Each draft now contains the same commented `SPLIT_CSV_FILES` and
`PLM_CACHE_FILES` block. Fill in exact Drive paths yourself. The shared CSV
loader and fingerprint read those configured files directly, including custom
filenames/folders; embedding loaders also honor the explicit cache paths.
No original CSV, cache, model or split contents were modified. Publication
requires the user's approval; no GitHub write was performed.

The uploaded validation/test files have the expected columns:
`partner_A_sequence`, `partner_B_sequence`, `SNOOPPI_final_label`, `label`.
Their numeric and documented labels agree row-by-row. Their aggregate audit:

| Uploaded split | Rows | Positive | Negative | Duplicate normalized pair rows | Normalized pairs with conflicting labels |
| --- | --- | --- | --- | --- | --- |
| validation.csv | 3900 | 3362 | 538 | 59 | 8 |
| test.csv | 3901 | 3363 | 538 | 76 | 10 |

The full-sequence pair check found **zero within-split duplicate pairs,
zero within-split conflicting pairs, and zero validation/test pair overlap**.
After normalization and 1024-residue truncation, there are **32 unordered pair
identities shared between validation and test** and **1457 shared protein input
identities**. Both historical PLM input rules find the same pair overlap and
within-split duplicate counts. The conflicting labels in the table refer to
collapsed model inputs from distinct full sequences, not a demonstrated conflict
in the original full-sequence pair labels. Reusing the cached vectors preserves
these input collisions; inspect them in 00 before reporting ablations.
These observations do not validate the full three-way split: `train.csv` could
not be downloaded because its size exceeds the tool's 32 MiB transfer limit.
The files are read-only inputs to this review. Path edits preserve the data;
they do not resolve model-input collisions. Run 00 in Colab with all three
cached CSVs and review truncation/normalization collisions before new reported
model comparisons.

Two new regression tests exercise renamed CSVs with matching fingerprints,
explicit PLM cache paths, and accidental reuse of one CSV for two partitions.
The follow-up checks include all seven shared setup/schema/syntax validations
and the original notebook preservation check. The regression suite now has
**20 passing tests**. All file-writing/training switches remain disabled by
default in the saved notebooks.

## Unknown-pair mining addition (07)

Added `07_mine_unknown_negatives.ipynb`, `snooppi_mining.py`, and the tracked
`mined/README.md` scaffold. Its shared setup is identical to 00–06. Drive input,
checkpoint and optional extra-cache paths have editable comments. Generated
mining data/configs are excluded from Git.

The four separately enabled stages audit the source unknown CSV, score all valid
rows with a frozen existing ProtT5 MLP, select a candidate review queue, and cache
its verdict. No training is implemented. The audit reports observed counts,
invalid-input reasons, duplicate full pairs, and row overlap against each/all
labeled partitions under three identity rules. No fixed 835k count is assumed.

Checkpoint inspection corrected the reference to draft 03: it has no export of
the original ProtT5 MLP. The compatible existing export is from the historical
ProtT5 notebook 01; a different user's checkpoint must match the validated
architecture. Exact tensor keys/shapes, finite values and any recorded training
split fingerprint are checked. Legacy missing split/cache signatures are
reported and require external provenance review before selection.

Scoring uses inference batches and the historical FP32 symmetric pair features;
it retains valid overlapping and duplicate rows. Missing embedding keys abort
without accepted partial score output. Scores have exactly the requested four
columns, with full unordered pair IDs and a separate input/output manifest.
Candidate selection requires strict `score < threshold` and excludes either
protein in either role from **all** labeled partitions, including full-sequence
and normalized/truncated-input identity collisions. Homology isolation is not
claimed. `K=None` retains all eligible unique pairs; an optional cap keeps the
lowest scores with deterministic tie breaking. Seed is retained without implying
random sampling. Unknown status and absence of negative evidence are explicit;
no binary label is assigned and no source split/cache is modified.

The score manifest, candidate YAML configuration, and verdict record source,
checkpoint, cache and split hashes. Changed files invalidate reuse; outputs
refuse overwrite. The YAML uses JSON syntax, valid YAML 1.2. A single uncalibrated
checkpoint does not supply per-pair confidence intervals, and no exponential
tilting or semi-supervised training is claimed.

Validation for this addition after proofreading: **58 regression tests passed**, all eight draft
notebooks passed schema/syntax/shared-setup/default-switch checks, and original
notebook code-cell preservation passed. The new notebook executed end to end
with synthetic weights/vectors and rows: 11 source rows, 7 valid scored rows,
4 low-score rows excluded for labeled protein overlap, and 2 unique candidates.
The separate audit-only execution checks that default switches load no scorer
or cache and create no mining output directory. These are software checks, not
real unknown-PPI results. No Drive checkpoint, complete unknown CSV, or complete
unknown-protein embedding cache was available here; real scoring remains a Colab
run with the user's input paths.

The user approved committing and publishing this mining addition after the
proofreading review. Publication includes source notebooks/helpers, tests and
documentation; Drive data, checkpoints, cached embeddings and synthetic outputs
remain excluded. Earlier approved revisions were committed separately.

### Predictor-loading proofreading follow-up

The loader now uses the shared trusted PyTorch file loader and supports both
authentic original exports: the `model_state_dict` checkpoint wrapper and bare
weights. It rejects non-floating/nonfinite weights and conflicting encoder,
normalization, pair-feature-order or flat/nested split-fingerprint metadata.
Explicit conflicting cache normalization is rejected too. Pair features and
logits must be finite before sigmoid, which could otherwise conceal infinite
logits as finite scores. Duplicate consistency is checked across every scored
copy, including copies above the mining cutoff; input/score hashes are verified
again after selection finishes streaming.

The new independent comparison tests execute only the historical normalization,
dataset and predictor **definitions**, without running original setup, embedding,
training or export cells. They save random frozen reference weights in both
original formats, then verify exact CPU equality of normalized inputs, FP32 pair
features and sigmoid scores between the original definitions and the mining
adapter. Swapping partners preserves scores, and loaded parameters are frozen
with dropout disabled. These tests verify format/behavior compatibility; they do
not identify or validate the user's unavailable Drive checkpoint.

Notebook prose and README instructions now name both supported export filenames
and the validation rules. All action switches remain disabled. The user granted
publication approval after the review and its checks finished.
