# Status: BLOCKED — no final model results

Prepared 29 October 2026. Scope: stronger cold-both generalization with full-length
MMseqs2 clusters and matched frozen-encoder, RF and MLP comparisons.

## What actually ran

Inspected repository commit `0ab1234e95b0e32b200ddbaa8ec09486255734dd`, located
three labeled exports and all three caches in Drive, downloaded validation/test,
and ran a source-only audit. Nine synthetic tests pass, including deliberate
reversed-pair leakage, homolog leakage, truncation collision, malformed input,
missing cluster mapping and deterministic splitting checks. No real clustering,
training, cache decoding or final evaluation has run.

## Partial findings

These counts cover **only 7,801 validation/test rows**, not the full labeled dataset:

| Audit outcome | Count |
|---|---:|
| Invalid sequence rows quarantined | 120 |
| Remaining source rows | 7,681 |
| Conflicting full-length pairs among those rows | 0 |
| Redundant same-label full-length copies | 0 |
| Distinct pairs colliding after normalization/truncation | 213 |
| Eligible pairs after these filters | 7,468 |
| Eligible positive / negative pairs | 6,445 / 1,023 |

Invalid fields contain digits and object-like punctuation; they are not silently
rewritten into protein sequences. Quarantines preserve the source rows locally.
Conflicts may still emerge when the inaccessible training export is included.
Class counts are source eligibility diagnostics, not model results or a final split.

## Selected design

30% identity, 80% bidirectional coverage, MMseqs2 connected-component clustering,
sensitivity 7.5, four CPU threads. Cluster full-length sequences before encoder
normalization/truncation. Merge groups that share either historical encoder input.
Assign those groups by fixed seed-44 hashing (80/10/10); exclude pairs crossing
partitions. This avoids forcing every interaction component into one partition,
but may substantially reduce retention, especially held-out pairs. Report the
component sizes, exclusions and realized class counts; require at least 50 examples
of each class per partition. Do not search seeds for desirable test balance.
Cluster-disjointness refers to detected MMseqs2 groups, not guaranteed absence of
all remote homology or shared short domains.

## Why execution stopped

The managed cloud HTTP proxy refuses connections (`curl` and `git` exit with
connection errors). The workspace downloader accepts at most 32 MiB; `train.csv`
is 44,669,229 bytes and caches are 69,754,503 / 82,308,919 / 78,936,475 bytes.
MMseqs2 and PyTorch are absent. Connector discovery works, but it does not mount
Drive into this executor. No paid compute was started and no encoder was downloaded.

Restore the cloud proxy and a supported large-file transfer or Drive mount for
all three labeled CSVs and caches. Then run the complete audit and clustering,
review feasibility/cache provenance, implement the registered training batch,
and evaluate once. Do not treat the partial audit as a completed first stage.

## Limits

The standalone preparation runner is not an installed autonomous ML controller.
No protected external evaluator or training-resume mechanism has been validated.
The historical test has appeared in prior experiments; these new splits would
constitute a new benchmark, not a fresh untouched confirmatory dataset. Exact
cache provenance, including the legacy ProtT5 preprocessing/checkpoint metadata,
remains to be established. No performance ranking or uncertainty claim is available.
