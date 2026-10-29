# Status: BLOCKED — no final model results

Prepared 29 October 2026. Scope: stronger cold-both generalization with full-length
MMseqs2 clusters and matched frozen-encoder, RF and MLP comparisons.

## What actually ran

Inspected repository commit `0ab1234e95b0e32b200ddbaa8ec09486255734dd`, located
three labeled exports and all three caches in Drive, downloaded validation/test,
and ran a source-only audit. Sixteen synthetic/integration tests pass, including deliberate
reversed-pair leakage, homolog leakage, truncation collision, malformed input,
missing cluster mapping and deterministic splitting checks. The real MMseqs2 binary also passed a synthetic similarity fixture, and real PyTorch/cache decoding and all nine fitting paths passed synthetic lifecycle checks. No scientific-data clustering, training or final evaluation has run.

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
sensitivity 7.5, three CPU threads. Cluster full-length sequences before encoder
normalization/truncation. Merge groups that share either historical encoder input.
Assign those groups by fixed seed-44 hashing (80/10/10); exclude pairs crossing
partitions. This avoids forcing every interaction component into one partition,
but may substantially reduce retention, especially held-out pairs. Report the
component sizes, exclusions and realized class counts; require at least 50 examples
of each class per partition. Do not search seeds for desirable test balance.
Cluster-disjointness refers to detected MMseqs2 groups, not guaranteed absence of
all remote homology or shared short domains.

## Current execution blocker (updated 29 October 2026)

The earlier proxy diagnosis was incomplete: shell commands need explicit network
permission. With that enabled, GitHub and package downloads succeed. CPU PyTorch
2.15.0 and MMseqs2 release 18-8cc5c are now installed and exercised.

The authenticated large-file host `sdmntprwestus.oaiusercontent.com` receives an
explicit proxy 403 under the current allowlist. The owner has been asked to add
that host in the cloud environment settings. No alternate network route was used.
The 32 MiB convenience downloader cannot materialize the training CSV or caches;
streaming their authenticated file references can proceed once the host is allowed.

`train.csv` is 44,669,229 bytes; caches are 69,754,503 / 82,308,919 / 78,936,475 bytes.
The complete input cohort and actual embedding provenance remain unavailable.
No final research split or scientific model result exists and no job is running.

## Implemented since initial preparation

The matched runner now covers all three RF/MLP comparisons and all three ProtT5
fusion-block ablations. It locks the configuration, validates cache/split identities,
keeps development separate from final testing, verifies checkpoint reload, preserves
completed attempts, blocks refitting after final exposure, and declines unsupported
component-bootstrap uncertainty. Synthetic tests exercise these behaviors with
reduced fixture-only training budgets; their scores are not research results.

The clustering command now explicitly uses single-step clustering and a 1,000-hit
candidate cap with three threads. Protocol version 2 records this change before
any scientific-data execution. Complete source auditing and feasibility must run
before the real classifier batch.

## Limits and next action

Apply the single-host allowlist change, transfer the original train CSV and caches,
then run full preparation and audit cache provenance before fitting. The legacy
ProtT5 provenance gate may need external evidence; its dimensions alone do not
establish its encoder/checkpoint identity. Do not overwrite prior evidence or turn
unknown labels into negatives.

The standalone runner is not an installed autonomous ML controller. No protected
external evaluator is installed. Historical test examples have appeared in prior
experiments; the new split is a new exploratory benchmark, not an untouched
confirmatory dataset. No performance ranking is available.
