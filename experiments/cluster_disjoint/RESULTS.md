# Completed real-data experiment — 30 October 2026

All **15 fresh classifier fits** completed using the same frozen partitions and
seed 44. No PLM was downloaded or recomputed. The classifier batch used two CPU
threads and approximately 13.1 minutes of summed fit/evaluation time.
MMseqs2 candidate search and alignment took approximately 7 minutes separately.
These are exploratory results on a newly constructed benchmark, not an untouched
confirmatory test set: historical test pairs were pooled with the other source CSVs.

## Audit and frozen partitions

| Stage | Pairs |
|---|---:|
| Original source pool | 39,003 |
| Invalid protein-string rows excluded | 588 |
| Conflicting normalized/truncated inputs quarantined | 1,086 |
| Additional distinct pairs colliding after preprocessing quarantined | 350 |
| Clean full-length pairs | 36,979 |
| Cross-partition interactions excluded | 19,946 |
| Final retained pairs | 17,033 |

No conflicting full-length pair identities or consistent full-pair duplicates were
found. The conflicting rows form 135 normalized/truncated pair groups; no labels
were decided by majority vote. Exclusion indexes preserve original file and row.

MMseqs2 clustered **22,490 full-length sequences into 9,680 clusters**, at 30%
identity and 80% bidirectional coverage. Maximum length was 34,350; 2,784 sequences
exceed 1,024 residues. Both legacy preprocessing rules give identical inputs on
retained sequences. Cluster, full-sequence, and both model-input overlap checks
are **zero across every partition pair**, considering either interaction partner.

The largest interaction component held 34,276 pairs (92.7% of cleaned data), so
intact components could not meet 80/10/10. The documented seed-44 stratified group
assignment selected candidate 123 of 256 before any classifier fit, then excluded
cross-partition edges. It retained **46.1% of cleaned pairs (43.7% of source rows)**.

| Partition | Total | Positive | Negative | Positive fraction |
|---|---:|---:|---:|---:|
| Train | 13,648 | 11,847 | 1,801 | 86.80% |
| Validation | 1,602 | 1,396 | 206 | 87.14% |
| Test | 1,783 | 1,546 | 237 | 86.71% |

## Held-out measurements

Full features are sum, absolute difference, and product. Each removal retrains the
MLP; all thresholds use validation macro-F1, and MLP checkpoints use validation
macro-AP. Full training/validation/test metrics are in `results/metrics.csv`.

| Encoder | Head | Features | AUROC | Macro-AP | Macro-F1 | MCC |
|---|---|---|---:|---:|---:|---:|
| prott5 | rf | full | 0.7279 | 0.6168 | 0.5931 | 0.1895 |
| prott5 | mlp | full | 0.6646 | 0.5817 | 0.5757 | 0.1560 |
| prott5 | mlp | no_sum | 0.5782 | 0.5484 | 0.5516 | 0.1036 |
| prott5 | mlp | no_absdiff | 0.6994 | 0.5972 | 0.5459 | 0.0976 |
| prott5 | mlp | no_product | 0.6674 | 0.5851 | 0.5711 | 0.1423 |
| esm2 | rf | full | 0.7024 | 0.5902 | 0.5554 | 0.1117 |
| esm2 | mlp | full | 0.6431 | 0.5810 | 0.5640 | 0.1293 |
| esm2 | mlp | no_sum | 0.6350 | 0.5671 | 0.5417 | 0.0834 |
| esm2 | mlp | no_absdiff | 0.6311 | 0.5669 | 0.5545 | 0.1541 |
| esm2 | mlp | no_product | 0.6164 | 0.5725 | 0.5585 | 0.1169 |
| esmc | rf | full | 0.6697 | 0.5706 | 0.5544 | 0.1184 |
| esmc | mlp | full | 0.6839 | 0.5996 | 0.5821 | 0.1645 |
| esmc | mlp | no_sum | 0.6116 | 0.5746 | 0.5511 | 0.1164 |
| esmc | mlp | no_absdiff | 0.6432 | 0.5733 | 0.5283 | 0.0658 |
| esmc | mlp | no_product | 0.6929 | 0.5992 | 0.5632 | 0.1316 |
| Prevalence control | Constant | Training prevalence | 0.5000 | 0.5000 | 0.4644 | 0.0000 |

The highest observed macro-AP is **ProtT5 + RF (0.6168)**. Among the full MLP
references, **ESM-C scores highest (0.5996)**, compared with ProtT5 (0.5817) and
ESM-2 (0.5810). The ranking therefore depends on the classifier; this run does
not establish a universally best encoder.

## Interpretation limits

This is one matched training seed; no significance or seed-uncertainty claim is
made. The test set has 214 interaction components, but the largest contains 1,212
pairs (68.0%), making independent-row bootstrap intervals inappropriate. Removal
of a fusion block changes both features and head parameter count. Retention and
class-aware assignment alter the evaluated population. MMseqs2's heuristic search
and transitive clusters do not prove the absence of all remote homology or shared
domains. The prefilter cap was 1,000 candidates and the median list reached that
cap; no exhaustive all-pairs homology audit is claimed. The legacy ProtT5 cache has no revision/preprocessing metadata; dimensions,
coverage and input equivalence pass, but its historical provenance remains unverified.

## Verification and artifacts

Four focused regression tests passed. The independent verifier reconciled all
39,003 source rows, verified frozen hashes and zero overlap, checked all 26,745 test
predictions against the same ordered pair/label identities, recomputed all 15 test
metric sets, and checked all 12 selected MLP epochs. `results/verification.json`
records the checks; histories, aggregate metrics, input/cache hashes and exact
MMseqs2 parameters are published. No raw sequences, pair-level labels, cluster
membership, caches or predictions are published to public Git.

Cloud artifacts remain under `/workspace/SNOOPPI/work/cluster_disjoint/`:
`run_v2/` contains the frozen splits, complete mapping and exclusions; `results/`
contains every test prediction and fit receipt; `published_bundle/` is a compact
verified copy for restoring splits with `unpack.py`. Original Drive files and
historical notebooks were not edited. No training job remains running after completion.

This completed run is isolated in `experiments/cluster_disjoint/`. Earlier
`cluster_disjoint_v1` and `mmseqs2_benchmark` preparation records are preserved.
Unlike their fixed-hash split proposal, this run explicitly stratifies cluster
assignments before fitting; this policy and the 15-model matrix were fixed before
any model result was inspected. The shared helper files are unchanged between
source commit `0ab1234` and publication base `6713dc2`.
