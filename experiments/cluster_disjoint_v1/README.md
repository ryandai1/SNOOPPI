# Cluster-disjoint experiment — BLOCKED at partial source audit

This isolated folder contains preparation code and actual partial-audit evidence,
not completed MMseqs2 or model results. Original notebooks and inputs are unchanged.
Read `REPORT.md` before resuming. Detailed protocol and run history remain in the cloud workspace.

Requirements for preparation: Python 3.12, pandas and MMseqs2. The tested environment
has pandas; MMseqs2 is missing. PyTorch and the three real caches are additionally
required for future model training. Do not run Python with optimization (`-O`).

```bash
python experiments/cluster_disjoint_v1/test_pipeline.py -v
python experiments/cluster_disjoint_v1/pipeline.py \
  --source /path/to/SNOOPPI/splits \
  --output work/cluster_disjoint_v1/full_attempt_001
```

The source directory must contain all three original labeled CSVs. Each output
folder must be new. The command audits sources, quarantines conflicts/invalid
sequences/input collisions, writes full-length FASTA, calls MMseqs2, checks mapping
coverage, assigns merged input/sequence clusters, audits both endpoints, and writes
hashed splits only if class-count gates pass. Failures preserve intermediate files;
inspect them before another attempt. A complete source manifest is mandatory.

`--partial-audit` exists solely for diagnosis of inaccessible sources. It never
clusters, splits or trains. `source_manifest.json` and `audit.json` here are from
that partial mode; raw sequences and row-level quarantine tables stay in ignored
`work/cluster_disjoint_v1/partial_audit_r002/` and are not published to GitHub.

The planned downstream training is **not implemented** in this folder. Existing
notebook 03 is a logistic feature probe, not the requested MLP ablation. Before
training, implement/freeze the original MLP plus three fusion-block removals and
all three encoder/RF/MLP comparisons under one split fingerprint. Verify legacy
cache provenance, actual preprocessing and coverage. Do not consume test metrics
until all comparison procedures are fixed. No model jobs are running unattended.

Aggregate audit outcomes and source-file hashes are published with explicit owner authorization. Raw data, protein sequences, embeddings and row-level quarantine tables remain private.
