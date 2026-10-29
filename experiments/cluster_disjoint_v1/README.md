# Cluster-disjoint preparation

Status: preparation only; no completed real-data clustering or training results.

Run synthetic gates: `python experiments/cluster_disjoint_v1/test_pipeline.py -v`.
Run preparation: `python experiments/cluster_disjoint_v1/pipeline.py --source /path/to/splits --output work/cluster_disjoint_v1/attempt_001`. Requires pandas, MMseqs2, and train.csv/validation.csv/test.csv with partner_A_sequence, partner_B_sequence and binary label columns. Existing outputs cannot be overwritten. Never run with Python -O.

Selected design: full-length MMseqs2 connected clustering at 30% identity and 80% bidirectional coverage, sensitivity 7.5; merge normalized/truncated input collisions across clusters; label-blind seed-44 group assignment at 80/10/10; exclude crossing edges; require 50 examples per class per split. The heuristic clustering does not guarantee absence of remote homology. Realized pair fractions differ from group fractions. Conflicting labels, malformed sequences and colliding distinct input pairs are quarantined, never relabeled.

Only reusable code and synthetic tests are published. Private source data, source hashes and audit outcomes are excluded. Complete source access and dependencies are required before full execution. The downstream matched encoder/RF/MLP training batch and unattended controller are not implemented here. No background training jobs are running.
