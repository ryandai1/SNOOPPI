# Executed checks — 29 October 2026

16 checks passed in Python 3.12 with CPU PyTorch 2.15.0 and MMseqs2
8cc5ce367b5638c4306c2d7cfc652dd099a4643f. Command:

```bash
python -m unittest discover -s experiments/cluster_disjoint_v1 -p 'test_*.py' -v
```

Nine preparation checks cover reversed pairs, conflict quarantine, truncation
collisions, invalid labels/sequences, deterministic partitions, missing cluster
membership and injected cross-split endpoint/homolog leakage.

Six training checks cover an end-to-end synthetic nine-model batch, validation-only
feature access during fitting, saved/reloaded predictions, symmetry of all fusion
variants, missing-cache rejection, artifact tampering, paired component intervals,
and declining inference when one component dominates. The lifecycle also verifies
that completed fits/inference are reused and test exposure blocks renewed fitting.
It uses two epochs and five trees solely to bound synthetic test cost; production
settings remain 50 epochs with early stopping and 400 trees.

One integration check executes the installed MMseqs2 binary on full-length synthetic
sequences; a one-residue variant clusters with its parent while an unrelated sequence
remains separate. No fake mapping replaces MMseqs2 in that check.

These checks establish software behavior on synthetic fixtures. They do not establish
real cache provenance, whole-source eligibility, scientific split feasibility or
model performance. The private training CSV and caches still require the pending
network-host allowlist update before scientific execution.
