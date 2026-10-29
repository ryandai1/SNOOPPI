# Execution status — 2026-10-29

## Completed

- Inspected repository main at `0ab1234e95b0e32b200ddbaa8ec09486255734dd`.
- Prepared this isolated audit → full-length FASTA → MMseqs2 → component-disjoint
  split workflow using existing repository helpers.
- Passed eight synthetic tests (Python unittest); no model training involved.

## Blocked / not run

| Stage | State | Missing requirement |
| --- | --- | --- |
| Real labeled-pair audit | Not run | Existing train/validation/test CSVs |
| Real MMseqs2 clustering | Not run | Input sequences and MMseqs2 binary |
| Real split feasibility / freeze | Not run | Audited data and actual cluster mapping |
| Broad-rule ProtT5 embeddings | Not run | Data, model access and suitable compute |
| MLP / RF / MLP ablation results | Not run | Frozen splits, compatible caches, training runtime; MLP ablation implementation still outstanding |
| Final tables and conclusions | Not produced | Matched real experiment outputs |

Workspace inspection found three reported CPUs, no `mmseqs`, no `nvidia-smi`, no
PyTorch, and no input CSVs or embedding caches. The environment has no configured
Drive connection, secrets, or outbound identities. Its network policy excludes
Hugging Face and Drive; a GitHub clone also failed because the configured proxy
could not be reached. GitHub connector access works, so code can be reviewed there.
No network restriction was bypassed. No data or encoder weights were downloaded.

Next prerequisite: make the existing split CSVs and cache files accessible to the
execution environment (mounted path or accessible download), and provide a
working MMseqs2 runtime. Keep the existing data revision/provenance. The cloud
environment's network configuration also needs repair before package/model
downloads; GPU inference should preferably reuse the existing Colab setup/caches.

This is a tested preparation deliverable, **not completed experimental results**.
No background experiment is running.
