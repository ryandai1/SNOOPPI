# Validation scripts

[Repository overview](../README.md) · [Full validation instructions](../docs/guides/draft-experiments.md#validation)

Run commands from the repository root with the documented dependencies installed.

| Script | Purpose |
| --- | --- |
| [`validate_notebooks.py`](validate_notebooks.py) | Validate all main/ablation notebooks and check historical preservation against the archived ProtT5 and main ESM code-cell hashes. |
| [`validate_drafts.py`](validate_drafts.py) | Check draft notebook structure, syntax, shared setup, and saved defaults. |
| [`smoke_mining.py`](smoke_mining.py) | Execute synthetic unknown-pair inference without training. |
| [`smoke_drafts.py`](smoke_drafts.py) | Execute the separate synthetic draft workflow, including baseline/head training. |

Smoke checks write under ignored `work/` and do not establish real-data scientific performance. Read the complete guide before running them.
