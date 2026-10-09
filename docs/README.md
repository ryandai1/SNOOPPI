# Documentation

[Repository overview](../README.md) · [简体中文](translations/README.zh-CN.md)

| Folder or file | Purpose |
| --- | --- |
| [`guides/`](guides/) | How to run draft experiments and review unknown-pair mining artifacts. |
| [`reference/`](reference/) | Reproducibility details and historical draft revision notes. |
| [`translations/`](translations/) | The Simplified Chinese version of the main README. |
| [`cleanup_manifest.json`](cleanup_manifest.json) | Original notebook hashes and code-cell mappings used by `scripts/validate_notebooks.py`. |

## Guides

- [Draft experiment guide](guides/draft-experiments.md): setup, run order, caches, strict protocols, and validation commands.
- [Unknown-pair mining guide](guides/unknown-pair-mining.md): scoring, candidate selection, artifacts, and interpretation.

## Reference

- [Reproducibility notes](reference/reproducibility.md).
- [Draft revision notes](reference/draft-revision-notes.md): the historical review and its validation record.

Documentation is grouped here; executable notebooks, helpers, scripts, and tests keep their established paths because notebook imports and validation tools depend on them.
