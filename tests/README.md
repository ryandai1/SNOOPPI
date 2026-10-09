# Regression tests

[Repository overview](../README.md) · [Validation instructions](../docs/guides/draft-experiments.md#validation)

| Test file | Coverage |
| --- | --- |
| [`test_draft_protocol.py`](test_draft_protocol.py) | Draft split, label, cache, prediction, and protocol checks. |
| [`test_unknown_mining.py`](test_unknown_mining.py) | Frozen-predictor compatibility, unknown-pair scoring, and candidate isolation. |

From the repository root, with the draft dependencies installed:

```bash
python -m pytest tests/test_draft_protocol.py tests/test_unknown_mining.py -q
```

Tests use fixtures; they do not validate your actual Drive data or establish scientific results.
