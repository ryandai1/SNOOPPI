# Mining artifacts

[Repository overview](../README.md) · [Complete mining guide](../docs/guides/unknown-pair-mining.md)

This tracked folder is a scaffold, not a mined dataset. Generated artifacts are ignored by Git. The default output location is `SNOOPPI_ROOT/mined/`, which may be on Google Drive rather than inside this clone.

Use [draft notebook 07](../drafting_code/07_mine_unknown_negatives.ipynb) after auditing your labeled split CSVs. Read the [mining guide](../docs/guides/unknown-pair-mining.md) for setup, output filenames, provenance checks, and interpretation. Low model scores do not confirm negative interactions; unknown candidates never receive automatic supervised labels.
