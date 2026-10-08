# Unknown-pair mining artifacts

Run [`07_mine_unknown_negatives.ipynb`](../drafting_code/07_mine_unknown_negatives.ipynb)
after auditing the existing labeled split CSVs. Fill in your Drive paths; the
default output location is `SNOOPPI_ROOT/mined/`, which may be on Drive rather
than inside this clone. This tracked folder is a scaffold, not a mined dataset.

Stages are independent and disabled by default:

1. Audit the exported unknown CSV; optionally save `mining_summary.csv` and
   `missing_embedding_keys.csv` when coverage checking is enabled.
2. Load an existing compatible frozen ProtT5 MLP and complete embedding caches;
   save `unlabeled_scores.csv` and `unlabeled_scores.manifest.json`.
3. Select unique pairs with `score < 0.1` and no protein overlap with any labeled
   partition; save `candidate_negatives.csv` and `mining_config.yaml`.
4. Cache `mining_verdict.json`, preserving unknown status and absence of
   confirmed negative evidence.

The compatible original predictor exports are `best_predictor_checkpoint.pt`
and `best_predictor_weights.pt` from the historical ProtT5 notebook 01. Draft 03
does not export this MLP. Fill in the actual checkpoint path; conflicting model,
preprocessing, feature order or split metadata fails validation.

The verdict is a review-queue status, never an experimental negative finding.
There is no training or automatic transfer into `splits/` or `splits_ablations/`.
Outputs refuse overwrite and are ignored by Git. Use a fresh subfolder of
`mined/` for another threshold/checkpoint/source run. Keep the score CSV and its
manifest together; modified inputs invalidate selection and verdict caching.

Scores are uncalibrated. A single checkpoint does not provide per-pair confidence
intervals. Identity filtering does not establish homology or MMseqs2 cluster
isolation. Low scores can reflect uncertainty on unseen proteins.

Source: [ChatterjeeLab/SNOOPPI](https://huggingface.co/datasets/ChatterjeeLab/SNOOPPI).
Cite [Vincoff and Chatterjee, *Bioinformatics* (2026)](https://doi.org/10.1093/bioinformatics/btag736).
