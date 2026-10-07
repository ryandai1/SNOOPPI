# Reproducibility notes

These notes describe the supplied code; no listed limitation was silently fixed during cleanup.

## Preserved implementation details

| Area | Observed behavior and consequence |
| --- | --- |
| Data selection | ProtT5 builds the supervised dataframe from the source positive/negative splits, bypassing its earlier exploratory filtered dataset. Do not describe those exploratory filters as training preprocessing. |
| Split isolation | Pair-row stratification does not guarantee unique unordered pairs or disjoint proteins across partitions. ESM-C audits raw overlaps but only warns about pair overlap. Normalized/truncated collisions and sequence clusters require separate checks. |
| Label provenance | Benchmark notebooks prefer an existing numeric `label`. Its agreement with the original SNOOPPI label is not established by that preference. Keep original label provenance. |
| Preprocessing | ProtT5 uppercases, replaces `U/Z/O/B` with `X`, and truncates. ESM-2 and ESM-C additionally remove literal spaces and map every noncanonical character to `X`. Text normalization is not a biological PTM parser. |
| Shared frozen embeddings | Caching embeddings for all partitions does not train the frozen encoder on their interaction labels. The split itself and downstream model-selection protocol still need validation. |
| Model capacity | Encoder dimensions differ; the first classifier layer therefore has different parameter counts despite matching hidden widths. |
| ProtT5 training | Two loops remain. The later seeded reset creates a new classifier and optimizer and records the history used by export. Skip the earlier loop for the documented single-run path. |
| ProtT5 recovery | Embeddings are saved after full computation, with no incremental recovery. A later runtime diagnostic shadows the imported `Dataset` class; do not blindly rerun earlier definitions afterward. |
| ProtT5 Drive export | A directory can exist without an active Drive mount. Historical manual recovery remains optional. Figure-writing failures can interrupt the large export cell before checkpoint saving. |
| ProtT5 unknown archive | An optional archive cell depends on `split_data` left by an earlier iteration. Verify which split it contains before executing; it is separate from supervised training. |
| ESM-2 recovery | Embedding progress saves every 1,000 new sequences. The best classifier checkpoint includes optimizer state, but the notebook does not implement automatic classifier-training resume or last-epoch/RNG restoration. |
| ESM-C cache recovery | The supplied setting is `CACHE_SAVE_EVERY = 1_000` new embeddings, plus completion/finally saves. A hard runtime loss can discard unsaved progress. This cleanup does not change the interval to 100. |
| ESM-C cache signature | Checks include model tag, normalization, package version, length limit, and dimension. Pooling text is stored but not checked; inference precision and an exact weight revision are not part of the checked signature. |
| ESM-C training recovery | Last-epoch checkpoints restore compatible training state; interrupted partial epochs repeat. Cross-runtime bitwise reproducibility is not established. |
| Dependencies | ESM-C pins `esm==3.2.1` and `transformers==4.48.1`, with constraints based on its runtime's installed scientific packages. The legacy notebooks do not provide complete lockfiles. Use separate environments and save resolved versions. |
| Results comparison | Loading an existing metrics CSV does not prove that its split files match another run. Verify input hashes and metric definitions before comparison. |

## Record with each experiment

Keep the dataset revision, split CSV SHA-256 hashes, class counts, overlap audit, model identifier/revision, normalization and pooling settings, embedding precision, seed, package versions, runtime/GPU, configuration, selected epoch, threshold, and evaluation tables together. Use a new output suffix when changing the experiment rather than overwriting its provenance.

The checked-in notebooks are source-only copies. Original execution logs and plots remain in the supplied notebooks, outside this cleaned repository. Their absence here is not evidence of successful re-execution. Runtime reproduction must be performed using the inputs and section order documented in the README.
