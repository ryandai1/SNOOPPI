"""Synthetic checks for frozen inference, identity exclusion, and cache integrity."""

import json
import math
from pathlib import Path
import re
import sys

import nbformat
import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "drafting_code"))
from snooppi_mining import (
    ProtT5Predictor,
    audit_unknown,
    unknown_chunks,
    validity_reason,
    load_mining_caches,
    load_predictor,
    input_record,
    score_unknown,
    select_candidates,
    write_candidates,
    cache_verdict,
    SCORE_COLUMNS,
)
from snooppi_utils import (
    SEQUENCE_A as A,
    SEQUENCE_B as B,
    experiment_paths,
    load_splits,
    split_manifest,
    file_sha256,
    full_pair_id,
    build_pair_features,
    prott5_normalize_sequence,
)


def make_mining_fixture(root):
    """No training: deterministic artificial weights, vectors, and residue strings."""
    root = Path(root)
    (root / "splits").mkdir(parents=True)
    partitions = {
        "train": [("A" * 1024 + "C", "ACDE"), ("FGHI", "KLMN")],
        "validation": [("PQRS", "TVWY"), ("ACFG", "DEHI")],
        "test": [("KLPQ", "MNRST"), ("ACWY", "FGTV")],
    }
    for name, pairs in partitions.items():
        pd.DataFrame(
            {A: [a for a, _ in pairs], B: [b for _, b in pairs], "label": [0, 1]}
        ).to_csv(root / "splits" / f"{name}.csv", index=False)
    paths = experiment_paths(root)
    rows = [
        ("AAAAC", "CCCDA"),  # eligible
        ("CCCDA", "AAAAC"),  # reversed duplicate, scored then deduplicated
        ("VVVVC", "WWWDA"),  # eligible
        ("ACDE", "GGGGA"),  # overlap with a labeled training B
        ("GGGGC", "TVWY"),  # validation B in unknown B
        ("ACWY", "GGGGD"),  # test A in unknown A
        ("A" * 1024 + "D", "GGGGE"),  # full sequence differs; input prefix overlaps
        ("", "AAAAD"),
        ("XXXX", "AAAAD"),
        ("<M1+MI:0170>ACD", "AAAAD"),
        ("AC DE", "AAAAD"),
    ]
    unknown = pd.DataFrame(
        {
            "Partner_A_sequence": [a for a, _ in rows],
            "Partner_B_sequence": [b for _, b in rows],
            "SNOOPPI_final_label": "unknown",
        }
    )
    unknown_path = root / "unknown.csv"
    unknown.to_csv(unknown_path, index=False)
    rng = np.random.default_rng(44)
    sequences = {s for pair in rows[:7] for s in pair}
    cache_path = root / "cache.pt"
    torch.save(
        {
            s: torch.from_numpy(rng.normal(0, 0.01, 1024).astype(np.float32))
            for s in sequences
        },
        cache_path,
    )
    model = ProtT5Predictor()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        # Route an actual symmetric feature through the layers, with low logits.
        for layer in (
            model.network[0],
            model.network[3],
            model.network[6],
            model.network[9],
        ):
            layer.weight[0, 0] = 1.0
        model.network[9].bias.fill_(math.log(0.05 / 0.95))
    checkpoint_path = root / "checkpoint.pt"
    torch.save(
        dict(
            model_state_dict=model.state_dict(),
            model_class="PPIClassifier",
            split_seed=44,
        ),
        checkpoint_path,
    )
    splits = load_splits(paths)
    split_record = split_manifest(paths)
    cache, cache_records = load_mining_caches([cache_path])
    predictor, checkpoint_record = load_predictor(checkpoint_path, split_record)
    record = input_record(unknown_path, split_record, checkpoint_record, cache_records)
    return dict(
        root=root,
        paths=paths,
        unknown=unknown_path,
        splits=splits,
        split_record=split_record,
        cache=cache,
        cache_records=cache_records,
        model=predictor,
        checkpoint=checkpoint_path,
        record=record,
        scores=root / "mined" / "unlabeled_scores.csv",
    )


@pytest.fixture
def mining(tmp_path):
    return make_mining_fixture(tmp_path)


def scored(mining):
    return score_unknown(
        mining["unknown"],
        mining["model"],
        mining["cache"],
        mining["record"],
        mining["scores"],
        chunk_size=3,
        batch_size=2,
    )


def test_audit_reports_validity_and_overlap_in_both_roles(mining):
    summary, missing = audit_unknown(
        mining["unknown"], mining["splits"], mining["cache"], chunk_size=2
    )
    counts = summary.set_index("metric")["count"]
    assert (
        counts.total_rows == 11 and counts.valid_rows == 7 and counts.invalid_rows == 4
    )
    assert counts.overlap_any_labeled_full == 3
    assert counts.overlap_any_labeled_prott5_input == 4
    assert counts.overlap_train_any_identity == 2
    assert counts.overlap_validation_any_identity == 1
    assert counts.overlap_test_any_identity == 1
    assert counts.valid_rows_without_any_labeled_protein_overlap == 3
    assert counts.duplicate_valid_full_pair_rows == 1
    assert missing.empty


@pytest.mark.parametrize(
    "sequence,valid",
    [
        ("acduzob", True),
        ("XACDX", True),
        ("UZOBXXXX", False),
        ("AC D", False),
        ("<MI:0170>ACD", False),
        ("ACD*", False),
        (None, False),
        ("X" * 1024 + "ACD", False),
    ],
)
def test_checkpoint_normalization_validity(sequence, valid):
    assert (validity_reason(sequence) is None) == valid


@pytest.mark.parametrize(
    "column,value",
    [
        ("label", 0),
        ("label", 1),
        ("SNOOPPI_final_label", "positive"),
        ("source_label", "negative"),
    ],
)
def test_unknown_reader_rejects_supervised_labels(tmp_path, column, value):
    path = tmp_path / "unknown.csv"
    pd.DataFrame({A: ["ACD"], B: ["EFG"], column: [value]}).to_csv(path, index=False)
    with pytest.raises(ValueError):
        list(unknown_chunks(path))


@pytest.mark.parametrize("column", ["Partner_A_molecule_type", "partner_A_mol_type"])
def test_nonprotein_metadata_invalidates_row(tmp_path, column):
    path = tmp_path / "unknown.csv"
    pd.DataFrame({A: ["ACD"], B: ["EFG"], column: ["DNA"]}).to_csv(path, index=False)
    assert list(unknown_chunks(path))[0].validity_reason.tolist() == [
        "non_protein_molecule_type"
    ]


def test_frozen_batched_scores_match_direct_inference_and_keep_overlap(mining):
    before = {
        p: file_sha256(p)
        for p in [
            mining["unknown"],
            mining["checkpoint"],
            Path(mining["cache_records"][0]["path"]),
            *map(Path, mining["split_record"]["files"].values()),
        ]
    }
    record = scored(mining)
    scores = pd.read_csv(mining["scores"])
    valid = pd.concat(list(unknown_chunks(mining["unknown"])))
    valid = valid[valid.validity_reason.eq("valid")]
    with torch.inference_mode():
        expected = torch.sigmoid(
            mining["model"](
                torch.from_numpy(build_pair_features(valid, mining["cache"]))
            )
        ).numpy()
    np.testing.assert_allclose(scores.score, expected, atol=1e-7)
    assert list(scores.columns) == SCORE_COLUMNS
    assert record["valid_scored_rows"] == 7 and not record["selection_performed"]
    assert scores.pair_id.duplicated().sum() == 1
    assert all(not p.requires_grad for p in mining["model"].parameters())
    assert all(file_sha256(path) == digest for path, digest in before.items())
    assert not (mining["root"] / "mined/candidate_negatives.csv").exists()


def test_missing_embedding_aborts_without_partial_outputs(mining):
    del mining["cache"]["VVVVC"]
    with pytest.raises(KeyError, match="no rows were dropped"):
        scored(mining)
    assert not mining["scores"].exists()
    assert not mining["scores"].with_suffix(".manifest.json").exists()
    assert not list(mining["scores"].parent.glob(".mining-*"))


def test_selection_excludes_all_partitions_prefixes_and_deduplicates(mining):
    scored(mining)
    with pytest.raises(ValueError, match="provenance"):
        select_candidates(mining["scores"], mining["splits"], mining["split_record"])
    candidates, config = select_candidates(
        mining["scores"],
        mining["splits"],
        mining["split_record"],
        provenance_reviewed=True,
    )
    assert len(candidates) == 2
    assert (
        config["low_score_rows"] == 7 and config["excluded_low_score_overlap_rows"] == 4
    )
    assert set(candidates.pair_id) == {
        full_pair_id("AAAAC", "CCCDA"),
        full_pair_id("VVVVC", "WWWDA"),
    }
    assert "label" not in candidates
    assert candidates.negative_evidence.str.startswith("none;").all()
    capped, cap_config = select_candidates(
        mining["scores"],
        mining["splits"],
        mining["split_record"],
        K=1,
        seed=999,
        provenance_reviewed=True,
    )
    assert capped.pair_id.tolist() == candidates.head(1).pair_id.tolist()
    assert cap_config["K"] == 1 and cap_config["seed"] == 999


def test_threshold_is_strict_and_empty_verdict_is_wellformed(mining):
    scored(mining)
    scores = pd.read_csv(mining["scores"])
    # Rewrite synthetic scores AND synthetic manifest, simulating a separate scorer.
    scores["score"] = 0.1
    scores.to_csv(mining["scores"], index=False)
    manifest_path = mining["scores"].with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest["score_file_sha256"] = file_sha256(mining["scores"])
    manifest_path.write_text(json.dumps(manifest))
    candidates, config = select_candidates(
        mining["scores"],
        mining["splits"],
        mining["split_record"],
        provenance_reviewed=True,
    )
    assert candidates.empty and config["low_score_rows"] == 0
    output, yaml, verdict = [
        mining["root"] / "mined" / name
        for name in (
            "candidate_negatives.csv",
            "mining_config.yaml",
            "mining_verdict.json",
        )
    ]
    write_candidates(candidates, config, output, yaml)
    cached = cache_verdict(output, yaml, verdict, mining["split_record"])
    assert cached["status"] == "no_eligible_candidates"
    assert not cached["training_performed"] and not cached["binary_labels_assigned"]
    assert json.loads(yaml.read_text())["K"] is None


@pytest.mark.parametrize("which", ["unknown", "checkpoint", "scores", "split", "cache"])
def test_changed_input_or_scores_invalidate_selection(mining, which):
    scored(mining)
    path = {
        "split": Path(mining["split_record"]["files"]["test"]),
        "cache": Path(mining["cache_records"][0]["path"]),
    }.get(which, mining.get(which))
    with Path(path).open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="changed"):
        select_candidates(
            mining["scores"],
            mining["splits"],
            mining["split_record"],
            provenance_reviewed=True,
        )


def test_checkpoint_shape_and_split_mismatch_rejected(mining):
    payload = torch.load(mining["checkpoint"], weights_only=False)
    payload["model_state_dict"]["network.0.weight"] = torch.zeros(512, 3840)
    torch.save(payload, mining["checkpoint"])
    with pytest.raises(ValueError, match="incompatible"):
        load_predictor(mining["checkpoint"], mining["split_record"])
    payload["model_state_dict"] = mining["model"].state_dict()
    payload["split_set_sha256"] = "wrong"
    torch.save(payload, mining["checkpoint"])
    with pytest.raises(ValueError, match="fingerprint"):
        load_predictor(
            mining["checkpoint"], mining["split_record"], provenance_reviewed=True
        )


def test_inconsistent_embedding_caches_rejected(mining):
    other = mining["root"] / "other.pt"
    torch.save({"AAAAC": torch.zeros(1024)}, other)
    with pytest.raises(ValueError, match="Different embeddings"):
        load_mining_caches([Path(mining["cache_records"][0]["path"]), other])


def test_outputs_refuse_overwrite_and_tampered_candidate_verdict(mining):
    scored(mining)
    with pytest.raises(FileExistsError):
        scored(mining)
    candidates, config = select_candidates(
        mining["scores"],
        mining["splits"],
        mining["split_record"],
        provenance_reviewed=True,
    )
    output, yaml = (
        mining["root"] / "mined/candidate_negatives.csv",
        mining["root"] / "mined/mining_config.yaml",
    )
    write_candidates(candidates, config, output, yaml)
    with output.open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="changed"):
        cache_verdict(
            output,
            yaml,
            mining["root"] / "mined/mining_verdict.json",
            mining["split_record"],
        )


@pytest.mark.parametrize("wrapped", [False, True])
def test_original_notebook_exports_and_predictions_match_loader(mining, wrapped):
    """Independent oracle: execute only historical class/normalization definitions.

    This never executes original setup, embedding, fitting, or export cells.
    It compares random frozen weights exported in both authentic formats.
    """
    original = nbformat.read(
        Path(__file__).resolve().parents[1] / "notebooks/01_prott5_predictor.ipynb", 4
    )
    namespace = {
        "nn": torch.nn,
        "torch": torch,
        "re": re,
        "Dataset": torch.utils.data.Dataset,
    }
    definitions = [
        cell.source
        for cell in original.cells
        if cell.cell_type == "code"
        and (
            cell.source.startswith("class PPIClassifier")
            or cell.source.startswith("class PPIPairDataset")
            or cell.source.startswith("MAX_RESIDUES = 1024\n\ndef normalize_sequence")
        )
    ]
    assert len(definitions) == 3
    for definition in definitions:
        exec(compile(definition, "historical_prott5_reference", "exec"), namespace)
    for raw in ("acDuzob", "AC D U", "A" * 1024 + "UZ", None):
        assert prott5_normalize_sequence(raw) == namespace["normalize_sequence"](raw)

    reference = namespace["PPIClassifier"]().eval()
    output = mining["root"] / (
        "original_checkpoint.pt" if wrapped else "original_weights.pt"
    )
    payload = (
        dict(
            model_state_dict=reference.state_dict(),
            model_class="PPIClassifier",
            model_architecture=str(reference),
            selected_epoch=3,
            decision_threshold=0.55,
            loss_function="BCEWithLogitsLoss",
            split_seed=44,
            validation_selection_metric="macro_ap",
            split_metrics={},
        )
        if wrapped
        else reference.state_dict()
    )
    torch.save(payload, output)
    loaded, record = load_predictor(output, mining["split_record"])
    frame = pd.DataFrame(
        {
            A: ["AAAAC", "CCCDA", "VVVVC"],
            B: ["CCCDA", "AAAAC", "WWWDA"],
            "label": [0, 0, 1],
        }
    )
    historical_cache = {
        s: torch.from_numpy(v).half() for s, v in mining["cache"].items()
    }
    dataset = namespace["PPIPairDataset"](frame, historical_cache)
    historical_features = torch.stack([dataset[i][0] for i in range(len(dataset))])
    loaded_cache = {s: v.float().numpy() for s, v in historical_cache.items()}
    actual_features = torch.from_numpy(build_pair_features(frame, loaded_cache))
    torch.testing.assert_close(actual_features, historical_features, rtol=0, atol=0)
    with torch.inference_mode():
        expected = torch.sigmoid(reference(historical_features))
        actual = torch.sigmoid(loaded(actual_features))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual[0], actual[1], rtol=0, atol=0)
    assert not loaded.training and all(not p.requires_grad for p in loaded.parameters())
    assert record["checkpoint_format"] == (
        "model_state_dict_wrapper" if wrapped else "bare_state_dict"
    )
    assert record["feature_operations"] == ["sum", "absdiff", "product"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("encoder", "ESM-2 650M"),
        ("normalization", "spaces_removed"),
        ("feature_operations", ["product", "sum", "absdiff"]),
        ("split_manifest", {"split_set_sha256": "different"}),
    ],
)
def test_contradicting_checkpoint_metadata_rejected(mining, field, value):
    payload = torch.load(mining["checkpoint"], weights_only=False)
    payload[field] = value
    torch.save(payload, mining["checkpoint"])
    with pytest.raises(ValueError):
        load_predictor(
            mining["checkpoint"], mining["split_record"], provenance_reviewed=True
        )


def test_nested_checkpoint_split_fingerprint_recorded(mining):
    payload = torch.load(mining["checkpoint"], weights_only=False)
    payload["split_manifest"] = mining["split_record"]
    torch.save(payload, mining["checkpoint"])
    _, record = load_predictor(mining["checkpoint"], mining["split_record"])
    assert record["training_provenance"] == "fingerprint_matched"
    payload["split_set_sha256"] = "contradicting_flat_field"
    torch.save(payload, mining["checkpoint"])
    with pytest.raises(ValueError, match="contradict"):
        load_predictor(mining["checkpoint"], mining["split_record"])


def test_nonfloating_checkpoint_weights_rejected(mining):
    payload = torch.load(mining["checkpoint"], weights_only=False)
    payload["model_state_dict"]["network.9.bias"] = torch.zeros(1, dtype=torch.int64)
    torch.save(payload, mining["checkpoint"])
    with pytest.raises(ValueError, match="incompatible"):
        load_predictor(mining["checkpoint"], mining["split_record"])


def test_cache_with_conflicting_preprocessing_metadata_rejected(mining):
    path = mining["root"] / "conflicting_cache.pt"
    torch.save(
        dict(
            embeddings={"AAAAC": torch.zeros(1024)},
            model_id="Rostlab/prot_t5_xl_half_uniref50-enc",
            max_residues=1024,
            embedding_dim=1024,
            pooling="mean over non-special residue tokens",
            normalization="spaces_removed",
        ),
        path,
    )
    with pytest.raises(ValueError, match="preprocessing"):
        load_mining_caches([path])


def test_infinite_logits_do_not_become_finite_sigmoid_scores(mining, monkeypatch):
    monkeypatch.setattr(
        mining["model"],
        "forward",
        lambda features: torch.full((len(features),), float("inf")),
    )
    with pytest.raises(ValueError, match="before sigmoid"):
        scored(mining)
    assert not mining["scores"].exists()
    assert not mining["scores"].with_suffix(".manifest.json").exists()


def test_inconsistent_duplicate_cannot_be_hidden_by_threshold(mining):
    scored(mining)
    scores = pd.read_csv(mining["scores"])
    scores.loc[1, "score"] = 0.9  # reversed copy of row 0; now above the cutoff
    scores.to_csv(mining["scores"], index=False)
    manifest_path = mining["scores"].with_suffix(".manifest.json")
    record = json.loads(manifest_path.read_text())
    record["score_file_sha256"] = file_sha256(mining["scores"])
    manifest_path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="inconsistent scores"):
        select_candidates(
            mining["scores"],
            mining["splits"],
            mining["split_record"],
            provenance_reviewed=True,
            chunk_size=1,
        )
