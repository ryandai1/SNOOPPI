"""Scientific protocol regressions; synthetic fixtures contain no research results."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "drafting_code"))
from snooppi_utils import (
    LABEL,
    SEQUENCE_A as A,
    SEQUENCE_B as B,
    normalize_split_columns,
    load_splits,
    experiment_paths,
    cache_paths,
    ENCODER_SPECS,
    load_embedding_cache,
    validate_cache_coverage,
    assert_model_ready,
    build_pair_features,
    add_pair_ids,
    prediction_identity,
    align_paired_predictions,
    validate_prediction_identity,
    split_manifest,
    full_sequence_id,
    require_comparable_preprocessing,
)
from snooppi_protocol import (
    IDENTITY_SCHEME,
    prepare_source_pairs,
    cold_target_split,
    grouped_three_way_split,
    interaction_components,
    mapped_component_frame,
    mapped_overlap_report,
    load_reviewed_mapping,
    write_protocol,
    matched_candidate_review,
)


def sequence(index):
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    digits = []
    for _ in range(5):
        digits.append(alphabet[index % 20])
        index //= 20
    return "".join(digits) + "ACDEFGHIKLMNPQRSTVWY"


def synthetic_pairs(groups=100):
    rows = []
    for group in range(groups):
        for label in (0, 1):
            rows.append(
                {
                    A: sequence(3 * group),
                    B: sequence(3 * group + label + 1),
                    LABEL: label,
                    "SNOOPPI_final_label": "positive" if label else "negative",
                }
            )
    return pd.DataFrame(rows)


def make_fixture(root):
    root = Path(root)
    (root / "splits").mkdir(parents=True)
    frame = synthetic_pairs()
    for name, part in [
        ("train", frame.iloc[:160]),
        ("validation", frame.iloc[160:180]),
        ("test", frame.iloc[180:]),
    ]:
        part.to_csv(root / "splits" / f"{name}.csv", index=False)
    paths = experiment_paths(root)
    paths["embeddings"].mkdir()
    sequences = sorted(set(frame[A]) | set(frame[B]))
    rng = np.random.default_rng(2026)
    for name, spec in ENCODER_SPECS.items():
        cache = {
            s: torch.from_numpy(
                rng.normal(size=spec["embedding_dim"]).astype(np.float16)
            )
            for s in sequences
        }
        if name == "ProtT5":
            payload = cache
        else:
            payload = {k: spec[k] for k in ("model_id", "embedding_dim")}
            payload.update(
                max_residues=1024,
                pooling="mean over non-special residue tokens",
                embeddings=cache,
            )
            if name == "ESM-C 600M":
                payload.update(
                    normalization="canonical_X_spaces_removed_prefix_v1",
                    esm_version="3.2.1",
                )
        torch.save(payload, cache_paths(paths)[name])
    (root / "raw").mkdir()
    sequence_groups = {sequence(i): str(i // 3) for i in range(300)}
    mapping = pd.DataFrame(
        [
            {
                "sequence_sha256": full_sequence_id(s),
                "cluster_id": sequence_groups[s],
                "identity_scheme": IDENTITY_SCHEME,
                "evidence_source": "synthetic_test_only",
                "family": sequence_groups[s],
                "species": "fixture",
                "localization": "fixture",
            }
            for i, s in enumerate(sequences)
        ]
    )
    mapping.to_csv(root / "raw" / "mmseqs2_clusters.csv", index=False)
    mapping.to_csv(root / "raw" / "protein_metadata.csv", index=False)
    return paths


@pytest.mark.parametrize("bad", [0.5, 256, -256, np.nan, np.inf])
def test_invalid_labels_rejected_before_cast(bad):
    frame = pd.DataFrame({A: ["ACD"], B: ["EFG"], LABEL: [bad]})
    with pytest.raises(ValueError):
        normalize_split_columns(frame, Path("test.csv"))


def test_unknown_conflicting_labels_and_empty_sequences():
    for extra in [
        {LABEL: [0], "SNOOPPI_final_label": ["unknown"]},
        {LABEL: [0], "SNOOPPI_final_label": ["positive"]},
        {LABEL: [1], A: [" "]},
    ]:
        frame = pd.DataFrame({A: ["ACD"], B: ["EFG"], **extra})
        with pytest.raises(ValueError):
            normalize_split_columns(frame, Path("test.csv"))


def test_path_selection_and_stale_manifest(tmp_path, monkeypatch):
    paths = make_fixture(tmp_path)
    splits = load_splits(paths)
    all_pairs, _ = prepare_source_pairs(splits)
    all_pairs["interaction_component"] = interaction_components(all_pairs)
    parts = grouped_three_way_split(all_pairs, "interaction_component")
    dest = tmp_path / "splits_ablations" / "cold_protein_test"
    write_protocol(
        dest,
        parts,
        split_manifest(paths),
        {"name": "cold_protein", "group_column": "interaction_component"},
    )
    monkeypatch.setenv("SNOOPPI_SPLIT_DIR", "splits_ablations/cold_protein_test")
    selected = experiment_paths(tmp_path)
    assert selected["splits"] == dest
    assert_model_ready(load_splits(selected))
    with (dest / "test.csv").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="hash"):
        load_splits(selected)
    with pytest.raises(FileExistsError):
        write_protocol(dest, parts, split_manifest(paths), {})
    with pytest.raises(ValueError):
        write_protocol(tmp_path / "splits", parts, split_manifest(paths), {})


def test_pair_duplicates_and_conflicts(tmp_path):
    paths = make_fixture(tmp_path)
    splits = load_splits(paths)
    reversed_row = splits["train"].iloc[[0]].copy()
    reversed_row[[A, B]] = reversed_row[[B, A]].to_numpy()
    splits["test"] = pd.concat([splits["test"], reversed_row], ignore_index=True)
    with pytest.raises(ValueError):
        assert_model_ready(splits)
    unique, quarantine = prepare_source_pairs(splits)
    assert len(quarantine) == 2 and len(unique) == 199
    splits["test"].iloc[-1, splits["test"].columns.get_loc(LABEL)] = 1
    with pytest.raises(ValueError, match="conflict"):
        prepare_source_pairs(splits)


def test_cold_target_cannot_reappear_as_other_partner():
    # Target identities appear in the other role in many rows.
    frame = synthetic_pairs(100)
    frame.loc[::4, A] = frame.loc[2::4, B].to_numpy()
    frame = add_pair_ids(frame)
    parts, rejected = cold_target_split(frame)
    for early, later in [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ]:
        early_proteins = set(parts[early].protein_A_id) | set(parts[early].protein_B_id)
        assert early_proteins.isdisjoint(set(parts[later].protein_B_id))
    assert len(rejected) > 0


def test_giant_component_and_custom_fractions():
    frame = add_pair_ids(synthetic_pairs())
    frame["component"] = interaction_components(frame)
    parts = grouped_three_way_split(
        frame, "component", fractions={"train": 0.6, "validation": 0.2, "test": 0.2}
    )
    assert [len(parts[n]) for n in ("train", "validation", "test")] == [120, 40, 40]
    frame["component"] = "giant"
    with pytest.raises(ValueError, match="three"):
        grouped_three_way_split(frame, "component")


def test_cluster_both_roles_and_mapping_coverage(tmp_path):
    paths = make_fixture(tmp_path)
    splits = load_splits(paths)
    frame, _ = prepare_source_pairs(splits)
    mapping = load_reviewed_mapping(paths["clusters"], "cluster_id", frame)
    # Map cluster IDs to known graph components so the fixture itself is feasible.
    mapping["cluster_id"] = mapping.sequence_sha256.map(
        {full_sequence_id(sequence(i)): str(i // 3) for i in range(300)}
    )
    mapped = mapped_component_frame(frame, mapping, "cluster_id")
    parts = grouped_three_way_split(mapped, "mapped_component")
    assert (
        not mapped_overlap_report(parts, mapping, "cluster_id")
        .filter(like="overlapping_")
        .to_numpy()
        .any()
    )
    mapping.iloc[:-1].to_csv(paths["clusters"], index=False)
    with pytest.raises(ValueError, match="misses"):
        load_reviewed_mapping(paths["clusters"], "cluster_id", frame)


def test_cache_metadata_coverage_and_symmetry(tmp_path):
    paths = make_fixture(tmp_path)
    splits = load_splits(paths)
    for name, path in cache_paths(paths).items():
        cache, metadata = load_embedding_cache(path, name)
        validate_cache_coverage(cache, splits)
        frame = splits["test"]
        swapped = frame.copy()
        swapped[[A, B]] = swapped[[B, A]].to_numpy()
        assert np.array_equal(
            build_pair_features(frame, cache), build_pair_features(swapped, cache)
        )
    payload = torch.load(cache_paths(paths)["ESM-2 650M"], weights_only=False)
    payload["model_id"] = "wrong"
    torch.save(payload, cache_paths(paths)["ESM-2 650M"])
    with pytest.raises(ValueError, match="model_id"):
        load_embedding_cache(cache_paths(paths)["ESM-2 650M"], "ESM-2 650M")
    cache.pop(next(iter(cache)))
    with pytest.raises(KeyError):
        validate_cache_coverage(cache, splits)


def test_cached_preprocessing_disagreement(tmp_path):
    paths = make_fixture(tmp_path)
    splits = load_splits(paths)
    splits["test"].loc[0, A] = "A CD"
    with pytest.raises(ValueError, match="different"):
        require_comparable_preprocessing(splits, ["ProtT5", "ESM-2 650M"])


def test_predictions_reject_partial_pairs_and_wrong_identity(tmp_path):
    paths = make_fixture(tmp_path)
    splits = load_splits(paths)
    expected = prediction_identity(splits["test"], split_manifest(paths))
    a = expected.assign(
        probability=np.linspace(0.1, 0.9, len(expected)),
        threshold=0.5,
        quick_mode=False,
    )
    validate_prediction_identity(a, expected)
    align_paired_predictions(a, a.sample(frac=1, random_state=1))
    for b in [
        a.iloc[:-1],
        pd.concat([a, a.iloc[[0]]]),
        a.assign(split_sha256="different"),
    ]:
        with pytest.raises(ValueError):
            align_paired_predictions(a, b)
        with pytest.raises(ValueError):
            validate_prediction_identity(b, expected)


def test_unknown_candidates_training_only_and_symmetric():
    train = pd.DataFrame({A: ["ACD"], B: ["EFG"], LABEL: [1]})
    val = pd.DataFrame({A: ["HIK"], B: ["LMN"], LABEL: [1]})
    known = pd.concat([train, val])
    candidates = pd.DataFrame(
        {
            A: ["PQR", "STV", "ACD", "EFG"],
            B: ["WYA", "CDE", "EFG", "ACD"],
            "SNOOPPI_final_label": ["unknown"] * 4,
        }
    )
    strata = {
        "ACD": "x",
        "EFG": "y",
        "HIK": "z",
        "LMN": "w",
        "PQR": "y",
        "WYA": "x",
        "STV": "z",
        "CDE": "w",
    }
    metadata = pd.DataFrame(
        [
            {
                "sequence_sha256": full_sequence_id(s),
                "species": value,
                "identity_scheme": IDENTITY_SCHEME,
                "evidence_source": "fixture",
            }
            for s, value in strata.items()
        ]
    )
    matched, report = matched_candidate_review(
        candidates, metadata, train, known, "unknown_fixture"
    )
    assert len(matched) == 1 and matched.iloc[0][A] == "PQR"
    assert LABEL not in matched and "SNOOPPI_final_label" not in matched
    assert matched.candidate_status.eq("unknown_requires_review").all()
    assert report["excluded_known_or_duplicate"] == 2
    candidates[LABEL] = 0
    with pytest.raises(ValueError):
        matched_candidate_review(candidates, metadata, train, known, "fixture")


def test_same_test_with_different_training_is_not_comparable(tmp_path):
    paths = make_fixture(tmp_path)
    splits = load_splits(paths)
    expected = prediction_identity(splits["test"], split_manifest(paths))
    a = expected.assign(probability=0.5, threshold=0.5, quick_mode=False)
    # Change only training-file identity; test labels, order and scores are unchanged.
    with (paths["splits"] / "train.csv").open("a") as stream:
        stream.write("\n")
    b = prediction_identity(splits["test"], split_manifest(paths)).assign(
        probability=0.5, threshold=0.5, quick_mode=False
    )
    assert a.split_sha256.equals(b.split_sha256)
    with pytest.raises(ValueError):
        align_paired_predictions(a, b)


def test_prott5_space_prefix_collisions_are_quarantined(tmp_path):
    paths = make_fixture(tmp_path)
    splits = load_splits(paths)
    a = splits["train"].iloc[[0]].copy()
    b = a.copy()
    a[A] = "A " * 550 + "C"
    b[A] = "A " * 550 + "D"
    splits["train"] = pd.concat([splits["train"], a], ignore_index=True)
    splits["test"] = pd.concat([splits["test"], b], ignore_index=True)
    with pytest.raises(ValueError, match="ProtT5"):
        assert_model_ready(splits)
    unique, quarantine = prepare_source_pairs(splits)
    assert len(quarantine) == 2
    assert len(unique) == 200


def test_cold_target_also_excludes_prott5_prefix_identity():
    frame = synthetic_pairs(100)
    for index in range(12):
        frame.loc[index, B] = "A " * 550 + sequence(1000 + index)
    parts, _ = cold_target_split(add_pair_ids(frame))
    for earlier, later in [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ]:
        early_inputs = set(parts[earlier].prott5_protein_A_id) | set(
            parts[earlier].prott5_protein_B_id
        )
        assert early_inputs.isdisjoint(set(parts[later].prott5_protein_B_id))


def test_explicit_csv_filenames_drive_loading_and_hashes(tmp_path):
    paths = make_fixture(tmp_path)
    custom = tmp_path / "renamed_cached_csvs"
    custom.mkdir()
    files = {}
    for name in ("train", "validation", "test"):
        source = paths["splits"] / f"{name}.csv"
        destination = custom / f"my_saved_{name}_pairs.csv"
        source.rename(destination)
        files[name] = destination
    selected = dict(paths, split_files=files)
    splits = load_splits(selected)
    manifest = split_manifest(selected)
    assert manifest["files"] == {name: str(path) for name, path in files.items()}
    assert_model_ready(splits)
    for encoder, path in cache_paths(selected).items():
        cache, _ = load_embedding_cache(path, encoder)
        validate_cache_coverage(cache, splits)
    with files["validation"].open("a") as stream:
        stream.write("\n")
    assert manifest["split_set_sha256"] != split_manifest(selected)["split_set_sha256"]
    with pytest.raises(ValueError, match="different"):
        load_splits(
            dict(
                paths,
                split_files={
                    "train": files["test"],
                    "validation": files["validation"],
                    "test": files["test"],
                },
            )
        )


def test_explicit_plm_cache_filenames_are_used(tmp_path):
    paths = make_fixture(tmp_path)
    relocated = tmp_path / "my_plm_caches"
    relocated.mkdir()
    files = {}
    for encoder, path in cache_paths(paths).items():
        destination = relocated / ("saved_" + path.name)
        path.rename(destination)
        files[encoder] = str(destination)
    selected = dict(paths, cache_files=files)
    assert cache_paths(selected) == {
        encoder: Path(path) for encoder, path in files.items()
    }
    for encoder, path in cache_paths(selected).items():
        cache, _ = load_embedding_cache(path, encoder)
        validate_cache_coverage(cache, load_splits(selected))
