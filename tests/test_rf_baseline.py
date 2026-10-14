"""Exercise the RF notebook with synthetic vectors; no Drive, PLM or torch needed.

Only .pt deserialization is substituted. Real cache validation, feature creation,
forest fitting, threshold selection, metrics, and exports execute as written.
These fixtures do not establish scientific performance.
"""
import ast
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'drafting_code'))
import snooppi_utils as utils

NOTEBOOK = ROOT / 'drafting_code/01_classical_embedding_baselines.ipynb'


def make_inputs(root):
    alphabet = 'ACDEFGHIKLMNPQRSTVWY'
    def sequence(number):
        chars = []
        for _ in range(5):
            chars.append(alphabet[number % 20])
            number //= 20
        return ''.join(chars) + alphabet
    frame = pd.DataFrame({
        utils.SEQUENCE_A: [sequence(i * 2) for i in range(150)],
        utils.SEQUENCE_B: [sequence(i * 2 + 1) for i in range(150)],
        'label': [int(i % 5 != 0) for i in range(150)],
    })
    (root / 'splits').mkdir(parents=True)
    for name, part in [('train', frame.iloc[:100]), ('validation', frame.iloc[100:125]), ('test', frame.iloc[125:])]:
        part.to_csv(root / 'splits' / f'{name}.csv', index=False)
    (root / 'embeddings').mkdir()
    (root / 'embeddings/prot_t5_embedding_cache.pt').write_bytes(b'SYNTHETIC DECODER FIXTURE ONLY')
    rng = np.random.default_rng(44)
    payload = {s: rng.normal(size=1024).astype(np.float32) for s in set(frame[utils.SEQUENCE_A]) | set(frame[utils.SEQUENCE_B])}
    return payload


@pytest.mark.parametrize('quick', [True, False])
def test_notebook_protocol(tmp_path, monkeypatch, quick):
    payload = make_inputs(tmp_path)
    monkeypatch.setenv('SNOOPPI_ROOT', str(tmp_path))
    monkeypatch.setenv('SNOOPPI_REPO_ROOT', str(ROOT))
    monkeypatch.setenv('SNOOPPI_SPLIT_DIR', 'splits')
    monkeypatch.delenv('SNOOPPI_CLUSTER_FILE', raising=False)
    monkeypatch.setattr(utils, '_load_trusted_torch_file', lambda path: payload)
    monkeypatch.setattr(plt, 'show', lambda: None)
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    state = {'__name__': '__main__'}
    for i, cell in enumerate(notebook.cells):
        if cell.cell_type != 'code':
            continue
        ast.parse(cell.source)
        exec(compile(cell.source, f'{NOTEBOOK.name}:{i}', 'exec'), state)
        if i == 3:
            # Exercise the actual reporting seeds/forest size. Bound only fixture rows.
            state.update(QUICK_MODE=quick, WRITE_OUTPUTS=True,
                         SEEDS=[44] if quick else [44, 45, 46],
                         MAX_TRAIN_ROWS=50 if quick else None,
                         RUN_ID='synthetic_development' if quick else 'synthetic_reporting')
        if i == 7 and quick:
            original_predict = state['positive_probability']
            def reject_test_prediction(model, features):
                assert features is state['x_validation'], 'Development mode predicted outside validation'
                return original_predict(model, features)
            state['positive_probability'] = reject_test_prediction

    assert state['RF_PARAMETERS']['n_estimators'] == 400
    assert state['CLASS_WEIGHTS'] == {0: 1.0, 1: 0.25}
    assert state['metrics_table']['feature_dimension'].eq(3072).all()
    assert set(state['metrics_table']['split']) == ({'validation'} if quick else {'train', 'validation', 'test'})
    assert state['metrics_table']['training_rows'].eq(50 if quick else 100).all()
    assert state['metrics_table']['quick_mode'].eq(quick).all()
    output_manifest = json.loads(state['output_files']['manifest'].read_text())
    assert output_manifest['selection']['test_evaluated'] is (not quick)
    assert 'no search' in output_manifest['selection']['hyperparameters']
    assert output_manifest['configuration']['class_weights'] == {'0': 1.0, '1': 0.25}
    for path in state['output_files'].values():
        assert path.is_file()
    if quick:
        assert state['predictions_table'].empty
        assert not list((tmp_path / 'tables/drafting_code').glob('*test_predictions.csv'))
    else:
        predictions = state['predictions_table']
        assert len(predictions) == 25 * 3
        for seed, per_seed in predictions.groupby('seed'):
            utils.validate_prediction_identity(per_seed, state['test_identity'])
            assert per_seed['threshold'].nunique() == 1
        assert len(state['confusion_rows']) == 3
        for row in state['confusion_rows']:
            assert sum(row[key] for key in ['tn', 'fp', 'fn', 'tp']) == 25
    with pytest.raises(FileExistsError):
        utils.check_output_paths(state['output_files'].values())
    plt.close('all')


def test_metrics_keep_extreme_score_rankings():
    labels = np.array([0, 1, 0, 1])
    scores = np.array([0.0, 1e-9, 1 - 1e-9, 1.0])
    metrics = utils.summarize_predictions(labels, scores, 1e-10)
    assert metrics['auroc'] == roc_auc_score(labels, scores)
    assert metrics['positive_ap'] == average_precision_score(labels, scores)
    assert np.isfinite(metrics['log_loss'])
    for invalid in [np.array([0, 1, np.nan, 1]), np.array([0, 1, 0, 1.1])]:
        with pytest.raises(ValueError):
            utils.summarize_predictions(labels, invalid, 0.5)


def test_threshold_uses_validation_macro_f1_grid():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.3, 0.7, 0.8])
    # First maximizing grid point is .305 (strictly above the second negative).
    assert utils.select_threshold(labels, probabilities) == pytest.approx(0.305)
