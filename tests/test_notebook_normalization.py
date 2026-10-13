"""CPU-only regression checks for normalization and isolated ablation caches.

Run with: python -m unittest discover -s tests -p test_notebook_normalization.py
No encoder downloads, GPU, or PyTorch installation required.
"""
import ast
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import re
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / 'notebooks'
BROAD = 'canonical_X_spaces_removed_prefix_v1'
LEGACY = 'uzob_only_prefix_v1'


def sources(name):
    notebook = json.loads((NOTEBOOKS / name).read_text())
    return [''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code']


def definitions(name, names, namespace):
    nodes = []
    for source in sources(name):
        if any(line.lstrip().startswith(('!', '%')) for line in source.splitlines()):
            continue
        for node in ast.parse(source).body:
            if isinstance(node, ast.FunctionDef) and node.name in names:
                nodes.append(node)
    assert {node.name for node in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), name, 'exec'), namespace)
    return namespace


def normalize(name):
    return definitions(name, ['normalize_sequence'], {'re': re, 'MAX_RESIDUES': 1024})['normalize_sequence']


class NormalizationTests(unittest.TestCase):
    main = ['01_prott5_predictor.ipynb', '02_esm2_benchmark.ipynb', '03_esmc_benchmark.ipynb']
    legacy = ['ablations/01_prott5_predictor_uzob_original.ipynb',
              'ablations/02_esm2_uzob_ablation.ipynb', 'ablations/03_esmc_uzob_ablation.ipynb']

    def test_main_rules_agree_on_noncanonical_characters_and_spaces(self):
        for name in self.main:
            fn = normalize(name)
            self.assertEqual(fn('acdefghiklmnpqrstvwy'), 'ACDEFGHIKLMNPQRSTVWY')
            self.assertEqual(fn('a uZobJx-*?.\t\nc'), 'A' + 'X' * 12 + 'C')
            self.assertEqual(fn('   '), '')
            self.assertEqual(fn('A ' * 1024 + 'C'), 'A' * 1024)

    def test_ablation_rules_match_exact_original(self):
        for name in self.legacy:
            fn = normalize(name)
            self.assertEqual(fn('a uZobJx-*?.\t\nc'), 'A XXXXJX-*?.\t\nC')
            self.assertEqual(fn('A ' * 1024 + 'C'), 'A ' * 512)
            self.assertEqual(fn('ACDEFGHIKLMNPQRSTVWY'), 'ACDEFGHIKLMNPQRSTVWY')

    def test_original_prott5_is_byte_for_byte_preserved(self):
        manifest = json.loads((ROOT / 'docs/cleanup_manifest.json').read_text())
        item = manifest['notebooks'][0]
        self.assertEqual(hashlib.sha256((ROOT / item['preserved_path']).read_bytes()).hexdigest(),
                         item['cleaned_file_sha256'])

    def test_main_prott5_payload_records_broad_policy(self):
        ns = definitions(self.main[0], ['embedding_payload'], {
            'MODEL_ID': 'prott5', 'NORMALIZATION': BROAD, 'MAX_RESIDUES': 1024})
        payload = ns['embedding_payload']({'ACD': [1, 2]})
        self.assertEqual(payload['normalization'], BROAD)
        self.assertEqual(payload['embeddings'], {'ACD': [1, 2]})

    def test_ablation_cache_load_rejects_broad_or_missing_policy(self):
        # Serialize real metadata through the notebook's save/load code, replacing
        # only PyTorch's file codec with pickle so no model dependencies are needed.
        def save(payload, path):
            Path(path).write_bytes(pickle.dumps(payload))
        def load(path, **kwargs):
            return pickle.loads(Path(path).read_bytes())
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            for name in self.legacy[1:]:
                ns = definitions(name, ['atomic_torch_save', 'load_trusted_torch_file',
                                        'save_embedding_cache', 'load_embedding_cache'], {
                    'Path': Path, 'os': os, 'torch': SimpleNamespace(save=save, load=load),
                    'NORMALIZATION': LEGACY, 'MODEL_ID': name, 'MAX_RESIDUES': 1024,
                    'EMBEDDING_DIM': 3, 'EMBEDDING_CACHE_PATH': Path(tmp) / Path(name).stem / 'cache.pt'})
                self.assertEqual(ns['load_embedding_cache'](), {})
                ns['save_embedding_cache']({'ACD': [1, 2, 3]})
                self.assertEqual(ns['load_embedding_cache'](), {'ACD': [1, 2, 3]})
                payload = load(ns['EMBEDDING_CACHE_PATH'])
                for policy in [BROAD, None]:
                    wrong = dict(payload)
                    if policy is None:
                        wrong.pop('normalization')
                    else:
                        wrong['normalization'] = policy
                    save(wrong, ns['EMBEDDING_CACHE_PATH'])
                    with self.assertRaisesRegex(ValueError, 'normalization'):
                        ns['load_embedding_cache']()

    def test_ablation_output_paths_do_not_touch_baselines(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            for name in self.legacy[1:]:
                def sandbox_path(value):
                    return Path(tmp) / str(value).lstrip('/')
                ns = {'Path': sandbox_path, 'drive': SimpleNamespace(mount=lambda *_: None),
                      'torch': SimpleNamespace(), 'DEVICE': SimpleNamespace(type='cpu')}
                config = next(s for s in sources(name) if 'PROJECT_ROOT = ' in s)
                exec(config, ns)
                self.assertEqual(ns['SPLITS_DIR'], ns['PROJECT_ROOT'] / 'splits')
                self.assertEqual(ns['ROOT'], ns['PROJECT_ROOT'] / 'ablations' / LEGACY)
                for var in ['EMBEDDING_CACHE_PATH', 'BEST_CHECKPOINT_PATH']:
                    self.assertTrue(ns[var].is_relative_to(ns['ROOT']))
                self.assertTrue(ns['TABLES_DIR'].is_relative_to(ns['ROOT']))
                self.assertTrue(ns['FIGURES_DIR'].is_relative_to(ns['ROOT']))

    def test_main_prott5_reuses_existing_partitions_without_writes(self):
        import pandas as pd
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            paths = {name: Path(tmp) / f'{name}.csv' for name in ('train', 'validation', 'test')}
            for path in paths.values():
                path.write_text('partner_A_sequence,partner_B_sequence,label\na u,CD,1\nJ?,EF,0\n')
            before = {key: path.read_bytes() for key, path in paths.items()}
            ns = {'USE_EXISTING_SPLITS': True, 'SPLIT_PATHS': paths, 'pd': pd, 'np': np}
            load_cell = next(s for s in sources(self.main[0]) if 'def load_shared_split' in s)
            split_cell = next(s for s in sources(self.main[0]) if s.startswith('if not USE_EXISTING_SPLITS:') and 'train_test_split' in s)
            exec(load_cell, ns)
            exec(split_cell, ns)  # Must not require the source dataset or write CSVs.
            self.assertEqual(ns['train_df'].iloc[0]['partner_A_sequence'], 'a u')
            self.assertEqual(before, {key: path.read_bytes() for key, path in paths.items()})


if __name__ == '__main__':
    unittest.main()
