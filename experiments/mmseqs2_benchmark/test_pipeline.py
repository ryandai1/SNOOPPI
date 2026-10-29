"""Synthetic protocol tests, not MMseqs2 or scientific performance validation."""
import json
import random
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from pipeline import (prepare, freeze, read_mapping, verified_manifest, command,
                      file_sha256, load_splits, SEQUENCE_A, SEQUENCE_B)


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        rng = random.Random(9)
        rows = []
        for _ in range(100):
            a, b, c = [''.join(rng.choices('ACDEFGHIKLMNPQRSTVWY', k=40)) for _ in range(3)]
            rows.extend([{SEQUENCE_A: a, SEQUENCE_B: b, 'label': 0},
                         {SEQUENCE_A: a, SEQUENCE_B: c, 'label': 1}])
        self.frame = pd.DataFrame(rows)
        for name, part in [('train', self.frame.iloc[:160]),
                           ('validation', self.frame.iloc[160:180]),
                           ('test', self.frame.iloc[180:])]:
            part.to_csv(self.source / f'{name}.csv', index=False)
        self.run = self.root / 'run'

    def fixture_mapping(self, giant=False):
        ids = [s[1:] for s in (self.run / 'proteins.fasta').read_text().splitlines() if s.startswith('>')]
        (self.run / 'mmseqs_cluster.tsv').write_text(''.join(
            f'{ids[0] if giant else s}\t{s}\n' for s in ids))
        (self.run / 'clustering.json').write_text(json.dumps({
            'version': 'SYNTHETIC_FIXTURE_NOT_MMSEQS', 'command': [],
            'fasta_sha256': file_sha256(self.run / 'proteins.fasta'),
            'tsv_sha256': file_sha256(self.run / 'mmseqs_cluster.tsv')}))

    def test_end_to_end_disjoint_and_reloadable(self):
        before = {p.name: file_sha256(p) for p in self.source.glob('*.csv')}
        prepare(self.source, self.run, 'synthetic')
        self.fixture_mapping()
        dest = self.root / 'splits_ablations' / 'fixture'
        result = freeze(self.run, dest)
        parts = load_splits({'splits': dest})
        self.assertEqual(sum(map(len, parts.values())), 200)
        sets = [{s for col in (SEQUENCE_A, SEQUENCE_B) for s in frame[col]}
                for frame in parts.values()]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        self.assertEqual(before, {p.name: file_sha256(p) for p in self.source.glob('*.csv')})
        self.assertEqual(len(result['split_set_sha256']), 64)
        with self.assertRaises(FileExistsError):
            freeze(self.run, dest)

    def test_conflicts_stop_before_fasta(self):
        frame = pd.read_csv(self.source / 'test.csv')
        extra = self.frame.iloc[[0]].copy()
        extra['label'] = 1
        pd.concat([frame, extra]).to_csv(self.source / 'test.csv', index=False)
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            prepare(self.source, self.run, 'synthetic')
        self.assertFalse((self.run / 'proteins.fasta').exists())

    def test_swapped_duplicates_quarantined(self):
        frame = pd.read_csv(self.source / 'test.csv')
        extra = self.frame.iloc[[0]].copy()
        extra[[SEQUENCE_A, SEQUENCE_B]] = extra[[SEQUENCE_B, SEQUENCE_A]].to_numpy()
        pd.concat([frame, extra]).to_csv(self.source / 'test.csv', index=False)
        prepare(self.source, self.run, 'synthetic')
        self.assertEqual(len(pd.read_csv(self.run / 'quarantine.csv')), 2)
        self.assertEqual(len(pd.read_csv(self.run / 'pairs.csv')), 199)

    def test_tampering_rejected(self):
        prepare(self.source, self.run, 'synthetic')
        with (self.run / 'pairs.csv').open('a') as f:
            f.write('\n')
        with self.assertRaisesRegex(ValueError, 'changed'):
            verified_manifest(self.run)

    def test_incomplete_mapping_rejected(self):
        prepare(self.source, self.run, 'synthetic')
        self.fixture_mapping()
        p = self.run / 'mmseqs_cluster.tsv'
        p.write_text('\n'.join(p.read_text().splitlines()[1:]) + '\n')
        manifest = json.loads((self.run / 'clustering.json').read_text())
        manifest['tsv_sha256'] = file_sha256(p)
        (self.run / 'clustering.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'membership'):
            read_mapping(self.run)

    def test_giant_component_does_not_freeze(self):
        prepare(self.source, self.run, 'synthetic')
        self.fixture_mapping(giant=True)
        dest = self.root / 'splits_ablations' / 'fixture'
        with self.assertRaises(ValueError):
            freeze(self.run, dest)
        self.assertFalse(dest.exists())

    def test_full_length_and_protected_source(self):
        frame = pd.read_csv(self.source / 'train.csv')
        frame.loc[0, SEQUENCE_A] = 'a ' * 1100 + 'U'
        frame.to_csv(self.source / 'train.csv', index=False)
        prepare(self.source, self.run, 'synthetic')
        sequences = (self.run / 'proteins.fasta').read_text().splitlines()[1::2]
        self.assertIn('A' * 1100 + 'X', sequences)
        with self.assertRaises(FileExistsError):
            prepare(self.source, self.run, 'synthetic')
        with self.assertRaises(ValueError):
            prepare(self.source, self.source / 'new', 'synthetic')

    def test_command_parameters(self):
        args = command(self.run, 'mmseqs', 2)
        for key, value in [('--min-seq-id', '0.3'), ('--cov-mode', '0'),
                           ('-c', '0.8'), ('--cluster-mode', '1'), ('--threads', '2')]:
            self.assertEqual(args[args.index(key) + 1], value)


if __name__ == '__main__':
    unittest.main()
