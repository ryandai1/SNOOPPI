"""Exercise the real installed MMseqs2 binary; never substitute a mocked clustering result."""
import random
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

class MMseqs(unittest.TestCase):
    @unittest.skipUnless(shutil.which('mmseqs'),'MMseqs2 not installed: integration remains unverified')
    def test_near_identical_and_unrelated_full_length_sequences(self):
        rng=random.Random(44)
        a=''.join(rng.choices('ACDEFGHIKLMNPQRSTVWY',k=150))
        b=a[:70]+('C' if a[70]!='C' else 'D')+a[71:]
        c=''.join(rng.choices('ACDEFGHIKLMNPQRSTVWY',k=150))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fasta=root/'sequences.fasta'
            fasta.write_text(f'>a\n{a}\n>b\n{b}\n>c\n{c}\n')
            with (root/'log').open('w') as log:
                subprocess.run(['mmseqs','easy-cluster',str(fasta),str(root/'result'),str(root/'tmp'),
                    '--min-seq-id','0.3','-c','0.8','--cov-mode','0','--cluster-mode','1',
                    '--single-step-clustering','1','--max-seqs','1000','-s','7.5','--threads','3'],
                    stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
            mapping={member:rep for rep,member in (line.split('\t') for line in (root/'result_cluster.tsv').read_text().splitlines())}
            self.assertEqual(set(mapping),{'a','b','c'})
            self.assertEqual(mapping['a'],mapping['b'])
            self.assertNotEqual(mapping['a'],mapping['c'])

if __name__=='__main__':unittest.main()
