"""Auditable full-sequence MMseqs2 split preparation; never trains or edits sources."""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd

if not __debug__:
    raise RuntimeError('Run without -O: validation assertions must remain enabled')

A, B, Y = 'partner_A_sequence', 'partner_B_sequence', 'label'
NAMES = ('train', 'validation', 'test')


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def normalized(s, mode):
    if mode == 'full':
        return s.upper().replace(' ', '')
    if mode == 't5':
        return re.sub('[UZOB]', 'X', s.upper())[:1024]
    return re.sub('[^ACDEFGHIKLMNPQRSTVWY]', 'X', s.upper().replace(' ', ''))[:1024]


def pair_id(a, b, mode='full'):
    return digest('|'.join(sorted([digest(normalized(a, mode)), digest(normalized(b, mode))])))


def audit(frame):
    d = frame.copy()
    assert set([A, B, Y]).issubset(d), 'Missing source columns'
    assert d[Y].notna().all() and d[Y].isin([0, 1]).all(), 'Invalid binary label'
    if 'SNOOPPI_final_label' in d:
        expected = d['SNOOPPI_final_label'].str.lower().map({'positive': 1, 'negative': 0})
        assert expected.notna().all() and expected.eq(d[Y]).all(), 'Label semantics disagree'
    for c in [A, B]:
        assert d[c].map(lambda x: isinstance(x, str) and bool(x.strip())).all(), 'Invalid sequence'
        assert d[c].map(lambda x: bool(re.fullmatch('[A-Za-z ]+', x))).all(), 'Unexpected sequence characters'
    for mode in ['full', 't5', 'esm']:
        d[mode + '_pair_id'] = [pair_id(a, b, mode) for a, b in zip(d[A], d[B])]
    conflict = d.groupby('full_pair_id')[Y].nunique()
    bad = set(conflict[conflict > 1].index)
    quarantined = d[d.full_pair_id.isin(bad)].copy()
    eligible = d[~d.full_pair_id.isin(bad)].copy()
    duplicates = eligible.duplicated('full_pair_id')
    clean = eligible[~duplicates].copy()
    # Input collisions between distinct full-length pairs are conservatively quarantined.
    collision = clean.duplicated('t5_pair_id', keep=False) | clean.duplicated('esm_pair_id', keep=False)
    collisions = clean[collision].copy()
    clean = clean[~collision].copy()
    report = {'source_rows': len(d), 'conflicting_full_pairs': len(bad),
              'conflict_rows_quarantined': len(quarantined), 'same_label_duplicate_rows_removed': int(duplicates.sum()),
              'input_collision_pairs_quarantined': len(collisions), 'eligible_pairs': len(clean),
              'eligible_class_counts': clean[Y].value_counts().to_dict()}
    return clean, quarantined, collisions, report


class Groups:
    def __init__(self):
        self.p = {}
    def find(self, x):
        self.p.setdefault(x, x)
        while x != self.p[x]:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        self.p[max(a, b)] = min(a, b)


def partition(d, clusters):
    seqs = sorted(set(d[A]) | set(d[B]))
    assert set(clusters) == {digest(normalized(s, 'full')) for s in seqs}, 'Incomplete/extra cluster coverage'
    uf = Groups()
    seen = {}
    for s in seqs:
        sid = digest(normalized(s, 'full'))
        uf.union(sid, 'cluster:' + clusters[sid])
        for mode in ['t5', 'esm']:
            key = (mode, normalized(s, mode))
            if key in seen:
                uf.union(sid, seen[key])
            seen[key] = sid
    assignment = {}
    for s in seqs:
        group = uf.find(digest(normalized(s, 'full')))
        u = int(digest('44|' + group)[:16], 16) / 2**64
        assignment[s] = 'train' if u < .8 else 'validation' if u < .9 else 'test'
    d = d.copy()
    d['split'] = [assignment[a] if assignment[a] == assignment[b] else 'excluded_cross_partition' for a,b in zip(d[A], d[B])]
    # Interaction components report feasibility of retaining all pairs without dropping edges.
    graph = Groups()
    for a, b in zip(d[A], d[B]):
        graph.union(uf.find(digest(normalized(a,'full'))), uf.find(digest(normalized(b,'full'))))
    sizes = pd.Series([graph.find(uf.find(digest(normalized(a,'full')))) for a in d[A]]).value_counts()
    return d, {'interaction_components': len(sizes), 'largest_component_pairs': int(sizes.max()),
               'realized_counts': d.groupby(['split', Y]).size().unstack(fill_value=0).to_dict('index')}


def validate(d, clusters, min_per_class=50):
    parts = {n: d[d.split == n] for n in NAMES}
    for n, p in parts.items():
        assert all(int((p[Y] == y).sum()) >= min_per_class for y in [0,1]), f'{n} lacks adequate class counts'
    for i, n in enumerate(NAMES):
        for m in NAMES[i+1:]:
            for mode in ['full','t5','esm']:
                left = {normalized(s,mode) for c in [A,B] for s in parts[n][c]}
                right = {normalized(s,mode) for c in [A,B] for s in parts[m][c]}
                assert not left & right, f'{mode} endpoint overlap'
            left = {clusters[digest(normalized(s,'full'))] for c in [A,B] for s in parts[n][c]}
            right = {clusters[digest(normalized(s,'full'))] for c in [A,B] for s in parts[m][c]}
            assert not left & right, 'Cluster overlap'
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--partial-audit', action='store_true')
    args = p.parse_args()
    try:
        execute(args)
    except Exception as exc:
        if args.output.exists() and not (args.output/'frozen_splits.json').exists():
            state = args.output/'RUN_STATE.json'
            # Never replace an existing attempt's state when overwrite protection fired.
            if not str(exc).startswith('Refusing overwrite'):
                state.write_text(json.dumps({'status':'BLOCKED','reason':str(exc),'stage':'PREPARATION'},indent=2)+'\n')
        raise


def execute(args):
    assert not args.output.exists(), 'Refusing overwrite; preserve every attempt'
    paths = [args.source / (n + '.csv') for n in NAMES]
    if not args.partial_audit:
        assert all(f.exists() for f in paths), 'All three labeled source CSVs required'
    paths = [f for f in paths if f.exists()]
    assert paths, 'No sources'
    args.output.mkdir(parents=True)
    def save(name, value):
        (args.output/name).write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')
    save('protocol.json', {'version':2,'identity':0.3,'coverage':0.8,'coverage_mode':0,
        'cluster_mode':1,'single_step':True,'max_sequences_per_query':1000,'sensitivity':7.5,
        'threads':3,'seed':44,'group_fractions':[0.8,0.1,0.1],
        'minimum_per_class':50,'mmseqs_timeout_seconds':3600,
        'conflicts':'quarantine without relabeling','invalid_sequences':'quarantine without sanitization'})
    save('source_manifest.json', {f.name: {'sha256': file_hash(f), 'bytes': f.stat().st_size} for f in paths})
    d = pd.concat([pd.read_csv(f).assign(original_split=f.stem, source_row=lambda x: range(len(x))) for f in paths], ignore_index=True)
    valid = pd.Series(True, index=d.index)
    for c in [A,B]:
        valid &= d[c].map(lambda s: isinstance(s,str) and bool(re.fullmatch("[A-Za-z ]+",s)) and bool(s.strip()))
    d[~valid].to_csv(args.output/"invalid_sequences.csv", index=False)
    valid_rows=d[valid].copy()
    valid_rows['full_pair_id']=[pair_id(a,b) for a,b in zip(valid_rows[A],valid_rows[B])]
    valid_rows[valid_rows.full_pair_id.duplicated(keep=False)].to_csv(args.output/'duplicate_rows.csv',index=False)
    clean, conflicts, collisions, report = audit(valid_rows)
    report["source_rows_before_sequence_filter"] = len(d)
    report["invalid_sequence_rows_quarantined"] = int((~valid).sum())
    report['partial_only'] = args.partial_audit
    save('audit.json', report)
    for name, frame in [('eligible',clean),('conflicts',conflicts),('input_collisions',collisions)]:
        frame.to_csv(args.output/(name+'.csv'), index=False)
    if args.partial_audit:
        save('RUN_STATE.json', {'status':'BLOCKED','stage':'B','reason':'Partial sources; no splitting or model fitting permitted'})
        return
    sequences = sorted({normalized(s,'full') for c in [A,B] for s in clean[c]})
    fasta = args.output/'proteins.fasta'
    fasta.write_text(''.join('>'+digest(s)+'\n'+s+'\n' for s in sequences))
    assert shutil.which('mmseqs'), 'MMseqs2 unavailable; do not substitute identity-only clustering'
    version = subprocess.check_output(['mmseqs','version'], text=True).strip()
    cmd = ['mmseqs','easy-cluster',str(fasta),str(args.output/'mmseqs'),str(args.output/'tmp'),
           '--min-seq-id','0.3','-c','0.8','--cov-mode','0','--cluster-mode','1','--single-step-clustering','1','--max-seqs','1000',
           '-s','7.5','--threads','3']
    save('clustering.json', {'version':version,'command':cmd,'fasta_sha256':file_hash(fasta)})
    with open(args.output/'mmseqs.log','w') as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=3600)
    mapping = pd.read_csv(args.output/'mmseqs_cluster.tsv', sep='\t', header=None, names=['cluster_id','sequence_sha256'])
    assert not mapping.sequence_sha256.duplicated().any(), 'Ambiguous cluster mapping'
    mapping['identity_scheme'] = 'full_uppercase_spaces_removed_sha256_v1'
    mapping['evidence_source'] = 'clustering.json'
    mapping.to_csv(args.output/'mmseqs2_clusters.csv',index=False)
    clusters = dict(zip(mapping.sequence_sha256,mapping.cluster_id))
    assigned, feasibility = partition(clean,clusters)
    save('feasibility.json',feasibility)
    assigned.to_csv(args.output/'assignments.csv', index=False)
    validate(assigned,clusters)
    for n in NAMES:
        assigned[assigned.split == n].to_csv(args.output/(n+'.csv'),index=False)
    save('split_provenance.json', {'mapping_sha256':file_hash(args.output/'mmseqs2_clusters.csv'),
        'clustering_sha256':file_hash(args.output/'clustering.json'),
        'source_manifest_sha256':file_hash(args.output/'source_manifest.json'),
        'pipeline_sha256':file_hash(Path(__file__)),
        'protocol_sha256':file_hash(args.output/'protocol.json')})
    save('frozen_splits.json', {n: file_hash(args.output/(n+'.csv')) for n in NAMES})
    save('RUN_STATE.json',{'status':'SPLITS_READY','stage':'C','next_action':'Validate cache provenance and run train.py fit before train.py evaluate'})

if __name__ == '__main__':
    main()
