"""Low-cost, fail-closed preparation for a matched SNOOPPI benchmark.

Uses existing repository split/audit helpers; never trains or downloads encoders.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'drafting_code'))
from snooppi_utils import (  # noqa: E402
    SEQUENCE_A, SEQUENCE_B, full_sequence_id, file_sha256, load_splits,
    split_manifest, save_json, leakage_report, preprocessing_report,
)
from snooppi_protocol import (  # noqa: E402
    IDENTITY_SCHEME, prepare_source_pairs, mapped_component_frame,
    grouped_three_way_split, mapped_overlap_report, summarize_protocol,
    write_protocol,
)


def write_csv(frame, path):
    frame.to_csv(path, index=False)


def verified_manifest(run):
    manifest = json.loads((run / 'preparation.json').read_text())
    for name, expected in manifest['artifact_sha256'].items():
        if file_sha256(run / name) != expected:
            raise ValueError(f'Prepared artifact changed: {name}')
    return manifest


def prepare(source, run, revision):
    source, run = Path(source).resolve(), Path(run).resolve()
    if not revision.strip():
        raise ValueError('Record the dataset revision or explicit historical provenance')
    if run == source or source in run.parents:
        raise ValueError('Run directory must be separate from source splits')
    if run.exists():
        raise FileExistsError(run)
    paths = {'splits': source}
    source_record = split_manifest(paths)
    splits = load_splits(paths)
    if source_record != split_manifest(paths):
        raise ValueError('Source files changed during loading')
    # Inputs are protein sequences, not PTM markup. Preserve raw keys for caches.
    for frame in splits.values():
        for col in (SEQUENCE_A, SEQUENCE_B):
            for seq in frame[col]:
                full = seq.upper().replace(' ', '')
                if not re.fullmatch('[A-Z]+', full) or not set(full) & set('ACDEFGHIKLMNPQRSTVWY'):
                    raise ValueError('Non-protein markup/whitespace or all-unknown input needs review')
    run.mkdir(parents=True)
    write_csv(leakage_report(splits), run / 'source_overlap.csv')
    save_json(preprocessing_report(splits), run / 'source_preprocessing.json')
    # Conflicts stop; all copies of colliding/duplicate pairs are quarantined.
    # Do not resolve contradictory experimental labels by majority vote.
    try:
        pairs, quarantine = prepare_source_pairs(splits)
    except ValueError as error:
        save_json({'status': 'blocked', 'reason': str(error)}, run / 'status.json')
        raise
    write_csv(pairs, run / 'pairs.csv')
    write_csv(quarantine, run / 'quarantine.csv')
    sequences = {}
    for col in (SEQUENCE_A, SEQUENCE_B):
        for seq in pairs[col]:
            full = seq.upper().replace(' ', '')
            sequences[full_sequence_id(seq)] = re.sub('[^ACDEFGHIKLMNPQRSTVWY]', 'X', full)
    with (run / 'proteins.fasta').open('w') as stream:
        for identity, sequence in sorted(sequences.items()):
            stream.write(f'>{identity}\n{sequence}\n')
    artifacts = ['pairs.csv', 'quarantine.csv', 'proteins.fasta',
                 'source_overlap.csv', 'source_preprocessing.json']
    save_json({
        'source_manifest': source_record, 'dataset_revision': revision,
        'code_sha256': {str(p.relative_to(REPO)): file_sha256(p) for p in [
            Path(__file__), REPO / 'drafting_code/snooppi_utils.py',
            REPO / 'drafting_code/snooppi_protocol.py']},
        'retained_pairs': len(pairs), 'quarantined_rows': len(quarantine),
        'unique_full_sequences': len(sequences),
        'clustering_input': 'full length; uppercase; remove ASCII spaces; noncanonical to X',
        'model_input': 'same normalization then first 1024 residues',
        'artifact_sha256': {p: file_sha256(run / p) for p in artifacts},
    }, run / 'preparation.json')
    return run


def command(run, executable, threads):
    # One prespecified protocol, not a sweep. Symmetric coverage, connected components.
    return [executable, 'easy-cluster', str(run / 'proteins.fasta'),
            str(run / 'mmseqs'), str(run / 'tmp'), '--min-seq-id', '0.3',
            '-c', '0.8', '--cov-mode', '0', '--cluster-mode', '1',
            '-s', '7.5', '--threads', str(threads)]


def cluster(run, threads):
    run = Path(run).resolve()
    verified_manifest(run)
    if threads < 1 or threads > 3:
        raise ValueError('Use 1–3 threads for this low-cost run')
    if list(run.glob('mmseqs*')) or (run / 'clustering.json').exists():
        raise FileExistsError('Clustering artifacts already exist; use a fresh run')
    executable = shutil.which('mmseqs')
    if not executable:
        raise RuntimeError('MMseqs2 must be installed before clustering')
    version = subprocess.check_output([executable, 'version'], text=True).strip()
    args = command(run, executable, threads)
    with (run / 'mmseqs.log').open('w') as log:
        subprocess.run(args, check=True, stdout=log, stderr=subprocess.STDOUT)
    save_json({'version': version, 'command': args,
               'fasta_sha256': file_sha256(run / 'proteins.fasta'),
               'tsv_sha256': file_sha256(run / 'mmseqs_cluster.tsv')},
              run / 'clustering.json')


def read_mapping(run):
    record = json.loads((run / 'clustering.json').read_text())
    if record['fasta_sha256'] != file_sha256(run / 'proteins.fasta'):
        raise ValueError('Clustering FASTA mismatch')
    if record['tsv_sha256'] != file_sha256(run / 'mmseqs_cluster.tsv'):
        raise ValueError('Clustering TSV mismatch')
    expected = {line[1:].strip() for line in (run / 'proteins.fasta').read_text().splitlines() if line.startswith('>')}
    table = pd.read_csv(run / 'mmseqs_cluster.tsv', sep='\t', header=None,
                        names=['cluster_id', 'sequence_sha256'], dtype=str)
    if table.isna().any().any() or table.sequence_sha256.duplicated().any():
        raise ValueError('Missing or repeated cluster membership')
    if set(table.sequence_sha256) != expected or not set(table.cluster_id) <= expected:
        raise ValueError('Cluster membership does not match the prepared FASTA')
    lookup = table.set_index('sequence_sha256').cluster_id
    if any(lookup[representative] != representative for representative in set(table.cluster_id)):
        raise ValueError('Cluster representatives must belong to their own clusters')
    table['identity_scheme'] = IDENTITY_SCHEME
    table['evidence_source'] = f"MMseqs2 {record['version']}; TSV SHA256 {record['tsv_sha256']}"
    return table, record


def freeze(run, destination):
    run, destination = Path(run).resolve(), Path(destination).resolve()
    prep = verified_manifest(run)
    if destination.exists():
        raise FileExistsError(destination)
    mapping, clustering = read_mapping(run)
    pairs = pd.read_csv(run / 'pairs.csv')
    frame = mapped_component_frame(pairs, mapping, 'cluster_id')
    sizes = frame.groupby('mapped_component').size().sort_values(ascending=False)
    write_csv(sizes.rename('pairs').reset_index(), run / 'component_sizes.csv')
    write_csv(mapping, run / 'mmseqs2_clusters.csv')
    try:
        parts = grouped_three_way_split(frame, 'mapped_component', seed=44)
        summary = summarize_protocol('cold_cluster', parts, 'mapped_component')
        write_csv(summary, run / 'split_preview.csv')
        targets = {'train': .8, 'validation': .1, 'test': .1}
        if any(abs(row.actual_row_fraction - targets[row.split]) > .05
               for row in summary.itertuples()):
            raise ValueError('Row fractions differ from 80/10/10 by >5 percentage points; review component feasibility')
        overlaps = mapped_overlap_report(parts, mapping, 'cluster_id')
        if overlaps.overlapping_cluster_id.any():
            raise ValueError('Cluster leakage')
        result = write_protocol(destination, parts, prep['source_manifest'], {
            'name': 'cold_cluster', 'group_column': 'mapped_component',
            'mapping_field': 'cluster_id',
            'seed': 44, 'fractions': targets, 'clustering': clustering,
            'dataset_revision': prep['dataset_revision'],
            'normalization': prep['model_input'],
            'assignment': 'whole interaction components; no cross-component edge dropping',
        }, {'cluster_overlap.csv': overlaps,
            'source_quarantine.csv': pd.read_csv(run / 'quarantine.csv')})
    except ValueError as error:
        save_json({'status': 'blocked_split_feasibility', 'reason': str(error)}, run / 'status.json')
        raise
    save_json({'status': 'splits_frozen_not_trained',
               'split_set_sha256': result['split_set_sha256']}, run / 'status.json')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='stage', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--dataset-revision', required=True)
    p = sub.add_parser('cluster')
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--threads', type=int, default=2)
    p = sub.add_parser('freeze')
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare(args.source, args.run, args.dataset_revision)
    elif args.stage == 'cluster':
        cluster(args.run, args.threads)
    else:
        freeze(args.run, args.destination)


if __name__ == '__main__':
    main()
