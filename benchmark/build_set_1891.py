"""Build the larger of the two evaluation sets: 1,891 triples.

A triple is allowed in on the same terms as in the smaller set: the source skeleton needs at
least three clips of the action and the target skeleton at least one. What changes is that
nothing is thrown away afterwards. Every source skeleton and action is paired with every other
skeleton that has that action, and there is no limit on how many triples result. The smaller
set is copied in first and keeps its query numbers, so results already produced for it stay
valid, and the new triples follow.

    python -m benchmark.build_set_1891 --out benchmark/sets/truebones_1891.json

Asking for only two source clips instead of three builds the wider version the appendix uses
to show the conclusion does not depend on where that line is drawn.
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from benchmark.action_taxonomy import parse_action_from_filename, action_to_cluster

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MOTIONS = 'dataset/truebones/zoo/truebones_processed/motions'
MAX_SRC = 5
SEED = 42


def index_cells(motion_dir):
    """Group the clip file names by skeleton and action, and list which skeletons have each
    action."""
    by_skel_action = defaultdict(list)
    for fp in sorted(motion_dir.glob('*.npy')):
        fname = fp.name
        if len(fname.split('___')) < 2:
            continue
        skel = fname.split('___')[0]
        action = parse_action_from_filename(fname)
        if action_to_cluster(action) is None:
            continue
        by_skel_action[(skel, action)].append(fname)
    by_action = defaultdict(set)
    for (skel, action) in by_skel_action:
        by_action[action].add(skel)
    return by_skel_action, by_action


def build(min_src, motion_dir, base_set_path, out_path):
    by_skel_action, by_action = index_cells(motion_dir)
    base = json.loads(Path(base_set_path).read_text())

    lengths = {}

    def src_T(fname):
        if fname not in lengths:
            try:
                lengths[fname] = int(np.load(motion_dir / fname).shape[0])
            except Exception:
                lengths[fname] = 32
        return lengths[fname]

    # Where the smaller set already chose which clips of a source skeleton and action to use,
    # reuse that choice, so every triple built on them asks about the same clips. Otherwise
    # take the first few by name.
    base_cell_sources = {(t['skel_a'], t['action']): list(t['sources']) for t in base['triples']}
    base_keys = {(t['skel_a'], t['skel_b'], t['action']) for t in base['triples']}

    def cell_sources(skel_a, action, clips):
        return base_cell_sources.get((skel_a, action), sorted(clips)[:MAX_SRC])

    # 1) the smaller set, unchanged, keeping its triple and query numbers
    triples = [{'skel_a': t['skel_a'], 'skel_b': t['skel_b'], 'action': t['action'],
                'cluster': t['cluster'], 'sources': list(t['sources'])} for t in base['triples']]
    queries = [dict(q) for q in base['queries']]
    next_qid = max(q['query_id'] for q in base['queries']) + 1

    # 2) every remaining triple: another target skeleton, or a source and action not yet used
    new_keys = []
    for (skel_a, action), clips in sorted(by_skel_action.items()):
        if len(clips) < min_src:
            continue
        for skel_b in sorted(by_action[action] - {skel_a}):
            if (skel_a, skel_b, action) not in base_keys:
                new_keys.append((skel_a, skel_b, action, clips))
    for skel_a, skel_b, action, clips in new_keys:
        tri_idx = len(triples)
        srcs = cell_sources(skel_a, action, clips)
        triples.append({'skel_a': skel_a, 'skel_b': skel_b, 'action': action,
                        'cluster': action_to_cluster(action), 'sources': srcs})
        for sf in srcs:
            queries.append({
                'query_id': next_qid, 'triple_id': tri_idx,
                'skel_a': skel_a, 'skel_b': skel_b, 'cluster': action_to_cluster(action),
                'src_fname': sf, 'src_action': action, 'src_T': src_T(sf), 'split': 'sif',
                'positives_cluster': [{'fname': sf, 'T': src_T(sf),
                                       'action': action, 'cluster': action_to_cluster(action)}],
            })
            next_qid += 1

    n_reused = len(base['queries'])
    manifest = {
        'version': 'truebones_sif_set',
        'description': f'Source-Instance Fidelity benchmark: every (source skeleton, target '
                       f'skeleton, action) triple whose source skeleton has at least {min_src} '
                       f'clips of the action, with no limit on target skeletons or on the number '
                       f'of triples. Query numbers 0-170 are the queries of the 49-triple set, '
                       f'so results for that set stay valid.',
        'seed': SEED, 'min_src_clips': min_src, 'max_src_clips': MAX_SRC,
        'n_triples': len(triples), 'n_queries': len(queries),
        'n_queries_from_49_set': n_reused,
        'n_new_queries': len(queries) - n_reused,
        'triples': triples, 'queries': queries,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(manifest, open(out_path, 'w'), indent=2)
    print(f'wrote {out_path}: {len(triples)} triples / {len(queries)} queries '
          f'({n_reused} reuse existing ids, {len(queries)-n_reused} new)')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', default=str(ROOT / 'benchmark/sets/truebones_1891.json'))
    parser.add_argument('--motions', default=DEFAULT_MOTIONS,
                        help='folder of prepared motion clips')
    parser.add_argument('--base_set', default=str(ROOT / 'benchmark/sets/truebones_49.json'),
                        help='the smaller set, copied in first so its query numbers are kept')
    parser.add_argument('--min_src_clips', type=int, default=3,
                        help='how many clips of an action a source skeleton must have')
    args = parser.parse_args()
    build(args.min_src_clips, Path(args.motions), args.base_set, args.out)


if __name__ == '__main__':
    main()
