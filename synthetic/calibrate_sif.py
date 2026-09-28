"""Source-Instance Fidelity (SIF) on synthetic data where the true source-to-target map is known.

Uses the dense, paired version of the made-up world (built by synthetic/build_dataset.py).
For every group of source clips (same source skeleton, target skeleton and action; at most
six clips) we score three kinds of output:

  oracle                  the true target motion of each source clip,
  random clip from cell   a random target clip of the same action, ignoring the source,
  random Gaussian noise   noise with the size of a real target clip, ignoring the source.

Groups whose source clips are identical once rotation and scale are removed have nothing to
preserve (SIF is undefined there) and are skipped; their number is reported.

    python -m synthetic.calibrate_sif --cell save/synthetic_2x2/dense_paired \\
        --out results/synthetic/sif_calibration.json
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from sif import distance_matrix, score_groups

DESCRIPTION = (
    'A small made-up world in which the true retargeting map is known, used to check that '
    'the fidelity score behaves. The true map scores near one; a random clip of the same '
    'action and pure noise both sit at zero. Written by synthetic/calibrate_sif.py.'
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', required=True, help='folder of the dense, paired synthetic cell')
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    cell = Path(a.cell)
    meta = json.loads((cell / 'meta.json').read_text())
    pairs = json.loads((cell / 'pairs.json').read_text())
    clips = np.load(cell / 'clips.npy')
    transport = np.load(cell / 'transport.npy', mmap_mode='r')
    joints = meta['joints_per_skel']
    same_cell = defaultdict(list)
    for c in meta['clips']:
        same_cell[(c['skel'], c['action'])].append(c['instance_id'])

    by_group = defaultdict(list)
    for i, p in enumerate(pairs):
        by_group[(p['skel_a'], p['skel_b'], p['action'])].append((p['src_instance_id'], i))

    rng = np.random.default_rng(a.seed)
    groups = {'oracle': [], 'random clip from cell': [], 'random Gaussian noise': []}
    n_candidates = n_skipped = 0
    for (skel_a, skel_b, action), rows in sorted(by_group.items()):
        first = {}
        for src_id, pair_id in rows:
            first.setdefault(src_id, pair_id)
        chosen = list(first.items())[:6]
        if len(chosen) < 3:
            continue
        n_candidates += 1
        sources = [clips[s][:, :joints[skel_a]] for s, _ in chosen]
        d = distance_matrix(sources)
        if d[~np.eye(len(sources), dtype=bool)].std() < 1e-8:
            n_skipped += 1
            continue
        oracle = [np.asarray(transport[p])[:, :joints[skel_b]] for _, p in chosen]
        random_clip = [clips[rng.choice(same_cell[(skel_b, action)])][:, :joints[skel_b]]
                       for _ in chosen]
        noise = [rng.normal(scale=o.std(), size=o.shape) for o in oracle]
        for name, outputs in [('oracle', oracle), ('random clip from cell', random_clip),
                              ('random Gaussian noise', noise)]:
            groups[name].append({'sources': sources, 'outputs': outputs, 'block': skel_a})

    result = {'description': DESCRIPTION,
              'n_groups_with_three_sources': n_candidates,
              'n_groups_left_out_identical_sources': n_skipped,
              'outputs': {}}
    print(f'{n_candidates} groups, {n_skipped} skipped (source clips identical up to rotation and scale)')
    printed_names = {'oracle': 'True map', 'random clip from cell': 'Random clip of the same action',
                     'random Gaussian noise': 'Random noise'}
    for name, g in groups.items():
        r = score_groups(g, length='raw')
        result['outputs'][name] = {'sif': r.sif, 'ci95': list(r.ci), 'p': r.p,
                                   'variation': r.variation, 'n_groups': r.n_groups}
        print(f'{printed_names[name]:31s} SIF {r.sif:+.4f} [{r.ci[0]:+.4f}, {r.ci[1]:+.4f}]  p = {r.p:.4f}  '
              f'variation {r.variation:.3g}  ({r.n_groups} groups)')
    Path(a.out).write_text(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
