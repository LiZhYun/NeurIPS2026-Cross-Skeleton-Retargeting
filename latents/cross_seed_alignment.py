"""Two training runs of one model end up with latent codes that are turns of one another.

If a model's internal code were fixed by the data, two runs that differ only in their random
seed would arrive at the same code. They do not. What they arrive at is close to the same
code seen from a different angle: a single rotation, one per target skeleton, brings one
run's codes most of the way onto the other's.

This is why the code itself carries no meaning that survives retraining, and it is the
reason the paper treats the latent space as a free choice rather than as something the data
pins down.

For every target skeleton the script stacks each run's codes in query order, then for each
pair of runs finds the one rotation that brings them closest together and reports how much
of the gap it closes and how well the codes line up afterwards. It also reports how spread
out each run's codes are on their own.

    python -m latents.cross_seed_alignment --name ACE-T \\
        --run ACE-T-seed42 outputs/latent_codes/ace_t_seed42/set37 \\
        --run ACE-T-seed43 outputs/latent_codes/ace_t_seed43/set37 \\
        --run ACE-T-seed44 outputs/latent_codes/ace_t_seed44/set37 \\
        --seeds 42 43 44 --out results/latents/cross_seed_alignment_ace_t.json
"""
from __future__ import annotations
import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from latents.read_codes import group_by_target_skeleton

ROOT = Path(__file__).resolve().parents[1]
SETS = ROOT / 'benchmark' / 'sets'

DESCRIPTION = (
    'Two training runs of {name} that differ only in their random seed end up with latent '
    'codes that are close to a rotation of one another. For each pair of seeds this file '
    'gives how much of the difference a single rotation explains, and how well the codes '
    'line up once that rotation is applied. Computed per target skeleton over the '
    '{n_queries} queries of the {n_triples}-triple evaluation, then averaged over '
    '{n_skeletons} skeletons.'
)


def spread_stats(Z):
    """How spread out a set of latent vectors is. Z is [n_vectors, width].

    Spectral flatness compares the typical size of the directions of variation against their
    average size, and is near zero when a handful of directions carry everything. Effective
    rank turns the same spectrum into a count of the directions actually in use.
    """
    Z_centered = Z - Z.mean(axis=0, keepdims=True)
    N, d = Z_centered.shape
    cov = Z_centered.T @ Z_centered / max(N - 1, 1)
    eigs = np.linalg.eigvalsh(cov)
    eigs = np.maximum(eigs, 0.0)                 # tiny negatives are rounding, not variation
    eigs = np.sort(eigs)[::-1]
    eigs_nz = eigs[eigs > 1e-12]
    if len(eigs_nz) == 0:
        return {'eig_max': 0.0, 'eig_min': 0.0, 'spectral_flatness': float('nan'),
                'effective_rank': 0.0, 'latent_dim': int(d), 'n_latent_vectors': int(N)}
    geomean = float(np.exp(np.mean(np.log(eigs_nz))))
    arithmean = float(eigs_nz.mean())
    flatness = geomean / arithmean if arithmean > 0 else float('nan')
    p = eigs_nz / eigs_nz.sum()
    eff_rank = float(np.exp(-np.sum(p * np.log(p + 1e-30))))
    return {
        'eig_max': float(eigs_nz[0]),
        'eig_min': float(eigs_nz[-1]),
        'top_five_eigenvalue_fraction': float(eigs_nz[:min(5, len(eigs_nz))].sum()
                                              / eigs_nz.sum()),
        'spectral_flatness': float(flatness),
        'effective_rank': eff_rank,
        'latent_dim': int(d),
        'n_latent_vectors': int(N),
        'top_five_eigenvalues': [float(x) for x in eigs_nz[:5]],
    }


def best_rotation(Z_a, Z_b):
    """Find the one rotation that brings Z_a closest to Z_b, and say how much it helps.

    Reflections are allowed as well as rotations, because a mirror image of a latent space
    is just as good a choice of axes as a turn of it.
    """
    assert Z_a.shape == Z_b.shape
    U, S, Vt = np.linalg.svd(Z_a.T @ Z_b)
    R = U @ Vt
    Z_a_rot = Z_a @ R
    res_aligned = float(np.sum((Z_b - Z_a_rot) ** 2))
    res_unaligned = float(np.sum((Z_b - Z_a) ** 2))
    res_total = float(np.sum(Z_b ** 2))
    frac_explained = 1.0 - (res_aligned / max(res_unaligned, 1e-12))
    Z_b_norms = np.linalg.norm(Z_b, axis=1, keepdims=True) + 1e-12
    Z_a_rot_norms = np.linalg.norm(Z_a_rot, axis=1, keepdims=True) + 1e-12
    cosines = (Z_b * Z_a_rot).sum(axis=1) / (Z_b_norms[:, 0] * Z_a_rot_norms[:, 0])
    return {
        'residual_aligned': res_aligned,
        'residual_unaligned': res_unaligned,
        'residual_total_b': res_total,
        'rel_aligned_b': res_aligned / max(res_total, 1e-12),
        'rel_unaligned_b': res_unaligned / max(res_total, 1e-12),
        'fraction_aligned_explained': frac_explained,
        'mean_row_cosine_aligned': float(np.mean(cosines)),
        'top_five_singular_values': [float(x) for x in S[:5]],
        'singular_value_sum': float(S.sum()),
        'rotation_trace_over_d': float(np.trace(R) / R.shape[0]),
        'latent_dim': int(Z_a.shape[1]),
        'n_latent_vectors': int(Z_a.shape[0]),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', nargs=2, action='append', metavar=('LABEL', 'DIR'), required=True,
                   help='a name for one training run and the folder of its latent codes; '
                        'give this once per run')
    p.add_argument('--seeds', type=int, nargs='+', required=True,
                   help='the random seed of each run, in the same order')
    p.add_argument('--name', required=True, help='the model name, as the paper writes it')
    p.add_argument('--set', choices=['37', '49', '1891'], default='37')
    p.add_argument('--out', required=True)
    p.add_argument('--min_vectors_per_skeleton', type=int, default=20,
                   help='skip target skeletons with fewer latent vectors than this')
    args = p.parse_args()

    labels = [label for label, _ in args.run]
    if len(args.seeds) != len(labels):
        p.error('--seeds must give one seed per --run, in the same order')

    print(f"Reading {len(labels)} runs")
    runs = {}
    for label, run_dir in args.run:
        print(f"  {label}: {run_dir}")
        runs[label] = group_by_target_skeleton(run_dir, args.min_vectors_per_skeleton)

    shared = set.intersection(*[set(r.keys()) for r in runs.values()])
    print(f"Target skeletons present in every run: {len(shared)}")

    per_skeleton = {}
    for skel in sorted(shared):
        stacked = {label: runs[label][skel][0] for label in labels}
        query_ids = {label: runs[label][skel][1] for label in labels}
        reference = query_ids[labels[0]]
        n_vectors = stacked[labels[0]].shape[0]
        if not all(query_ids[l] == reference and stacked[l].shape[0] == n_vectors
                   for l in labels):
            print(f"  skipping {skel}: the runs cover different queries")
            continue

        per_skeleton[skel] = {
            'latent_dim': stacked[labels[0]].shape[1],
            'n_latent_vectors': n_vectors,
            'spread_of_the_latent_codes': {l: spread_stats(stacked[l]) for l in labels},
            'seed_pair_alignment': {f'{a}_vs_{b}': best_rotation(stacked[a], stacked[b])
                                    for a, b in combinations(labels, 2)},
        }

    spread = {}
    for label in labels:
        flat = np.array([r['spread_of_the_latent_codes'][label]['spectral_flatness']
                         for r in per_skeleton.values()])
        rank = np.array([r['spread_of_the_latent_codes'][label]['effective_rank']
                         for r in per_skeleton.values()])
        flat = flat[np.isfinite(flat)]
        rank = rank[np.isfinite(rank)]
        spread[label] = {
            'spectral_flatness_mean': float(flat.mean()) if len(flat) else float('nan'),
            'spectral_flatness_std': float(flat.std()) if len(flat) > 1 else float('nan'),
            'effective_rank_mean': float(rank.mean()) if len(rank) else float('nan'),
            'effective_rank_std': float(rank.std()) if len(rank) > 1 else float('nan'),
            'latent_dim': (next(iter(per_skeleton.values()))['latent_dim']
                           if per_skeleton else None),
            'n_skeletons': int(len(flat)),
        }

    seed_of = dict(zip(labels, args.seeds))
    alignment = {}
    for a, b in combinations(labels, 2):
        key = f'{a}_vs_{b}'
        after = np.array([r['seed_pair_alignment'][key]['rel_aligned_b']
                          for r in per_skeleton.values()])
        before = np.array([r['seed_pair_alignment'][key]['rel_unaligned_b']
                           for r in per_skeleton.values()])
        cosine = np.array([r['seed_pair_alignment'][key]['mean_row_cosine_aligned']
                           for r in per_skeleton.values()])
        after = after[np.isfinite(after)]
        before = before[np.isfinite(before)]
        cosine = cosine[np.isfinite(cosine)]
        closed = (before - after) / np.maximum(before, 1e-12)
        alignment[f'seed{seed_of[a]}_vs_seed{seed_of[b]}'] = {
            'distance_after_rotation': float(after.mean()) if len(after) else float('nan'),
            'distance_before_rotation': float(before.mean()) if len(before) else float('nan'),
            'fraction_explained_by_rotation': (float(closed.mean()) if len(closed)
                                               else float('nan')),
            'row_cosine_after_rotation': float(cosine.mean()) if len(cosine) else float('nan'),
            'n_skeletons': int(len(after)),
        }

    with open(SETS / f'truebones_{args.set}.json') as f:
        benchmark_set = json.load(f)

    out = {
        'description': DESCRIPTION.format(
            name=args.name, n_queries=len(benchmark_set['queries']),
            n_triples=benchmark_set['n_triples'], n_skeletons=len(per_skeleton)),
        'seeds': list(args.seeds),
        'seed_pair_alignment': alignment,
        'spread_of_the_latent_codes': spread,
        'per_skeleton': per_skeleton,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1, default=float))
    print(f"\nsaved {out_path}")

    print(f"\nHow spread out each run's codes are, averaged over {len(per_skeleton)} skeletons:")
    for label, s in spread.items():
        print(f"  {label}: flatness={s['spectral_flatness_mean']:.5f}, "
              f"directions in use={s['effective_rank_mean']:.1f}")
    print("\nHow much of the difference between two runs one rotation explains:")
    for key, s in alignment.items():
        print(f"  {key}: {s['fraction_explained_by_rotation']:.3f} explained, "
              f"codes line up at cosine {s['row_cosine_after_rotation']:.3f}")


if __name__ == '__main__':
    main()
