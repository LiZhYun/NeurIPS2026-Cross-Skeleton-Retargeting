"""Turning a model's latent code against adding noise of the same size to it.

If the paper's argument is right, two training runs of one model differ by a turn of the
latent space and nothing more. That only matters if turning the code changes the motion that
comes out. A decoder that ignored part of its code would produce the same motion however
that part were turned, and then the turn would be invisible and harmless.

This script measures whether it is. For each query it takes the model's saved latent code,
turns it a random way in all 256 directions, decodes both the turned and the unturned code,
and records how far the motion moved. It then does the same with random noise added to the
code instead of a turn, scaled so that the code is disturbed by the same amount. The ratio
of the two says whether turning the code does anything that noise does not already do: a
ratio near one means it does not.

    python -m latents.rotation_test --run ACE-T-seed42 --codes outputs/latent_codes/ace_t_seed42/set37
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from core.truebones.param_utils import OBJECT_SUBSETS_DICT
from methods.tokenizer.registry import TokenizerRegistry
from latents.read_codes import read_run

ROOT = Path(__file__).resolve().parents[1]
SETS = ROOT / 'benchmark' / 'sets'

DESCRIPTION = (
    'A decoder that ignores part of its latent code would produce the same motion whichever '
    'way that part is turned. This file turns the latent code of each query and, separately, '
    'adds noise of the same size, then records how far the decoded motion moves in each case. '
    'A ratio near one means the turn does nothing that the noise does not already do. Over '
    'the {n_queries} queries of the {n_triples}-triple evaluation, with {n_rotations} turns '
    'and {n_noise} noise draws per query. "rel_div_rotation" and "rel_div_noise" are how far '
    'the decoded motion moves, relative to its own size, under a turn and under noise; '
    '"rotation_over_noise_ratio" divides the first by the second for one query, and '
    '"ratio_mean" averages that over the queries.'
)


def random_turn(d, rng):
    """A random way of turning a d-dimensional space, mirror images included."""
    A = rng.randn(d, d).astype(np.float32)
    Q, R = np.linalg.qr(A)
    Q = Q * np.sign(np.diag(R))[None, :]
    return Q


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--codes', required=True, help='folder of latent codes for one model')
    p.add_argument('--run', required=True, help='a name for that model, as the paper writes it')
    p.add_argument('--n_rotations', type=int, default=20, help='turns per query')
    p.add_argument('--n_noise', type=int, default=20, help='noise draws per query')
    p.add_argument('--set', choices=['37', '49', '1891'], default='37')
    p.add_argument('--out', default=None,
                   help='where to write the result; the default names the file after the run')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max_queries', type=int, default=None)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print("Loading the tokenizers for all 70 skeletons...")
    registry = TokenizerRegistry(list(OBJECT_SUBSETS_DICT['all']), device=device)

    index = read_run(args.codes)
    if args.max_queries:
        index = index[:args.max_queries]
    print(f"{len(index)} queries from {args.codes}")

    rng = np.random.RandomState(args.seed)
    per_query = []
    t0 = time.time()
    with torch.no_grad():
        for entry in index:
            qid = entry['query_id']
            tgt_skel = entry['target_skeleton']
            try:
                z = entry['code']
                if z.ndim != 2:
                    print(f"  query {qid}: skipped, the code is not a table of vectors")
                    continue
                T, d = z.shape
                z_t = torch.from_numpy(z).to(device).unsqueeze(0)

                base_norm = registry.decode_tokens(tgt_skel, z_t)
                base = registry.unnormalize(tgt_skel, base_norm)
                base_size = float(np.sum(base.squeeze(0).cpu().numpy() ** 2))

                turned = []
                for _ in range(args.n_rotations):
                    R = random_turn(d, rng)
                    z_rot = torch.from_numpy((z @ R).astype(np.float32)).to(device).unsqueeze(0)
                    decoded = registry.unnormalize(
                        tgt_skel, registry.decode_tokens(tgt_skel, z_rot))
                    moved = float(torch.sum((decoded - base) ** 2).item())
                    turned.append(moved / max(base_size, 1e-12))

                # How far a turn moves the code itself, so the noise can be matched to it
                reference = z @ random_turn(d, rng) - z
                step = float(np.linalg.norm(reference) / max(np.linalg.norm(z), 1e-12))

                noised = []
                for _ in range(args.n_noise):
                    eps = rng.randn(T, d).astype(np.float32)
                    eps = eps * (step * np.linalg.norm(z) / max(np.linalg.norm(eps), 1e-12))
                    z_eps = torch.from_numpy((z + eps).astype(np.float32)).to(device).unsqueeze(0)
                    decoded = registry.unnormalize(
                        tgt_skel, registry.decode_tokens(tgt_skel, z_eps))
                    moved = float(torch.sum((decoded - base) ** 2).item())
                    noised.append(moved / max(base_size, 1e-12))

                turned_mean = float(np.mean(turned))
                noised_mean = float(np.mean(noised))
                per_query.append({
                    'query_id': qid,
                    'target_skeleton': tgt_skel,
                    'latent_dim': d,
                    'frames': T,
                    'rel_div_rotation_mean': turned_mean,
                    'rel_div_rotation_std': float(np.std(turned)),
                    'rel_div_noise_mean': noised_mean,
                    'rel_div_noise_std': float(np.std(noised)),
                    'rotation_over_noise_ratio': turned_mean / max(noised_mean, 1e-12),
                    'perturbation_size': step,
                })
            except Exception as e:
                print(f"  query {qid}: failed, {type(e).__name__}: {e}")
            if per_query and len(per_query) % 25 == 0:
                print(f"  [{len(per_query)}/{len(index)}] {time.time() - t0:.0f}s")

    if per_query:
        turned = np.array([r['rel_div_rotation_mean'] for r in per_query])
        noised = np.array([r['rel_div_noise_mean'] for r in per_query])
        ratios = np.array([r['rotation_over_noise_ratio'] for r in per_query])
        aggregate = {
            'n_queries': len(per_query),
            'rel_div_rotation_mean': float(turned.mean()),
            'rel_div_rotation_std': float(turned.std()),
            'rel_div_noise_mean': float(noised.mean()),
            'rel_div_noise_std': float(noised.std()),
            'ratio_mean': float(ratios.mean()),
            'ratio_std': float(ratios.std()),
            'ratio_median': float(np.median(ratios)),
            'ratio_min': float(ratios.min()),
            'ratio_max': float(ratios.max()),
            'verdict': ('turning the code changes the motion more than noise does'
                        if ratios.mean() > 1.5 else
                        'turning the code changes the motion about as much as noise does'),
        }
    else:
        aggregate = {'n_queries': 0}

    with open(SETS / f'truebones_{args.set}.json') as f:
        benchmark_set = json.load(f)

    out = {
        'description': DESCRIPTION.format(
            n_queries=len(benchmark_set['queries']), n_triples=benchmark_set['n_triples'],
            n_rotations=args.n_rotations, n_noise=args.n_noise),
        'run': args.run,
        'n_rotations_per_query': args.n_rotations,
        'n_noise_draws_per_query': args.n_noise,
        'aggregate': aggregate,
        'per_query': per_query,
    }
    out_path = (Path(args.out) if args.out
                else ROOT / 'results' / 'latents' / 'perturbation' / f'{args.run}.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1, default=float))
    print(f"\nsaved {out_path}")
    for k, v in aggregate.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == '__main__':
    main()
