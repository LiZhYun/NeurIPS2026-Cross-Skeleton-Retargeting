"""How many directions of its latent space each model actually uses.

A model's latent code has 256 numbers in it, but that does not mean it uses 256 independent
directions. Effective rank counts the directions that carry real variation, and spectral
flatness is near zero when a handful of directions carry nearly everything.

The number matters because a model that has squeezed its codes into two or three directions
has almost no room left to say anything about the particular source clip it was shown. Read
next to the fidelity scores, it separates models that never encoded the source from models
that encoded it and lost it later.

Every code of every query is stacked into one table per model, and the two numbers are
computed from the spread of that table.

    python -m latents.effective_rank \\
        --run ACE-T-seed42 outputs/latent_codes/ace_t_seed42/set37 \\
        --run MoReFlow-T   outputs/latent_codes/moreflow_t/set37 \\
        --out results/latents/effective_rank.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from latents.cross_seed_alignment import spread_stats
from latents.read_codes import pooled_matrix

ROOT = Path(__file__).resolve().parents[1]
SETS = ROOT / 'benchmark' / 'sets'

DESCRIPTION = (
    'How many directions a model actually uses inside its {width}-dimensional latent space. '
    'Effective rank is the number of directions that carry real variation; spectral flatness '
    'is near zero when a few directions dominate. Computed over the {n_queries} queries of '
    'the {n_triples}-triple evaluation. The models that compress hardest have the least room '
    'left to describe a particular source clip.'
)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', nargs=2, action='append', metavar=('LABEL', 'DIR'), required=True,
                   help='a name for one trained model and the folder of its latent codes; '
                        'give this once per model')
    p.add_argument('--set', choices=['37', '49', '1891'], default='37')
    p.add_argument('--out', required=True)
    args = p.parse_args()

    rows = []
    for label, run_dir in args.run:
        Z = pooled_matrix(run_dir)
        stats = spread_stats(Z)
        rows.append({
            'model': label,
            'shape': list(Z.shape),
            'spectral_flatness': stats['spectral_flatness'],
            'effective_rank': stats['effective_rank'],
        })
        print(f"  {label:32s} {Z.shape[0]:6d} x {Z.shape[1]:<4d} "
              f"flatness={stats['spectral_flatness']:.4f}  "
              f"directions in use={stats['effective_rank']:.2f}")

    rows.sort(key=lambda r: r['effective_rank'])

    with open(SETS / f'truebones_{args.set}.json') as f:
        benchmark_set = json.load(f)

    out = {
        'description': DESCRIPTION.format(
            width=rows[0]['shape'][1], n_queries=len(benchmark_set['queries']),
            n_triples=benchmark_set['n_triples']),
        'rows': rows,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1, default=float))
    print(f"\nsaved {out_path}")


if __name__ == '__main__':
    main()
