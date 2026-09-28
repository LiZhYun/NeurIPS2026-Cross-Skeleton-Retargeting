"""Are the ratios from rotation_test.py far enough from one to mean anything?

rotation_test.py gives, for each query, how much further the decoded motion moves when the
latent code is turned than when it is disturbed by noise of the same size. This script asks
whether those per-query ratios are convincingly above one, and whether one model's ratios
are convincingly above another's. It uses a rank test that makes no assumption about the
shape of the spread, pairing the runs query by query.

It prints its answers and writes nothing.

    python -m latents.rotation_significance
    python -m latents.rotation_significance --dir results/latents/perturbation
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / 'results' / 'latents' / 'perturbation'

SEEDS = ['seed42', 'seed43', 'seed44']
CROSS_MODEL_PAIRS = [
    ('MoReFlow-T', 'ACE-T-seed42'),
    ('MoReFlow-I', 'ACE-I-seed42'),
    ('AL-Flow', 'MoReFlow-T'),
    ('AL-Flow', 'ACE-T-seed42'),
    ('ACE-T-seed42', 'ACE-no-source-latent-seed42'),
    ('ACE-I-seed42', 'ACE-no-source-latent-seed42'),
]


def read_ratios(directory):
    ratios = {}
    for path in sorted(Path(directory).glob('*.json')):
        data = json.loads(path.read_text())
        ratios[data['run']] = np.array([q['rotation_over_noise_ratio']
                                        for q in data['per_query']])
    return ratios


def against_one(name, values):
    """Two questions at once: is the ratio above one, and is it different from one at all?

    The second matters for the models whose ratio sits at one, where the point is that
    turning the code does nothing that noise does not already do.
    """
    higher = wilcoxon(values - 1.0, alternative='greater')
    either_way = wilcoxon(values - 1.0, alternative='two-sided')
    print(f'  {name:32s} median above one {np.median(values) - 1:+.3f}  '
          f'p = {higher.pvalue:.3e} for higher, {either_way.pvalue:.3e} either way')


def compare(a_name, b_name, ratios):
    if a_name not in ratios or b_name not in ratios:
        print(f'  {a_name} against {b_name}: one of them has no file here')
        return
    a, b = ratios[a_name], ratios[b_name]
    if len(a) != len(b):
        print(f'  {a_name} against {b_name}: different numbers of queries, not compared')
        return
    two_sided = wilcoxon(a, b, alternative='two-sided')
    greater = wilcoxon(a, b, alternative='greater')
    print(f'  {a_name:32s} against {b_name:32s} median gap {np.median(a - b):+.3f}  '
          f'p = {two_sided.pvalue:.3e} either way, {greater.pvalue:.3e} for higher')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dir', default=str(DEFAULT_DIR),
                   help='folder of files written by rotation_test.py')
    args = p.parse_args()

    ratios = read_ratios(args.dir)
    if not ratios:
        raise SystemExit(f'No files in {args.dir}. Run python -m latents.rotation_test first.')

    print(f"{'run':32s} {'queries':>8s} {'median':>8s} {'mean':>8s}")
    for name, values in ratios.items():
        print(f'{name:32s} {len(values):8d} {np.median(values):8.3f} {values.mean():8.3f}')

    print('\nIs the ratio above one?')
    for name, values in ratios.items():
        against_one(name, values)

    print('\nDoes the adversarial term make a difference? Same seed on both sides.')
    for seed in SEEDS:
        for family in ['ACE-T', 'ACE-I']:
            compare(f'{family}-{seed}', f'ACE-no-adversarial-loss-{seed}', ratios)

    print('\nOne model against another, query by query:')
    for a_name, b_name in CROSS_MODEL_PAIRS:
        compare(a_name, b_name, ratios)


if __name__ == '__main__':
    main()
