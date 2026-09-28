"""The fidelity score, computed on the latent codes instead of on the motion.

The fidelity score asks whether two different source clips lead to two correspondingly
different answers. Run on the motions a model writes, a low score can mean either of two
things: the model never took note of which source clip it was shown, or it took note and
then lost it on the way out. Running the same score on the latent codes tells the two apart.

  high on the codes and low on the motion   the source survived the encoder and was lost
                                            in the decoder
  low on both                               the source was never taken note of
  high on both                              the source is carried all the way through

For every group of queries that share a source skeleton, a target skeleton and an action,
this compares how far apart the source clips are with how far apart the model's latent codes
are, and reports the correlation. Codes of different lengths are averaged over time first,
so that clips of different lengths can be compared.

The answer goes into results/latents/latent_sif.json, beside the fidelity score of the same
run's motions. That file holds both scores for every trained run. --run fills in the latent
score from a folder of latent codes. --motions fills in the motion score from the folder of
motions the same run wrote (query_XXXX.npy), using the benchmark scorer's own groups
(benchmark/score.py) and comparing clips over the frames they share. The 37-triple set keeps
the query numbers of the 49-triple and 1891-triple sets, so a folder written for either of
those serves as well.
Everything else in the file is left as it was found.

    python -m latents.latent_sif \\
        --run ACE-T-seed42 outputs/latent_codes/ace_t_seed42/set37 \\
        --run MoReFlow-T   outputs/latent_codes/moreflow_t/set37 \\
        --motions ACE-T-seed42 outputs/ace_t/set49 \\
        --motions MoReFlow-T   outputs/moreflow_t/set49
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from benchmark.score import load_groups
from core.truebones.param_utils import DATASET_DIR
from latents.read_codes import read_run
from sif import score_groups

ROOT = Path(__file__).resolve().parents[1]
SETS = ROOT / 'benchmark' / 'sets'
MOTION_DIR = ROOT / DATASET_DIR / 'motions'
DEFAULT_OUT = ROOT / 'results' / 'latents' / 'latent_sif.json'

DESCRIPTION = (
    'The fidelity score computed on a model\'s latent codes rather than on the motions it '
    'writes. A high number here next to a low one on the motions means the source clip '
    'survived the encoder and was lost in the decoder; low on both means it was never taken '
    'note of. Over the groups of the {n_triples}-triple evaluation that hold at least two '
    'different source clips.'
)


def clip_distance(a, b):
    """How far apart two clips of one skeleton are, once position and turn are matched.

    Only the joint positions are used. Both clips are centred, scaled to the same size and
    then turned to line up as well as they can, so what is left is the difference in the
    movement itself.
    """
    T = min(a.shape[0], b.shape[0])
    if T == 0:
        return np.nan
    a_flat = a[:T, :, :3].reshape(T, -1)
    b_flat = b[:T, :, :3].reshape(T, -1)
    a_c = a_flat - a_flat.mean(0, keepdims=True)
    b_c = b_flat - b_flat.mean(0, keepdims=True)
    a_n = a_c / (np.linalg.norm(a_c) + 1e-8)
    b_n = b_c / (np.linalg.norm(b_c) + 1e-8)
    U, _, Vt = np.linalg.svd(a_n.T @ b_n, full_matrices=False)
    diff = (a_n @ (U @ Vt)) - b_n
    return float(np.sqrt((diff * diff).sum()))


def average_over_time(z):
    """Collapse the time axis so that codes of different lengths can be compared.

    A code saved as one vector per moment becomes a single vector; AnyTop's code, which has a
    vector per joint and moment, becomes one vector per joint.
    """
    if z is None:
        return None
    if z.ndim == 2:
        return z.mean(axis=0)
    if z.ndim == 3:
        return z.mean(axis=1)
    return z.flatten()


def code_distance(z_a, z_b):
    """How far apart two latent codes are, as a fraction of their own size."""
    if z_a is None or z_b is None:
        return np.nan
    pa, pb = average_over_time(z_a), average_over_time(z_b)
    if pa.shape != pb.shape:
        return np.nan
    fa, fb = pa.flatten(), pb.flatten()
    norm = np.linalg.norm(fa) + np.linalg.norm(fb) + 1e-8
    return float(np.linalg.norm(fa - fb) / norm)


def bootstrap_ci(values, n_boot=10000, seed=42):
    """A 95% range for the average, by resampling the groups."""
    if not values:
        return [float('nan'), float('nan'), float('nan')]
    rng = np.random.RandomState(seed)
    arr = np.array(values)
    boots = [arr[rng.randint(0, len(arr), size=len(arr))].mean() for _ in range(n_boot)]
    return [float(np.percentile(boots, 2.5)), float(np.mean(arr)), float(np.percentile(boots, 97.5))]


def score(codes_by_query, queries, min_clips=2, max_clips=8):
    by_group = defaultdict(list)
    for q in queries:
        sa, sb, act = q.get('skel_a'), q.get('skel_b'), q.get('src_action')
        if sa and sb and act:
            by_group[(sa, sb, act)].append(q)

    groups, correlations = [], []
    for (sa, sb, act), members in sorted(by_group.items()):
        first_of_clip = {}
        for q in members:
            name = q.get('src_fname', '')
            if name and name not in first_of_clip:
                first_of_clip[name] = q
        chosen = list(first_of_clip.values())
        if len(chosen) < min_clips:
            continue
        chosen = chosen[:max_clips]

        clips, codes = [], []
        for q in chosen:
            path = MOTION_DIR / q['src_fname']
            clips.append(np.load(path).astype(np.float32) if path.exists() else None)
            codes.append(codes_by_query.get(q.get('query_id')))

        valid = [(c, z) for c, z in zip(clips, codes) if c is not None and z is not None]
        if len(valid) < min_clips:
            continue

        n = len(valid)
        clip_d = np.full((n, n), np.nan)
        code_d = np.full((n, n), np.nan)
        for i in range(n):
            for j in range(i + 1, n):
                clip_d[i, j] = clip_d[j, i] = clip_distance(valid[i][0], valid[j][0])
                code_d[i, j] = code_d[j, i] = code_distance(valid[i][1], valid[j][1])

        usable = ~np.eye(n, dtype=bool) & ~np.isnan(clip_d) & ~np.isnan(code_d)
        if usable.sum() < 2:
            continue
        clip_vec, code_vec = clip_d[usable], code_d[usable]
        if clip_vec.std() < 1e-8 or code_vec.std() < 1e-8:
            rho = 0.0
        else:
            rho = float(np.corrcoef(clip_vec, code_vec)[0, 1])
        if np.isnan(rho):
            continue
        groups.append({
            'skel_a': sa, 'skel_b': sb, 'action': act,
            'n_source_clips': n, 'correlation': rho,
            'mean_clip_distance': float(clip_vec.mean()),
            'mean_code_distance': float(code_vec.mean()),
            'code_over_clip_spread': float(code_vec.mean() / max(clip_vec.mean(), 1e-8)),
        })
        correlations.append(rho)

    return groups, correlations


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', nargs=2, action='append', metavar=('LABEL', 'DIR'), default=[],
                   help='a name for one trained run and the folder of its latent codes; '
                        'give this once per run')
    p.add_argument('--motions', nargs=2, action='append', metavar=('LABEL', 'DIR'), default=[],
                   help='a name for one trained run and the folder of motions it wrote for '
                        'the benchmark set; give this once per run')
    p.add_argument('--set', choices=['37', '49', '1891'], default='37')
    p.add_argument('--out', default=str(DEFAULT_OUT),
                   help='the file that holds both scores for every run')
    p.add_argument('--per_group_dir', default=None,
                   help='also write the per-group detail of each run into this folder')
    p.add_argument('--min_source_clips', type=int, default=2)
    p.add_argument('--max_source_clips', type=int, default=8)
    args = p.parse_args()
    if not args.run and not args.motions:
        p.error('give at least one --run or --motions')

    with open(SETS / f'truebones_{args.set}.json') as f:
        benchmark_set = json.load(f)
    queries = benchmark_set['queries']

    out_path = Path(args.out)
    combined = (json.loads(out_path.read_text()) if out_path.exists()
                else {'description': DESCRIPTION.format(n_triples=benchmark_set['n_triples']),
                      'runs': {}, 'rows': []})

    for label, motions_dir in args.motions:
        groups = load_groups(args.set, motions_dir, MOTION_DIR)
        result = score_groups(groups, 'raw', min_clips=2)
        run = combined['runs'].setdefault(label, {})
        run['n_triples'] = result.n_groups
        run['sif'] = result.sif
        run['variation'] = result.variation
        print(f"{label}: {result.n_groups} groups, fidelity {result.sif:.4f}, "
              f"variation {result.variation:.4f}")

    for label, codes_dir in args.run:
        codes_by_query = {e['query_id']: e['code'] for e in read_run(codes_dir)}
        print(f"{label}: {len(codes_by_query)} latent codes from {codes_dir}")
        groups, correlations = score(codes_by_query, queries,
                                     args.min_source_clips, args.max_source_clips)
        if not correlations:
            print(f"  no group had two usable source clips; {label} left unchanged")
            continue
        low, _, high = bootstrap_ci(correlations)
        run = combined['runs'].setdefault(label, {})
        if 'sif' not in run:
            print(f"  note: {label} has no motion score in this file yet; "
                  f"that one comes from benchmark/score.py")
        run['n_triples_with_latents'] = len(correlations)
        run['latent_sif'] = float(np.mean(correlations))
        run['latent_sif_ci95'] = [low, high]
        print(f"  {len(correlations)} groups, latent fidelity {run['latent_sif']:.4f} "
              f"[{low:.3f}, {high:.3f}]")
        if args.per_group_dir:
            detail = Path(args.per_group_dir) / f'{label}.json'
            detail.parent.mkdir(parents=True, exist_ok=True)
            detail.write_text(json.dumps({'run': label, 'per_group': groups},
                                         indent=1, default=float))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(combined, indent=1, default=float))
    print(f"\nsaved {out_path}")


if __name__ == '__main__':
    main()
