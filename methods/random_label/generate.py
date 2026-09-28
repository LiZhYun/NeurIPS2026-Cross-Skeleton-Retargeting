"""Two reference points that answer a query from the action name alone.

Both return a clip the target skeleton already has, drawn at random. They differ in what they
draw from:

  random-same-cluster       clips doing the broad action group the classifier thinks the
                            source clip belongs to
  random-same-exact-action  clips carrying the same action name as the source clip

Neither looks at how the source clip actually moves. Whatever they score is therefore what can
be had from knowing the name of the action and nothing else, which is the level any real
method has to beat. Answers are cut to the typical length of the clips they will be compared
against, as with the other methods that answer by retrieving.

    python -m methods.random_label.generate --baseline random_same_cluster \
        --set 49 --out_dir outputs/random_same_cluster/set49
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np

from benchmark.action_classifier import train_classifier, feature_vector, CLUSTER_TO_IDX
from benchmark.clip_features import open_clip_features
from core.truebones.param_utils import OBJECT_SUBSETS_DICT, DATASET_DIR
from methods.common.queries import reference_length

ROOT = Path(__file__).resolve().parents[2]
MOTION_DIR = Path(DATASET_DIR) / 'motions'
CLIP_INDEX_PATH = ROOT / 'benchmark/clip_index.json'
SETS = ROOT / 'benchmark/sets'
QUERIES_ROOT = ROOT / 'benchmark/queries'


def paper_runs(args):
    """Split the queries into the runs that made the paper's answers.

    The random pick of each query depends on every pick drawn before it in the same run, and
    the paper made its answers in several runs, each starting the random generator afresh from
    the same seed. Following the same runs gives the same picks: fold 43 was answered straight
    after fold 42 in one run, the 37-triple set is part of the 49-triple set, and the 1,891-triple
    set was answered in three runs starting at queries 0, 171 and 176. Each run is a list of
    (query, keep) pairs; queries with keep False are drawn but not saved.
    """
    if args.fold:
        queries_path = QUERIES_ROOT / f'fold_{args.fold}/manifest.json'
        fold_42 = json.load(open(QUERIES_ROOT / 'fold_42/manifest.json'))['queries']
        if args.fold == 42:
            return queries_path, [[(q, True) for q in fold_42]]
        fold_43 = json.load(open(queries_path))['queries']
        return queries_path, [[(q, False) for q in fold_42] + [(q, True) for q in fold_43]]
    queries_path = SETS / f'truebones_{args.set}.json'
    wanted = json.load(open(queries_path))['queries']
    if args.set == '37':
        ids = {q['query_id'] for q in wanted}
        set_49 = json.load(open(SETS / 'truebones_49.json'))['queries']
        return queries_path, [[(q, q['query_id'] in ids) for q in set_49]]
    starts = [0, 171, 176, len(wanted)] if args.set == '1891' else [0, len(wanted)]
    return queries_path, [[(q, True) for q in wanted[a:b]] for a, b in zip(starts, starts[1:])]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--baseline', choices=['random_same_cluster', 'random_same_exact_action'],
                        required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--set', choices=['37', '49', '1891'],
                       help='which evaluation set to answer, by its number of triples')
    group.add_argument('--fold', type=int, choices=[42, 43],
                       help='answer one of the two query sets in benchmark/queries instead')
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print("Loading the clip features, the classifier and the clip index...")
    qc = open_clip_features()
    fname_to_qc_idx = {m['fname']: i for i, m in enumerate(qc['meta'])}
    train_skels = set(OBJECT_SUBSETS_DICT['train'])
    clf = train_classifier(qc, train_skels)
    idx_to_cluster = {v: k for k, v in CLUSTER_TO_IDX.items()}
    clip_idx = json.load(open(CLIP_INDEX_PATH))['index']
    print(f"  {len(clip_idx)} skeletons indexed")

    def candidates_for(q):
        """The clips a query may be answered with, or None when it cannot be answered."""
        if args.baseline == 'random_same_cluster':
            if q.get('src_fname', '') not in fname_to_qc_idx:
                return None
            qi = fname_to_qc_idx[q['src_fname']]
            feat = feature_vector(qc['com_path'][qi], qc['heading_vel'][qi],
                                  qc['contact_sched'][qi], qc['cadence'][qi],
                                  qc['limb_usage'][qi])
            pred_cluster = idx_to_cluster.get(int(clf.predict(feat.reshape(1, -1))[0]), '')
            return clip_idx.get(q['skel_b'], {}).get(pred_cluster, [])
        # Every clip of the target skeleton carrying the source's action name
        return [c for clips in clip_idx.get(q['skel_b'], {}).values() for c in clips
                if c.get('action') == q.get('src_action', '')]

    queries_path, runs = paper_runs(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_query = []
    n_ok = n_fail = 0
    t0 = time.time()
    for run in runs:
        rng = np.random.RandomState(args.seed)
        for q, keep in run:
            candidates = candidates_for(q)
            if not keep:
                # A query the paper answered earlier in the same run: draw its pick, save nothing
                if candidates:
                    rng.choice(len(candidates))
                continue

            qid = q['query_id']
            rec = {'query_id': qid, 'skel_a': q.get('skel_a'), 'skel_b': q['skel_b'],
                   'src_action': q.get('src_action', ''), 'cluster': q.get('cluster'),
                   'split': q.get('split'), 'status': 'pending'}
            if candidates is None:
                rec['status'] = 'skipped_no_clip_features'
            elif not candidates:
                rec['status'] = 'skipped_no_candidates'
            else:
                try:
                    pick = candidates[rng.choice(len(candidates))]
                    motion = np.load(MOTION_DIR / pick['fname']).astype(np.float32)
                    # Cut to the typical length of the clips this answer will be compared against
                    T_tgt = reference_length(q, motion.shape[0])
                    T_out = min(T_tgt, motion.shape[0])
                    motion = motion[:T_out]

                    np.save(out_dir / f'query_{qid:04d}.npy', motion)
                    rec.update({'status': 'ok', 'picked_fname': pick['fname'],
                                'picked_action': pick.get('action'),
                                'picked_cluster': pick.get('cluster')})
                    n_ok += 1
                except Exception as e:
                    rec['status'] = f'error: {e}'
                    n_fail += 1
            per_query.append(rec)

    summary = {
        'method': args.baseline, 'queries': str(queries_path),
        'n_queries': len(per_query),
        'n_ok': n_ok, 'n_failed': n_fail,
        'wall_clock_s': time.time() - t0,
        'per_query': per_query,
    }
    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"ok={n_ok}, fail={n_fail}, wall_clock={time.time()-t0:.0f}s")
    print(f"  saved: {out_dir}/metrics.json")


if __name__ == '__main__':
    main()
