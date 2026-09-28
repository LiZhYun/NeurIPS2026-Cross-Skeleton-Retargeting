"""ANCHOR: answer every query by handing back a clip the target skeleton already has.

ANCHOR is the strongest of the reference points in the paper that use no motion generation at
all. It trains nothing but the action classifier, and never makes a new motion: for each query
it returns an existing clip of the target skeleton, unchanged, chosen as
methods/anchor/retrieval.py describes. Its answers have the same form as any generator's, so
the same scoring code applies to both. What it scores is therefore what can be reached without
generating anything.

    python -m methods.anchor.generate --set 49 --out_dir outputs/anchor/set49
    python -m methods.anchor.generate --fold 42 --out_dir outputs/anchor/fold_42
    python -m methods.anchor.generate --queries <manifest.json> --out_dir outputs/anchor/full

The last form answers a query list written by benchmark.build_queries_full. Its files are
named with five digits, query_00000.npy and so on, as benchmark.action_auc_full expects.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np

from benchmark.action_classifier import train_classifier
from benchmark.clip_features import open_clip_features
from core.truebones.param_utils import OBJECT_SUBSETS_DICT, DATASET_DIR
from methods.anchor.retrieval import build_signature_table, rerank_one_query

ROOT = Path(__file__).resolve().parents[2]
MOTION_DIR = Path(DATASET_DIR) / 'motions'
CLIP_INDEX_PATH = ROOT / 'benchmark/clip_index.json'
SETS = ROOT / 'benchmark/sets'
QUERIES_ROOT = ROOT / 'benchmark/queries'


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--set', choices=['37', '49', '1891'],
                       help='which evaluation set to answer, by its number of triples')
    group.add_argument('--fold', type=int, choices=[42, 43],
                       help='answer one of the two query sets in benchmark/queries instead')
    group.add_argument('--queries',
                       help='answer a query list written by benchmark.build_queries_full instead')
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--max_queries', type=int, default=None)
    parser.add_argument('--w_cluster', type=float, default=1.0)
    parser.add_argument('--w_q', type=float, default=2.0)
    parser.add_argument('--w_action', type=float, default=3.0)
    parser.add_argument('--topk_q', type=int, default=10)
    args = parser.parse_args()

    if args.set:
        queries_path = SETS / f'truebones_{args.set}.json'
    elif args.queries:
        queries_path = Path(args.queries)
    else:
        queries_path = QUERIES_ROOT / f'fold_{args.fold}/manifest.json'

    name_width = 5 if args.queries else 4
    weights = {'cluster': args.w_cluster, 'q': args.w_q, 'action': args.w_action}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading the clip features, the classifier and the clip index...")
    qc = open_clip_features()
    fname_to_idx = {m['fname']: i for i, m in enumerate(qc['meta'])}
    qsig_table = build_signature_table(qc)
    train_skels = set(OBJECT_SUBSETS_DICT['train'])
    clf = train_classifier(qc, train_skels)
    clip_index = json.load(open(CLIP_INDEX_PATH))

    queries = json.load(open(queries_path))['queries']
    if args.max_queries:
        queries = queries[:args.max_queries]
    print(f"{queries_path}: {len(queries)} queries")

    per_query = []
    t0 = time.time()
    n_ok = n_fail = 0
    for i, q in enumerate(queries):
        qid = q['query_id']
        rec = {'query_id': qid, 'skel_a': q['skel_a'], 'skel_b': q['skel_b'],
               'src_action': q['src_action'], 'src_fname': q['src_fname'],
               'status': 'pending'}
        try:
            r = rerank_one_query(q, clf, clip_index, qsig_table, qc,
                                 fname_to_idx, weights, topk_q=args.topk_q)
            picked = r['picked_fname']
            motion = np.load(MOTION_DIR / picked).astype(np.float32)
            np.save(out_dir / f'query_{qid:0{name_width}d}.npy', motion)
            rec.update({'status': 'ok', 'picked_fname': picked,
                        'picked_action': r['picked_action'],
                        'picked_cluster': r['picked_cluster']})
            n_ok += 1
        except Exception as e:
            rec['status'] = f'error: {e}'
            n_fail += 1
        per_query.append(rec)
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(queries)}] ok={n_ok} fail={n_fail} ({time.time()-t0:.0f}s)")

    summary = {'method': 'ANCHOR', 'queries': str(queries_path),
               'n_queries': len(queries), 'n_ok': n_ok, 'n_failed': n_fail,
               'wall_clock_s': time.time() - t0, 'per_query': per_query}
    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved: {out_dir}/metrics.json")
    print(f"  ok={n_ok}, fail={n_fail}, wall_clock={time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
