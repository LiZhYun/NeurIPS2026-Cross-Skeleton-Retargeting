"""Answer a whole set of queries with Motion2Motion working straight on the motion files.

Every query needs one clip of the target animal for the method to take its style from. That
clip is drawn at random from what the animal has, leaving out every clip the answer will later
be compared against, so the method is never handed the answer it is being asked for. The draw
depends only on the query and the target animal, not on the order the queries are worked
through, so it comes out the same every time. Answers are stretched to the typical length of
the clips they will be compared against.

    python -m methods.motion2motion.generate_direct --set 49 --out_dir outputs/motion2motion_direct/set49
"""
from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from core.truebones.param_utils import DATASET_DIR
from methods.common.queries import (
    REFERENCE_KEYS, per_query_seed, reference_length, stretch_to_length,
)
from methods.motion2motion.core import author_sparse_mapping, retarget, load_assets

ROOT = Path(__file__).resolve().parents[2]
MOTION_DIR = Path(DATASET_DIR) / 'motions'
SETS = ROOT / 'benchmark/sets'
QUERIES_ROOT = ROOT / 'benchmark/queries'

def list_target_skel_motions(skel_name, motion_dir):
    return sorted([f for f in os.listdir(motion_dir)
                   if (f.startswith(skel_name + '___') or f.startswith(skel_name + '_'))
                   and f.endswith('.npy')])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--set', choices=['37', '49', '1891'],
                       help='which evaluation set to answer, by its number of triples')
    group.add_argument('--fold', type=int, choices=[42, 43],
                       help='answer one of the two query sets in benchmark/queries instead')
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--max_queries', type=int, default=10000)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    if args.set:
        queries_path = SETS / f'truebones_{args.set}.json'
        seed_tag = 999   # the value the paper used when answering the evaluation sets
    else:
        queries_path = QUERIES_ROOT / f'fold_{args.fold}/manifest.json'
        seed_tag = args.fold

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Queries: {queries_path}")
    print(f"Output dir: {out_dir}")

    cond, contact_groups, _ = load_assets()

    with open(queries_path) as f:
        manifest = json.load(f)
    queries = manifest['queries'][:args.max_queries]
    print(f"Running {len(queries)} queries")

    tgt_clip_cache = {}
    per_query = []
    t_total = time.time()

    for i, q in enumerate(queries):
        qid = q['query_id']
        skel_a = q['skel_a']
        skel_b = q['skel_b']
        src_fname = q['src_fname']

        rec = {'query_id': qid, 'cluster': q.get('cluster'), 'split': q.get('split'),
               'skel_a': skel_a, 'skel_b': skel_b, 'status': 'pending'}

        try:
            src_motion = np.load(MOTION_DIR / src_fname).astype(np.float32)

            if skel_b not in tgt_clip_cache:
                tgt_clip_cache[skel_b] = list_target_skel_motions(skel_b, MOTION_DIR)
            full_pool = tgt_clip_cache[skel_b]

            # The example clip must never be one the answer will be compared against
            forbidden = set()
            for key in REFERENCE_KEYS:
                for x in q.get(key, []):
                    forbidden.add(x['fname'])
            pool = [f for f in full_pool if f not in forbidden]
            if not pool:
                pool = full_pool
                rec['example_fallback'] = 'all_clips_are_references'

            qseed = per_query_seed(seed_tag, qid, skel_b)
            qrng = np.random.RandomState(qseed)
            ex_fname = pool[qrng.randint(0, len(pool))]
            ex_motion = np.load(MOTION_DIR / ex_fname).astype(np.float32)
            if ex_motion.shape[0] < 13:
                pool_long = [f for f in pool
                             if np.load(MOTION_DIR / f, mmap_mode='r').shape[0] >= 13]
                if not pool_long:
                    raise RuntimeError(f'No target clips >=13 frames for {skel_b}')
                ex_fname = pool_long[qrng.randint(0, len(pool_long))]
                ex_motion = np.load(MOTION_DIR / ex_fname).astype(np.float32)

            if skel_a not in contact_groups or skel_b not in contact_groups:
                rec['status'] = 'skipped_no_contact_groups'
                per_query.append(rec); continue

            pairs_ij, mapping_desc = author_sparse_mapping(
                skel_a, skel_b, cond, contact_groups, max_pairs=6)
            if not pairs_ij:
                rec['status'] = 'skipped_no_mapping'
                per_query.append(rec); continue
            src_j_idxs = [p[0] for p in pairs_ij]
            tgt_j_idxs = [p[1] for p in pairs_ij]

            t0 = time.time()
            output = retarget(
                src_motion, ex_motion, src_j_idxs, tgt_j_idxs,
                tgt_n_joints=ex_motion.shape[1],
                device=args.device, seed=qseed)
            runtime = time.time() - t0

            output = stretch_to_length(output, reference_length(q, output.shape[0]))

            np.save(out_dir / f'query_{qid:04d}.npy', output.astype(np.float32))
            rec.update({
                'status': 'ok', 'example_fname': ex_fname,
                'n_pairs': len(pairs_ij), 'mapping_desc': mapping_desc,
                'runtime_s': runtime, 'output_T': int(output.shape[0]),
            })

        except Exception as e:
            rec['status'] = 'failed'
            rec['error'] = f'{type(e).__name__}: {e}'
            print(f'  q{qid} FAILED: {e}')

        per_query.append(rec)

        if (i + 1) % 25 == 0 or i == 0:
            elapsed = time.time() - t_total
            n_ok = sum(1 for r in per_query if r['status'] == 'ok')
            eta = elapsed / (i + 1) * (len(queries) - i - 1)
            print(f"  [{i+1}/{len(queries)}] elapsed {elapsed:.0f}s, "
                  f"ETA {eta:.0f}s, ok={n_ok}")

    n_ok = sum(1 for r in per_query if r['status'] == 'ok')
    summary = {
        'method': 'Motion2Motion-Direct',
        'queries': str(queries_path),
        'n_queries': len(per_query), 'n_ok': n_ok,
        'total_time_sec': time.time() - t_total,
        'per_query': per_query,
    }
    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n{n_ok}/{len(per_query)} ok. Saved to {out_dir}")


if __name__ == '__main__':
    main()
