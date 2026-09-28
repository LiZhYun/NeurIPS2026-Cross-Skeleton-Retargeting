"""List every possible query, rather than drawing a sample of them.

benchmark/build_queries.py draws three hundred queries at a time, balanced across the action
groups. This instead takes every clip in the dataset and pairs it with every other skeleton
that has at least one clip of the same action group to judge the answer against. The result is
about thirty thousand queries covering every pair of skeletons. The appendix uses it to show
that a method's score does not depend on which queries happened to be drawn.

    python -m benchmark.build_queries_full --out_dir <folder>

The file is about seventy megabytes, so it is not included here. Build it when you need it.
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from benchmark.build_queries import (
    load_clip_index_full, build_query, TEST_SKELETONS, DEFAULT_CLIP_INDEX,
)

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--clip_index', default=str(DEFAULT_CLIP_INDEX))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_per_skel_a', type=int, default=0,
                        help='use at most this many clips per skeleton (0 means all of them)')
    args = parser.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    by_skel_action, by_skel_cluster, by_skel = load_clip_index_full(args.clip_index)
    all_skels = sorted(by_skel.keys())
    test_skels = set(TEST_SKELETONS)

    queries = []
    qid = 0
    n_skels = len(all_skels)
    n_pairs_attempted = 0
    n_pairs_with_positives = 0

    for sa_idx, skel_a in enumerate(sorted(all_skels)):
        src_clips = by_skel.get(skel_a, [])
        if args.max_per_skel_a > 0 and len(src_clips) > args.max_per_skel_a:
            idx = rng.choice(len(src_clips), args.max_per_skel_a, replace=False)
            src_clips = [src_clips[i] for i in idx]
        for src_clip in src_clips:
            for skel_b in sorted(all_skels):
                if skel_b == skel_a: continue
                n_pairs_attempted += 1
                # Was neither skeleton seen in training, one of them, or both?
                a_held = skel_a in test_skels
                b_held = skel_b in test_skels
                if a_held and b_held:
                    split = 'test_test'
                elif a_held or b_held:
                    split = 'mixed'
                else:
                    split = 'train_train'
                # Skip the pair if the target skeleton has nothing to judge the answer against
                src_cluster = src_clip['cluster']
                if not by_skel_cluster.get((skel_b, src_cluster), []):
                    continue
                n_pairs_with_positives += 1
                q = build_query(src_clip, skel_a, skel_b,
                                by_skel_action, by_skel_cluster, by_skel,
                                rng, qid, split)
                queries.append(q)
                qid += 1
        if (sa_idx + 1) % 10 == 0:
            print(f'  skel_a {sa_idx + 1}/{n_skels} processed, {len(queries)} queries built')


    by_split = defaultdict(int); by_cluster_split = defaultdict(int)
    n_cluster_eligible = 0; n_exact_eligible = 0
    for q in queries:
        by_split[q['split']] += 1
        by_cluster_split[(q['split'], q['cluster'])] += 1
        if q['cluster_tier_eligible']: n_cluster_eligible += 1
        if q['exact_tier_eligible']: n_exact_eligible += 1

    print(f'\nTotal queries: {len(queries)}')
    print(f'  pairs_attempted: {n_pairs_attempted}, pairs_with_positives: {n_pairs_with_positives}')
    print(f'  cluster_tier_eligible: {n_cluster_eligible}')
    print(f'  exact_tier_eligible: {n_exact_eligible}')
    print(f'  by split: {dict(by_split)}')

    manifest = {
        'version': 'truebones_queries_full',
        'description': 'Every (source clip, target skeleton) query with at least one '
                       'same-cluster reference clip on the target skeleton',
        'seed': args.seed,
        'n_queries': len(queries),
        'cluster_eligible': n_cluster_eligible,
        'exact_eligible': n_exact_eligible,
        'by_split': dict(by_split),
        'test_skeletons': sorted(test_skels),
        'queries': queries,
    }
    out_path = out / 'manifest.json'
    with open(out_path, 'w') as f:
        json.dump(manifest, f, indent=1, default=str)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
