"""Draw the two sets of queries used by the action-level AUC evaluation.

A query asks a method to produce a motion on one skeleton, given a clip of a different
skeleton doing something. Each query also carries the clips its answer will be judged
against, in two rounds of increasing difficulty. The easier round asks whether the answer
resembles clips of the same broad action group more than clips of another group. The harder
round asks whether it resembles clips of the very same action more than clips of the same
group but a different action.

Queries come in four blocks. The last uses only skeletons that no method was trained on,
which is where generalisation to new animals shows.

    python -m benchmark.build_queries --folds 42 43

The result is written to benchmark/queries/fold_<number>/manifest.json. Two sets are drawn,
with different random choices, so that a result can be checked against a second draw.
"""
from __future__ import annotations
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from benchmark.action_taxonomy import ACTION_CLUSTERS
from core.truebones.param_utils import HELD_OUT_SKELETONS

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLIP_INDEX = ROOT / 'benchmark/clip_index.json'
DEFAULT_OUT_ROOT = ROOT / 'benchmark/queries'

TEST_SKELETONS = set(HELD_OUT_SKELETONS)

ALL_CLUSTERS = sorted(ACTION_CLUSTERS.keys())

# How many queries to draw in each block
TARGETS = {'train': 50, 'dev': 50, 'mixed': 100, 'test_test': 100}

# Upper limits on how many clips accompany a query
MAX_HARD = 5              # same action group, different action
MAX_EASY = 5              # a different action group
MAX_DISTRACTORS = 20      # any other clip of the target skeleton
MAX_POSITIVES_CLUSTER = 8
MAX_POSITIVES_EXACT = 4
LENGTH_TOL = 0.30         # "similar length": within 30% of the source clip


def load_clip_index_full(clip_index_path):
    """Read the clip list and index it by skeleton and action, by skeleton and action group,
    and by skeleton alone."""
    cidx = json.load(open(clip_index_path))
    by_skel_action = defaultdict(list)
    by_skel_cluster = defaultdict(list)
    by_skel = defaultdict(list)
    for skel, clusters in cidx['index'].items():
        for cluster, clips in clusters.items():
            for clip in clips:
                rec = {'fname': clip['fname'], 'T': clip['T'],
                       'action': clip['action'], 'cluster': cluster}
                by_skel_action[(skel, clip['action'])].append(rec)
                by_skel_cluster[(skel, cluster)].append(rec)
                by_skel[skel].append(rec)
    return by_skel_action, by_skel_cluster, by_skel


def length_match(clips, target_T, tol=LENGTH_TOL):
    return [c for c in clips if abs(c['T'] - target_T) / max(target_T, 1) <= tol]


def build_query(src_clip, src_skel, tgt_skel, by_skel_action, by_skel_cluster, by_skel,
                rng, query_id, split):
    """Build one query: the clips it is judged against in both rounds, and whether each round
    has enough clips to be scored."""
    src_action = src_clip['action']
    src_cluster = src_clip['cluster']
    src_T = src_clip['T']

    # Easier round, clips the answer should resemble: same action group on the target
    pos_cluster_all = by_skel_cluster.get((tgt_skel, src_cluster), [])
    # Harder round, clips the answer should resemble: the very same action on the target
    pos_exact_all = by_skel_action.get((tgt_skel, src_action), [])

    # Easier round, clips it should not resemble: a different action group, similar length
    neg_easy_pool = [c for c in by_skel.get(tgt_skel, []) if c['cluster'] != src_cluster]
    neg_easy_pool = length_match(neg_easy_pool, src_T) or neg_easy_pool[:MAX_EASY*2]

    # Harder round, clips it should not resemble: same group, different action
    neg_hard_pool = [c for c in by_skel_cluster.get((tgt_skel, src_cluster), [])
                     if c['action'] != src_action]
    # These are scarce enough already, so do not also filter them by length

    # One entry per clip, and not too many
    pos_cluster = sorted({c['fname']: c for c in pos_cluster_all}.values(),
                        key=lambda c: c['fname'])[:MAX_POSITIVES_CLUSTER]
    pos_exact = sorted({c['fname']: c for c in pos_exact_all}.values(),
                       key=lambda c: c['fname'])[:MAX_POSITIVES_EXACT]

    # Draw the clips the answer should not resemble
    if neg_easy_pool:
        idx_easy = rng.choice(len(neg_easy_pool), min(MAX_EASY, len(neg_easy_pool)), replace=False)
        neg_easy = [neg_easy_pool[i] for i in idx_easy]
    else:
        neg_easy = []
    if neg_hard_pool:
        idx_hard = rng.choice(len(neg_hard_pool), min(MAX_HARD, len(neg_hard_pool)), replace=False)
        neg_hard = [neg_hard_pool[i] for i in idx_hard]
    else:
        neg_hard = []

    # Other clips of the target skeleton, for open-ended searching
    excluded = {c['fname'] for c in pos_cluster} | {c['fname'] for c in pos_exact}
    distractor_pool = [c for c in by_skel.get(tgt_skel, []) if c['fname'] not in excluded]
    if distractor_pool:
        n_take = min(MAX_DISTRACTORS, len(distractor_pool))
        idx_d = rng.choice(len(distractor_pool), n_take, replace=False)
        distractors = [distractor_pool[i] for i in idx_d]
    else:
        distractors = []

    n_easy = len(neg_easy)
    n_hard = len(neg_hard)
    n_pos_cluster = len(pos_cluster)
    n_pos_exact = len(pos_exact)

    # A round can only be scored if it has clips on both sides
    cluster_tier_eligible = (n_pos_cluster >= 1) and (n_easy >= 1)
    exact_tier_eligible = (n_pos_exact >= 1) and (n_hard >= 1)

    # Whether the target skeleton has the very same action, only the group, or neither
    if n_pos_exact >= 1:
        support_reason = 'exact_action'
    elif n_pos_cluster >= 1:
        support_reason = 'cluster_only'
    else:
        support_reason = 'absent'

    return {
        'query_id': query_id,
        'split': split,
        'skel_a': src_skel,
        'skel_b': tgt_skel,
        'cluster': src_cluster,
        'src_fname': src_clip['fname'],
        'src_T': src_T,
        'src_action': src_action,
        'positives_cluster': pos_cluster,
        'positives_exact': pos_exact,
        'adversarials_hard': neg_hard,
        'adversarials_easy': neg_easy,
        'distractors_same_target_skel': distractors,
        'n_pos_cluster': n_pos_cluster,
        'n_pos_exact': n_pos_exact,
        'n_hard': n_hard,
        'n_easy': n_easy,
        'cluster_tier_eligible': cluster_tier_eligible,
        'exact_tier_eligible': exact_tier_eligible,
        'support_reason': support_reason,
    }


def build_fold(seed, by_skel_action, by_skel_cluster, by_skel, all_skels):
    rng = np.random.RandomState(seed)
    py_rng = random.Random(seed)
    queries = []
    qid = 0

    train_skels = sorted(set(all_skels) - TEST_SKELETONS)
    test_skels_l = sorted(TEST_SKELETONS)

    # Everything that could serve as a source clip, by action group and by block
    candidates_by_cluster_and_split = defaultdict(lambda: {'train_train': [], 'mixed': [], 'test_test': []})
    for skel, clips in by_skel.items():
        for c in clips:
            if c['cluster'] not in ALL_CLUSTERS: continue
            split_class = 'test_test' if skel in TEST_SKELETONS else 'train_train'
            candidates_by_cluster_and_split[c['cluster']][split_class].append((skel, c))

    def sample_skel_pair_for(split_name, allowed_clusters):
        """Draw a source skeleton, a source clip and a target skeleton for this block."""
        cluster = py_rng.choice(allowed_clusters)
        if split_name in ('train', 'dev'):
            cands = candidates_by_cluster_and_split[cluster]['train_train']
            if not cands: return None
            src_skel, src_clip = py_rng.choice(cands)
            tgt_skel = py_rng.choice([s for s in train_skels if s != src_skel])
        elif split_name == 'mixed':
            # one skeleton seen in training, one not
            if py_rng.random() < 0.5:
                cands = candidates_by_cluster_and_split[cluster]['train_train']
                if not cands: return None
                src_skel, src_clip = py_rng.choice(cands)
                tgt_skel = py_rng.choice(test_skels_l)
            else:
                cands = candidates_by_cluster_and_split[cluster]['test_test']
                if not cands: return None
                src_skel, src_clip = py_rng.choice(cands)
                tgt_skel = py_rng.choice(train_skels)
        else:  # test_test
            cands = candidates_by_cluster_and_split[cluster]['test_test']
            if not cands: return None
            src_skel, src_clip = py_rng.choice(cands)
            tgt_skel = py_rng.choice([s for s in test_skels_l if s != src_skel])
        return src_skel, src_clip, tgt_skel

    for split, target_n in TARGETS.items():
        # Leave out action groups that have nothing to draw from in this block
        if split in ('train', 'dev'):
            available = [c for c in ALL_CLUSTERS if candidates_by_cluster_and_split[c]['train_train']]
        elif split == 'mixed':
            available = [c for c in ALL_CLUSTERS if (
                candidates_by_cluster_and_split[c]['train_train'] or
                candidates_by_cluster_and_split[c]['test_test'])]
        else:
            available = [c for c in ALL_CLUSTERS if candidates_by_cluster_and_split[c]['test_test']]
        if not available:
            print(f"  Warning: block {split} has no action group to draw from")
            continue
        n_built = 0
        max_tries = target_n * 50
        tries = 0
        while n_built < target_n and tries < max_tries:
            tries += 1
            res = sample_skel_pair_for(split, available)
            if res is None: continue
            src_skel, src_clip, tgt_skel = res
            q = build_query(src_clip, src_skel, tgt_skel,
                           by_skel_action, by_skel_cluster, by_skel,
                           rng, qid, split)
            # Keep the query only if at least one of the two rounds can be scored
            if not (q['cluster_tier_eligible'] or q['exact_tier_eligible']):
                continue
            queries.append(q)
            qid += 1
            n_built += 1
        print(f"  fold {seed} split {split}: built {n_built}/{target_n} ({tries} tries)")

    return queries


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--folds', nargs='+', type=int, default=[42, 43])
    parser.add_argument('--clip_index', default=str(DEFAULT_CLIP_INDEX))
    parser.add_argument('--out_root', default=str(DEFAULT_OUT_ROOT))
    args = parser.parse_args()

    OUT_ROOT = Path(args.out_root)

    print("Loading clip index...")
    by_skel_action, by_skel_cluster, by_skel = load_clip_index_full(args.clip_index)
    all_skels = sorted(by_skel.keys())
    print(f"  {len(all_skels)} skels, {sum(len(c) for c in by_skel.values())} total clips")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for seed in args.folds:
        print(f"\n=== Building fold {seed} ===")
        queries = build_fold(seed, by_skel_action, by_skel_cluster, by_skel, all_skels)

        from collections import Counter
        cluster_dist = Counter(q['cluster'] for q in queries)
        split_dist = Counter(q['split'] for q in queries)
        ct_eligible = sum(q['cluster_tier_eligible'] for q in queries)
        et_eligible = sum(q['exact_tier_eligible'] for q in queries)
        n_hard_dist = Counter(q['n_hard'] for q in queries)
        print(f"  Total queries: {len(queries)}")
        print(f"  Splits: {dict(split_dist)}")
        print(f"  Clusters: {dict(cluster_dist)}")
        print(f"  Cluster-tier eligible: {ct_eligible}/{len(queries)}")
        print(f"  Exact-tier eligible: {et_eligible}/{len(queries)}")
        print(f"  n_hard distribution: {dict(n_hard_dist)}")

        out_dir = OUT_ROOT / f'fold_{seed}'
        out_dir.mkdir(exist_ok=True)
        with open(out_dir / 'manifest.json', 'w') as f:
            json.dump({
                'version': 'truebones_queries',
                'seed': seed,
                'all_clusters': ALL_CLUSTERS,
                'test_skeletons': sorted(TEST_SKELETONS),
                'max_adversarials_hard': MAX_HARD, 'max_adversarials_easy': MAX_EASY,
                'max_distractors': MAX_DISTRACTORS,
                'n_queries': len(queries),
                'queries': queries,
            }, f, indent=2)
        print(f"  Saved: {out_dir / 'manifest.json'}")


if __name__ == '__main__':
    main()
