"""Action-level AUC: does a produced motion look more like the right action than a wrong one?

For each query, the motion a method produced is compared with a set of real clips on the
target skeleton, and those clips are put in order from closest to furthest. Two questions are
asked of that order:

  cluster tier   are clips of the same broad action group ranked above clips of another group?
  exact tier     are clips of the very same action ranked above clips of the same group but a
                 different action?

The answer for one query is the chance that a randomly picked right-action clip comes out
ahead of a randomly picked wrong-action one. This is the area under the ROC curve, written
AUC. One means the order is always right, and one half means the method tells us nothing.

The score is averaged over queries. The interval around it is found by resampling whole
source-and-target skeleton pairs rather than single queries, because two queries that share
the same pair of skeletons are not independent of each other. The same numbers are reported
again for the queries where neither skeleton was used in training, and once per action group.

    python -m benchmark.action_auc --method_dir outputs/anytop/fold_42 \\
        --fold 42 --method_name AnyTop --distance procrustes

Writes <out_dir>/action_auc_<distance>.json.
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from benchmark.distances import (
    dist_zscore_dtw_inv, dist_procrustes_trajectory, q_component_distances,
)
from benchmark.clip_features import load_contact_groups, load_clip_features
from benchmark.slot_encoder import encode_motion_to_invariant
from core.truebones.param_utils import DATASET_DIR

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(DATASET_DIR)
MOTION_DIR = DATA_ROOT / 'motions'
QUERIES_ROOT = ROOT / 'benchmark/queries'


def make_skel_cond(cond_dict, name):
    return {
        'joints_names': cond_dict[name]['joints_names'],
        'parents': cond_dict[name]['parents'],
        'object_type': name,
    }


def _recover_positions(motion, skel_name, cond_dict):
    """Return the joint positions of a motion, one point in space per joint per frame.

    A file that already holds positions (three numbers per joint, as Motion2Motion-BVH writes)
    is returned as it is. A file in the dataset's own format (thirteen numbers per joint) is
    converted first.
    """
    n_joints = len(cond_dict[skel_name]['parents'])
    if motion.shape[1] > n_joints:
        motion = motion[:, :n_joints]
    if motion.shape[-1] == 3:
        # Already positions, no recovery needed
        return motion.astype(np.float32)
    from core.truebones.motion_process import recover_from_bvh_ric_np
    return recover_from_bvh_ric_np(motion.astype(np.float32))


def _features_from_motion(motion_13, skel_name, cond_dict, contact_groups):
    """Compute the clip features of a motion held in memory."""
    from benchmark.clip_features import extract_features_from_array
    cg = contact_groups.get(skel_name)
    return extract_features_from_array(motion_13, cond_dict[skel_name], cg)


def _zscore_feature_pool(comp_dicts):
    """Turn the four feature differences of each candidate into one number.

    The four quantities are in different units, so each is first centred and scaled using all
    the candidates of this query, then added up. Candidates of every kind go into that scaling,
    so no kind of candidate is favoured.
    """
    if not comp_dicts:
        return []
    keys = sorted(comp_dicts[0].keys())
    arrs = {k: np.asarray([c[k] for c in comp_dicts], dtype=np.float64) for k in keys}
    out = np.zeros(len(comp_dicts), dtype=np.float64)
    for k in keys:
        v = arrs[k]
        s = float(v.std()) if v.std() > 1e-12 else 1.0
        m = float(v.mean())
        out += (v - m) / s
    return out.tolist()


def auc_safe(pos_dists, neg_dists):
    if not pos_dists or not neg_dists:
        return None
    scores = -np.concatenate([pos_dists, neg_dists])
    labels = np.concatenate([np.ones(len(pos_dists)), np.zeros(len(neg_dists))])
    try: return float(roc_auc_score(labels, scores))
    except Exception: return None


def block_bootstrap_ci(values_with_blocks, n_boot=500, ci=0.95, seed=42):
    """Mean of the values with a confidence interval, resampling whole blocks.

    Each value comes with the name of the block it belongs to. Resampling blocks rather than
    single values keeps values that are not independent of each other together.
    """
    if not values_with_blocks: return (0.0, 0.0, 0.0)
    rng = np.random.RandomState(seed)
    block_to_values = defaultdict(list)
    for v, b in values_with_blocks:
        if v is None: continue
        block_to_values[b].append(v)
    blocks = list(block_to_values.keys())
    if not blocks: return (0.0, 0.0, 0.0)
    means = []
    for _ in range(n_boot):
        sampled_blocks = rng.choice(len(blocks), len(blocks), replace=True)
        all_vals = []
        for bi in sampled_blocks:
            all_vals.extend(block_to_values[blocks[bi]])
        means.append(np.mean(all_vals) if all_vals else 0.0)
    means = np.array(means)
    lo = float(np.percentile(means, (1 - ci) / 2 * 100))
    hi = float(np.percentile(means, (1 + ci) / 2 * 100))
    raw_mean = float(np.mean([v for v, _ in values_with_blocks if v is not None]))
    return (lo, raw_mean, hi)


def _compute_distance(distance_metric, pos_pred, inv_pred, feat_pred,
                      ref_motion, skel_b, cond_dict, skel_cond,
                      contact_groups, get_features, ref_fname):
    if distance_metric == 'procrustes':
        pos_ref = _recover_positions(ref_motion, skel_b, cond_dict)
        return dist_procrustes_trajectory(pos_pred, pos_ref)
    elif distance_metric == 'zscore_dtw':
        inv_ref = encode_motion_to_invariant(ref_motion, skel_cond)
        return dist_zscore_dtw_inv(inv_pred, inv_ref)
    elif distance_metric == 'q_component':
        feat_ref = (get_features(ref_fname)
                    or _features_from_motion(ref_motion, skel_b, cond_dict, contact_groups))
        return q_component_distances(feat_pred, feat_ref)
    raise ValueError(distance_metric)


def evaluate_method(method_dir: Path, fold_seed: int, method_name: str,
                    distance: str, max_queries: int = 10000,
                    queries_root: Path = QUERIES_ROOT):
    manifest = json.load(open(queries_root / f'fold_{fold_seed}/manifest.json'))
    queries = manifest['queries']
    cond_dict = np.load(DATA_ROOT / 'cond.npy', allow_pickle=True).item()
    contact_groups = load_contact_groups()
    get_features = load_clip_features()

    per_query = []
    n_skipped = 0
    for q in queries[:max_queries]:
        qid = q['query_id']
        skel_b = q['skel_b']
        pred_path = method_dir / f'query_{qid:04d}.npy'
        if not pred_path.exists():
            n_skipped += 1; continue
        try:
            pred = np.load(pred_path)
            skel_cond = make_skel_cond(cond_dict, skel_b)
            is_pos_only = pred.ndim == 3 and pred.shape[-1] == 3
            if is_pos_only and distance in ('zscore_dtw', 'q_component'):
                n_skipped += 1; continue
            if not is_pos_only:
                inv_pred = encode_motion_to_invariant(pred.astype(np.float32), skel_cond)
                feat_pred = (_features_from_motion(pred, skel_b, cond_dict, contact_groups)
                             if distance == 'q_component' else None)
            else:
                inv_pred = None; feat_pred = None
            pos_pred = _recover_positions(pred, skel_b, cond_dict)

            args_for_dist = dict(distance_metric=distance, pos_pred=pos_pred,
                                 inv_pred=inv_pred, feat_pred=feat_pred, skel_b=skel_b,
                                 cond_dict=cond_dict, skel_cond=skel_cond,
                                 contact_groups=contact_groups, get_features=get_features)

            # Cluster tier: clips of the same action group against clips of another group
            pos_c_dists, neg_c_dists = [], []
            for p in q['positives_cluster']:
                ref = np.load(MOTION_DIR / p['fname']).astype(np.float32)
                pos_c_dists.append(_compute_distance(ref_motion=ref, ref_fname=p['fname'],
                                                     **args_for_dist))
            for a in q['adversarials_easy']:
                ref = np.load(MOTION_DIR / a['fname']).astype(np.float32)
                neg_c_dists.append(_compute_distance(ref_motion=ref, ref_fname=a['fname'],
                                                     **args_for_dist))

            # Exact tier: clips of the same action against clips of the same group only
            pos_e_dists, neg_e_dists = [], []
            for p in q['positives_exact']:
                ref = np.load(MOTION_DIR / p['fname']).astype(np.float32)
                pos_e_dists.append(_compute_distance(ref_motion=ref, ref_fname=p['fname'],
                                                     **args_for_dist))
            for a in q['adversarials_hard']:
                ref = np.load(MOTION_DIR / a['fname']).astype(np.float32)
                neg_e_dists.append(_compute_distance(ref_motion=ref, ref_fname=a['fname'],
                                                     **args_for_dist))

            # Put the four feature differences on a common scale, using this query's candidates
            if distance == 'q_component':
                pool_c = pos_c_dists + neg_c_dists
                if pool_c:
                    z_c = _zscore_feature_pool(pool_c)
                    pos_c_dists = z_c[:len(pos_c_dists)]
                    neg_c_dists = z_c[len(pos_c_dists):]
                pool_e = pos_e_dists + neg_e_dists
                if pool_e:
                    z_e = _zscore_feature_pool(pool_e)
                    pos_e_dists = z_e[:len(pos_e_dists)]
                    neg_e_dists = z_e[len(pos_e_dists):]

            cluster_auc = auc_safe(pos_c_dists, neg_c_dists) if q['cluster_tier_eligible'] else None
            exact_auc = auc_safe(pos_e_dists, neg_e_dists) if q['exact_tier_eligible'] else None

            per_query.append({
                'query_id': qid,
                'split': q['split'],
                'cluster': q['cluster'],
                'skel_a': q['skel_a'], 'skel_b': skel_b,
                'support_reason': q['support_reason'],
                'n_pos_cluster': q['n_pos_cluster'], 'n_pos_exact': q['n_pos_exact'],
                'n_easy': q['n_easy'], 'n_hard': q['n_hard'],
                'cluster_tier_eligible': q['cluster_tier_eligible'],
                'exact_tier_eligible': q['exact_tier_eligible'],
                'cluster_auc': cluster_auc,
                'exact_auc': exact_auc,
                'block_id': f"{q['skel_a']}__{q['skel_b']}",
            })
        except Exception as e:
            print(f"  q{qid}: FAIL {e}")
            n_skipped += 1

    # Combine the per-query scores
    out = {
        'method': method_name, 'distance': distance, 'fold': fold_seed,
        'n_queries': len(per_query), 'n_skipped': n_skipped,
        'per_query': per_query,
    }
    # Cluster tier, over all queries and over the held-out ones
    cluster_blocks = [(r['cluster_auc'], r['block_id']) for r in per_query if r['cluster_auc'] is not None]
    cluster_blocks_tt = [(r['cluster_auc'], r['block_id']) for r in per_query
                         if r['cluster_auc'] is not None and r['split'] == 'test_test']
    out['cluster_tier_overall_auc_ci'] = block_bootstrap_ci(cluster_blocks)
    out['cluster_tier_test_test_auc_ci'] = block_bootstrap_ci(cluster_blocks_tt)
    out['cluster_tier_n'] = len(cluster_blocks)
    out['cluster_tier_test_test_n'] = len(cluster_blocks_tt)

    # Exact tier, over all queries and over the held-out ones
    exact_blocks = [(r['exact_auc'], r['block_id']) for r in per_query if r['exact_auc'] is not None]
    exact_blocks_tt = [(r['exact_auc'], r['block_id']) for r in per_query
                       if r['exact_auc'] is not None and r['split'] == 'test_test']
    out['exact_tier_overall_auc_ci'] = block_bootstrap_ci(exact_blocks)
    out['exact_tier_test_test_auc_ci'] = block_bootstrap_ci(exact_blocks_tt)
    out['exact_tier_n'] = len(exact_blocks)
    out['exact_tier_test_test_n'] = len(exact_blocks_tt)

    # Cluster tier again, one score per action group, held-out queries only
    by_cluster_tt = defaultdict(list)
    for r in per_query:
        if r['cluster_auc'] is not None and r['split'] == 'test_test':
            by_cluster_tt[r['cluster']].append((r['cluster_auc'], r['block_id']))
    out['per_cluster_test_test_cluster_tier'] = {
        c: {'auc_ci': block_bootstrap_ci(v), 'n': len(v)}
        for c, v in by_cluster_tt.items()
    }

    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--method_dir', required=True,
                        help='folder holding this method\'s answers, query_0000.npy and so on')
    parser.add_argument('--fold', type=int, required=True, choices=[42, 43],
                        help='which of the two query sets in benchmark/queries to score on')
    parser.add_argument('--method_name', required=True,
                        help='name to record in the result file')
    parser.add_argument('--distance', choices=['procrustes', 'zscore_dtw', 'q_component'],
                        default='procrustes',
                        help='how to measure the distance between two motions')
    parser.add_argument('--queries_root', default=str(QUERIES_ROOT),
                        help='folder holding the query sets')
    parser.add_argument('--out_dir', default=None,
                        help='where to write the result (default: beside the answers)')
    args = parser.parse_args()

    method_dir = Path(args.method_dir)
    out_dir = Path(args.out_dir) if args.out_dir else method_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = evaluate_method(method_dir, args.fold, args.method_name, args.distance,
                              queries_root=Path(args.queries_root))
    out_path = out_dir / f'action_auc_{args.distance}.json'
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)

    cl = summary['cluster_tier_overall_auc_ci']
    cl_tt = summary['cluster_tier_test_test_auc_ci']
    ex = summary['exact_tier_overall_auc_ci']
    ex_tt = summary['exact_tier_test_test_auc_ci']
    print(f"\n=== {args.method_name} fold {args.fold} {args.distance} ===")
    print(f"  cluster-tier overall:   {cl[1]:.3f} [{cl[0]:.3f}, {cl[2]:.3f}] (n={summary['cluster_tier_n']})")
    print(f"  cluster-tier held out:  {cl_tt[1]:.3f} [{cl_tt[0]:.3f}, {cl_tt[2]:.3f}] (n={summary['cluster_tier_test_test_n']})")
    print(f"  exact-tier   overall:   {ex[1]:.3f} [{ex[0]:.3f}, {ex[2]:.3f}] (n={summary['exact_tier_n']})")
    print(f"  exact-tier   held out:  {ex_tt[1]:.3f} [{ex_tt[0]:.3f}, {ex_tt[2]:.3f}] (n={summary['exact_tier_test_test_n']})")
    print(f"  per-cluster (cluster-tier, held out):")
    for c, info in sorted(summary['per_cluster_test_test_cluster_tier'].items(),
                          key=lambda x: -x[1]['n']):
        a = info['auc_ci']
        print(f"    {c}: {a[1]:.3f} [{a[0]:.3f}, {a[2]:.3f}] (n={info['n']})")
    print(f"  Saved: {out_path}")


if __name__ == '__main__':
    main()
