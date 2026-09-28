"""Action cluster classifier: guess a clip's action cluster from its clip features alone.

A random forest is fitted on the training skeletons, using the clip features of
benchmark/clip_features.py as input and the cluster read off the file name as the label. It is
used in two places: the ANCHOR retrieval baseline and the random-same-cluster baseline both
ask it which cluster a source clip belongs to, without ever reading the source's own label.

Running this module directly gives the simplest baseline built on it, which answers a query by
picking a random clip of the predicted cluster on the target skeleton:

    python -m benchmark.action_classifier --folds 42 43 --out_dir outputs/action_classifier
"""
from __future__ import annotations
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from benchmark.action_taxonomy import (
    ACTION_CLUSTERS, parse_action_from_filename, action_to_cluster,
)
from benchmark.clip_features import open_clip_features
from core.truebones.param_utils import OBJECT_SUBSETS_DICT, DATASET_DIR

ROOT = Path(__file__).resolve().parents[1]
MOTION_DIR = Path(DATASET_DIR) / 'motions'
CLIP_INDEX_PATH = ROOT / 'benchmark/clip_index.json'
QUERIES_ROOT = ROOT / 'benchmark/queries'

CLUSTERS = sorted(ACTION_CLUSTERS.keys())
CLUSTER_TO_IDX = {c: i for i, c in enumerate(CLUSTERS)}


def feature_vector(com_path, heading_vel, contact_sched, cadence, limb_usage):
    """Turn one clip's features into a vector of 30 numbers the forest can read."""
    feats = []
    # Centre-of-mass path: how far it moves per frame on average, how variable that is,
    # the total distance travelled, and the straight-line distance from start to end
    com = np.asarray(com_path, dtype=np.float32)
    if com.ndim == 1: com = com.reshape(-1, 3) if com.shape[0] % 3 == 0 else com.reshape(-1, 1)
    if com.shape[0] > 1:
        steps = np.linalg.norm(np.diff(com, axis=0), axis=-1)
        feats += [float(steps.mean()), float(steps.std()), float(steps.sum()),
                  float(np.linalg.norm(com[-1] - com[0]))]
    else:
        feats += [0.0, 0.0, 0.0, 0.0]
    # Forward speed: average, spread, and average size regardless of direction
    hv = np.asarray(heading_vel, dtype=np.float32)
    if hv.ndim == 1 and hv.size > 0:
        feats += [float(hv.mean()), float(hv.std()), float(np.abs(hv).mean())]
    elif hv.ndim == 2 and hv.size > 0:
        feats += [float(hv.mean()), float(hv.std()), float(np.abs(hv).mean())]
    else:
        feats += [0.0, 0.0, 0.0]
    # Contacts: how much of the time each limb is on the ground, six busiest limbs first
    cs = np.asarray(contact_sched, dtype=np.float32)
    if cs.ndim == 1: cs = cs.reshape(-1, 1)
    if cs.size > 0:
        density = cs.mean(axis=0)
        density_sorted = np.sort(density)[::-1]
        density_top6 = np.pad(density_sorted, (0, max(0, 6 - len(density_sorted))))[:6]
        feats += list(map(float, density_top6))
        # how often contacts switch on and off
        if cs.shape[0] > 1:
            change_rate = float(np.abs(np.diff(cs.astype(np.float32), axis=0)).mean())
        else:
            change_rate = 0.0
        feats.append(change_rate)
    else:
        feats += [0.0] * 7
    # How many steps per second
    feats.append(float(cadence))
    # Share of the movement done by each limb, six busiest first
    lu = np.asarray(limb_usage, dtype=np.float32)
    lu_sorted = np.sort(lu)[::-1]
    lu_top6 = np.pad(lu_sorted, (0, max(0, 6 - len(lu_sorted))))[:6]
    feats += list(map(float, lu_top6))
    # Whether the movement is spread over many limbs or concentrated in a few
    if lu.sum() > 0:
        p = lu / (lu.sum() + 1e-9)
        entropy = float(-(p * np.log(p + 1e-12)).sum())
    else:
        entropy = 0.0
    feats.append(entropy)
    return np.array(feats, dtype=np.float32)


def train_classifier(qc, train_skels):
    """Fit the random forest on the clips of `train_skels`."""
    from sklearn.ensemble import RandomForestClassifier
    meta = qc['meta']
    X, y = [], []
    for i, m in enumerate(meta):
        if m['skeleton'] not in train_skels:
            continue
        action = parse_action_from_filename(m['fname'])
        cluster = action_to_cluster(action)
        if cluster is None or cluster not in CLUSTER_TO_IDX:
            continue
        feat = feature_vector(qc['com_path'][i], qc['heading_vel'][i],
                              qc['contact_sched'][i], qc['cadence'][i],
                              qc['limb_usage'][i])
        X.append(feat)
        y.append(CLUSTER_TO_IDX[cluster])
    X = np.array(X)
    y = np.array(y)
    print(f"Training set: {X.shape}, {len(set(y))} clusters")
    print(f"Cluster dist: {[(CLUSTERS[i], int((y==i).sum())) for i in range(len(CLUSTERS))]}")
    clf = RandomForestClassifier(n_estimators=200, max_depth=20, n_jobs=-1, random_state=42)
    clf.fit(X, y)
    train_acc = float((clf.predict(X) == y).mean())
    print(f"Train accuracy: {train_acc:.3f}")
    return clf


def build_target_pool(clip_index_path=CLIP_INDEX_PATH):
    """{(skel, cluster) -> [fname, ...]}."""
    cidx = json.load(open(clip_index_path))
    pool = defaultdict(list)
    for skel, clusters in cidx['index'].items():
        for cluster, clips in clusters.items():
            for clip in clips:
                pool[(skel, cluster)].append(clip['fname'])
    return pool


def gen_predictions_for_fold(clf, qc, target_pool, fold_seed, out_root,
                             queries_root=QUERIES_ROOT):
    manifest = json.load(open(Path(queries_root) / f'fold_{fold_seed}/manifest.json'))
    rng = random.Random(fold_seed)
    out_dir = Path(out_root) / f'fold_{fold_seed}'
    out_dir.mkdir(parents=True, exist_ok=True)

    fname_to_idx = {m['fname']: i for i, m in enumerate(qc['meta'])}

    n_done = 0
    n_correct = 0
    n_skipped = 0
    n_fallback = 0
    for q in manifest['queries']:
        qid = q['query_id']
        skel_b = q['skel_b']
        src_fname = q['src_fname']
        gt_action = q['src_action']
        gt_cluster = action_to_cluster(gt_action)
        if src_fname not in fname_to_idx:
            n_skipped += 1
            continue
        idx = fname_to_idx[src_fname]
        feat = feature_vector(qc['com_path'][idx], qc['heading_vel'][idx],
                              qc['contact_sched'][idx], qc['cadence'][idx],
                              qc['limb_usage'][idx])
        pred_cluster_idx = int(clf.predict(feat[None, :])[0])
        pred_cluster = CLUSTERS[pred_cluster_idx]
        if gt_cluster == pred_cluster:
            n_correct += 1
        cands = target_pool.get((skel_b, pred_cluster), [])
        if not cands:
            # That group has no clip on this skeleton, so take a clip from any group
            for c in CLUSTERS:
                cands = target_pool.get((skel_b, c), [])
                if cands: break
            n_fallback += 1
        if not cands:
            n_skipped += 1
            continue
        chosen = rng.choice(cands)
        motion = np.load(MOTION_DIR / chosen).astype(np.float32)
        np.save(out_dir / f'query_{qid:04d}.npy', motion)
        n_done += 1

    print(f"fold {fold_seed}: {n_done}/{len(manifest['queries'])} done, "
          f"{n_correct} correct ({100*n_correct/max(n_done,1):.1f}%), "
          f"{n_fallback} fallback, {n_skipped} skipped")
    return out_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--folds', nargs='+', type=int, default=[42, 43])
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--queries_root', default=str(QUERIES_ROOT))
    args = parser.parse_args()

    print("Loading the clip features...")
    qc = open_clip_features()
    print(f"  {len(qc['meta'])} clips, {len(set(m['skeleton'] for m in qc['meta']))} skels")

    train_skels = set(OBJECT_SUBSETS_DICT['train'])
    print(f"Train skels: {len(train_skels)}")

    print("\nTraining classifier...")
    clf = train_classifier(qc, train_skels)

    print("\nBuilding target pool from the clip index...")
    target_pool = build_target_pool()
    print(f"  {len(target_pool)} (skel, cluster) entries")

    for fold in args.folds:
        print(f"\n=== Generating predictions for fold {fold} ===")
        gen_predictions_for_fold(clf, qc, target_pool, fold, args.out_dir,
                                 queries_root=args.queries_root)


if __name__ == '__main__':
    main()
