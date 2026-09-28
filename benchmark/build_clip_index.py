"""List every clip in the dataset that the benchmark can use, by skeleton and action.

The action of a clip is written in its file name. This goes through the prepared dataset,
reads each clip's action and length, and leaves out clips shorter than thirty frames and
clips whose action is not one of the ten the benchmark recognises. What is left is written to
benchmark/clip_index.json. That file is the pool the query builders draw from and the pool
the retrieval methods search.

    python -m benchmark.build_clip_index
"""
from __future__ import annotations
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from benchmark.action_taxonomy import (
    parse_action_from_filename, action_to_cluster, is_other_label,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = 'dataset/truebones/zoo/truebones_processed'
DEFAULT_OUT = ROOT / 'benchmark/clip_index.json'

MIN_FRAMES = 30  # minimum clip length


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--data_root', default=DEFAULT_DATA_ROOT,
                        help='the prepared dataset folder')
    parser.add_argument('--out', default=str(DEFAULT_OUT))
    args = parser.parse_args()

    data_root = Path(args.data_root)
    motion_dir = data_root / 'motions'
    out_path = Path(args.out)

    print("Loading data...")
    cond = np.load(data_root / 'cond.npy', allow_pickle=True).item()
    skel_names = sorted(cond.keys())
    print(f"  {len(skel_names)} skeletons in cond_dict")

    clip_files = sorted([f for f in os.listdir(motion_dir) if f.endswith('.npy')])
    print(f"  {len(clip_files)} motion files")

    # Sort the clips by skeleton and action
    index = defaultdict(lambda: defaultdict(list))
    n_dropped_other = 0
    n_dropped_short = 0
    n_dropped_no_skel = 0
    n_kept = 0

    for f in clip_files:
        # The skeleton is the start of the file name, before the action
        skel = None
        # Most names separate the two with three underscores
        if '___' in f:
            candidate = f.split('___')[0]
            if candidate in cond:
                skel = candidate
        # Otherwise take the longest skeleton name the file starts with
        if skel is None:
            for s in sorted(skel_names, key=len, reverse=True):
                if f.startswith(s + '___') or f.startswith(s + '_'):
                    skel = s
                    break
        if skel is None:
            n_dropped_no_skel += 1
            continue

        # Too short to judge
        try:
            arr = np.load(motion_dir / f, mmap_mode='r')
            T = arr.shape[0]
        except Exception:
            n_dropped_no_skel += 1
            continue
        if T < MIN_FRAMES:
            n_dropped_short += 1
            continue

        # Read the action off the file name
        action = parse_action_from_filename(f)
        if is_other_label(action):
            n_dropped_other += 1
            continue
        cluster = action_to_cluster(action)

        index[skel][cluster].append({
            'fname': f,
            'T': int(T),
            'action': action,
            'cluster': cluster,
        })
        n_kept += 1

    print(f"\n=== CLIP INDEX SUMMARY ===")
    print(f"  Kept: {n_kept}")
    print(f"  Dropped (no skel): {n_dropped_no_skel}")
    print(f"  Dropped (T<{MIN_FRAMES}): {n_dropped_short}")
    print(f"  Dropped ('other' label): {n_dropped_other}")

    # How much each skeleton has
    skel_stats = {}
    skels_with_no_clips = []
    for skel in skel_names:
        n_clips = sum(len(v) for v in index.get(skel, {}).values())
        n_clusters = len(index.get(skel, {}))
        skel_stats[skel] = {'n_clips': n_clips, 'n_clusters': n_clusters,
                             'clusters': list(index.get(skel, {}).keys())}
        if n_clips == 0:
            skels_with_no_clips.append(skel)

    print(f"\n  Skeletons with ZERO clean clips: {len(skels_with_no_clips)}")
    for s in skels_with_no_clips:
        print(f"    - {s}")

    # How many different actions each skeleton covers
    print(f"\n  Skeletons by cluster count:")
    by_count = defaultdict(list)
    for s, st in skel_stats.items():
        by_count[st['n_clusters']].append(s)
    for c in sorted(by_count.keys(), reverse=True):
        print(f"    {c} clusters: {len(by_count[c])} skels")

    # How many skeletons have each action
    cluster_coverage = defaultdict(int)
    for s, clusters in index.items():
        for c, clips in clusters.items():
            cluster_coverage[c] += 1
    print(f"\n  Cluster coverage (# skels with >=1 clip in cluster):")
    for c, n in sorted(cluster_coverage.items(), key=lambda x: -x[1]):
        print(f"    {c}: {n} skels")


    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = {
        'index': {k: dict(v) for k, v in index.items()},
        'skel_stats': skel_stats,
        'cluster_coverage': dict(cluster_coverage),
        'meta': {
            'min_frames': MIN_FRAMES,
            'n_kept': n_kept,
            'n_dropped_other': n_dropped_other,
            'n_dropped_short': n_dropped_short,
            'n_dropped_no_skel': n_dropped_no_skel,
        },
    }
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
