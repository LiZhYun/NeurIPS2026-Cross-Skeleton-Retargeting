"""Action-level AUC and realism for the six robots.

Source-Instance Fidelity (SIF) and variation come from `score.py`. This step adds the other
two measures, defined exactly as in the human-to-G1 study: whether real robot clips of the
same action rank closer to a produced motion than clips of other actions, and whether the
produced poses look like real robot poses. The cut-off for "real enough" is set for each
robot from real clips held out of its own pose bank. Here "action" is the routine's kind
(dance, fight, aiming and so on), so two numbered routines of one kind count as the same
action.

Runs in the main environment, on the processor. It uses a dozen processes by default and
then takes about ten minutes:
    python -m robots.lafan1_to_six_robots.score_four_measures \
        --out output/robots/six_robots_four_measures.json
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from robots.action_auc import (distance_to_nearest, pose_shape, rank_auc, resample_frames,
                               trajectory_distance)
from robots.lafan1_to_six_robots.data import DEFAULT_DATA, SETTINGS, load_clip_list, load_tensor
from robots.lafan1_to_six_robots.methods import METHODS, method_reader

HELD_OUT_SHARE, MAX_POSES, MAX_OTHER = 0.2, 4000, 100
DESCRIPTION = ("Whether the action is still recognisable, and whether the motion looks like "
               "real robot motion, for the six robots, one entry per robot and setting.")


def run_one(job):
    """Both measures for every model on one robot in one setting."""
    robot, setting, data_dir, generated = job
    clip_list = load_clip_list(data_dir)
    setting_clips = clip_list["settings"][setting]
    action_of_group = {g["name"]: g["action"] for g in clip_list["action_groups"]}
    group_of_clip = {c: group for group, clips in setting_clips.items() for c in clips}
    clip_ids = sorted(c for c in group_of_clip
                      if load_tensor(data_dir, robot, c) is not None)
    action_of = {c: action_of_group[group_of_clip[c]] for c in clip_ids}

    rng = np.random.RandomState(0)
    order = rng.permutation(len(clip_ids))
    n_held_out = max(5, int(round(HELD_OUT_SHARE * len(clip_ids))))
    if len(clip_ids) <= n_held_out:
        raise SystemExit(f"{robot} [{setting}]: the realism measure needs more than "
                         f"{n_held_out} robot clips ({n_held_out} held out to set the "
                         f"cut-off, the rest as the bank of real poses), but only "
                         f"{len(clip_ids)} were found")
    held_out = [clip_ids[i] for i in order[:n_held_out]]
    bank = np.concatenate([pose_shape(load_tensor(data_dir, robot, clip_ids[i]))
                           for i in order[n_held_out:]])
    if len(bank) > MAX_POSES:
        bank = bank[rng.choice(len(bank), MAX_POSES, replace=False)]
    threshold = float(np.percentile(
        [distance_to_nearest(pose_shape(load_tensor(data_dir, robot, c)), bank)
         for c in held_out], 95))

    stretched = {c: resample_frames(load_tensor(data_dir, robot, c), 64) for c in clip_ids}
    result = {"robot": robot, "setting": setting, "n_clips": len(clip_ids),
              "realistic_threshold": threshold, "rows": {}}
    for model in METHODS:
        reader = method_reader(model, data_dir, robot, setting, generated)
        rng_other = np.random.RandomState(42)
        scores, realistic = [], []
        for clip_id in clip_ids:
            motion = reader(clip_id, setting_clips[group_of_clip[clip_id]])
            if motion is None:
                continue
            realistic.append(distance_to_nearest(pose_shape(motion), bank) <= threshold)
            same = [c for c in clip_ids
                    if c != clip_id and action_of[c] == action_of[clip_id]]
            other = [c for c in clip_ids if action_of[c] != action_of[clip_id]]
            if len(other) > MAX_OTHER:
                other = [other[i] for i in
                         rng_other.choice(len(other), MAX_OTHER, replace=False)]
            if same and other:
                produced = resample_frames(motion, 64)
                scores.append(rank_auc(
                    [trajectory_distance(produced, stretched[c]) for c in same],
                    [trajectory_distance(produced, stretched[c]) for c in other]))
        result["rows"][model] = {"n_outputs": len(realistic),
                                 "realistic_pct": float(100 * np.mean(realistic)),
                                 "action_auc_mean": float(np.mean(scores)),
                                 "n_auc": len(scores)}
        print(f"{robot:18s} {setting:6s} {model:20s} action AUC={np.mean(scores):.3f} "
              f"realistic={100 * np.mean(realistic):.1f}%", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=str(DEFAULT_DATA))
    ap.add_argument("--generated_dir", default=None)
    ap.add_argument("--processes", type=int, default=12)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    generated = Path(args.generated_dir) if args.generated_dir else data_dir / "generated"
    robots = list(load_clip_list(data_dir)["meta"]["robots"])
    jobs = [(r, s, data_dir, generated) for r in robots for s in SETTINGS]
    with Pool(args.processes) as pool:
        results = pool.map(run_one, jobs)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"description": DESCRIPTION, "results": results}, f, indent=2)
    print("written to", args.out)


if __name__ == "__main__":
    main()
