"""Score the human-to-G1 study on all four measures the paper reports.

  SIF               do the robot motions differ the way their human sources do?
                    (Source-Instance Fidelity)
  variation         how far apart are the robot motions, relative to their sources?
  action-level AUC  is the action still recognisable in the robot motion?
  realism           does the robot motion look like real robot motion at all?

The first two come from the same comparison the animal study uses. The third asks, for
each produced motion, whether real robot clips of the same action come out closer than
clips of other actions. The fourth measures how far each produced pose is from the nearest
real robot pose; the cut-off for "real enough" is set from real clips held out of that
comparison, before any produced motion is measured.

Runs in the main environment, on the processor. Each length setting takes about twenty
minutes, nearly all of it in the action measure:
    python -m robots.human_to_g1.score_four_measures --length raw \
        --out output/robots/g1_four_measures_raw.json
    python -m robots.human_to_g1.score_four_measures --length length_controlled \
        --out output/robots/g1_four_measures_length_controlled.json
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from robots.action_auc import (distance_to_nearest, grouped_interval, joints_in_bank,
                               pose_shape, rank_auc, resample_frames, trajectory_distance)
from robots.human_to_g1.data import DEFAULT_DATA, action_groups, load_clip_list, load_tensor
from robots.human_to_g1.methods import METHODS, method_reader
from robots.human_to_g1.score import score_rows
from robots.scoring import summarize

DESCRIPTIONS = {
    "raw": ("Human motion capture retargeted to the Unitree G1 humanoid, scored on four "
            "measures: Source-Instance Fidelity (SIF), how far apart the outputs "
            "are, whether the action is still recognisable, and whether the motion looks "
            "like real robot motion. Rows are the true retarget, a random clip of the same "
            "action, and four trained models. Scored by comparing motions over the frames "
            "they share."),
    "length_controlled": (
        "Human motion capture retargeted to the Unitree G1 humanoid, scored on four "
        "measures: Source-Instance Fidelity (SIF), how far apart the outputs are, "
        "whether the action is still recognisable, and whether the motion looks like real "
        "robot motion. Rows are the true retarget, a random clip of the same action, and "
        "four trained models. Scored by comparing motions after stretching them to the same "
        "length."),
}


def realism_bank(groups, data_dir, n_held_out=20, max_poses=4000, seed=0):
    """A bank of real robot poses and the distance that counts as realistic.

    Some real clips are kept out of the bank; the cut-off is the distance that 95 per cent
    of those held-out clips stay below, so it is fixed on real motion alone. Returns the
    bank, the cut-off, the held-out clips' median distance and how many clips were held out.
    """
    clips = []
    for group in groups:
        for clip_id in group["clip_ids"]:
            motion = load_tensor(data_dir, "g1", clip_id)
            if motion is not None:
                clips.append(pose_shape(motion))
    if len(clips) <= n_held_out:
        raise SystemExit(f"the realism measure needs at least {n_held_out + 1} converted robot "
                         f"clips ({n_held_out} held out to set the cut-off, the rest as the "
                         f"bank of real poses), but only {len(clips)} were found")
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(clips))
    held_out = [clips[i] for i in order[:n_held_out]]
    bank = np.concatenate([clips[i] for i in order[n_held_out:]])
    if len(bank) > max_poses:
        bank = bank[rng.choice(len(bank), max_poses, replace=False)]
    distances = [distance_to_nearest(clip, bank) for clip in held_out]
    return (bank, float(np.percentile(distances, 95)), float(np.median(distances)),
            len(distances))


def references(groups, data_dir):
    """Every real robot clip stretched to 64 frames, and which action it belongs to."""
    stretched, by_action = {}, defaultdict(list)
    for group in groups:
        for clip_id in group["clip_ids"]:
            motion = load_tensor(data_dir, "g1", clip_id)
            if motion is not None:
                stretched[clip_id] = resample_frames(motion, 64)
                by_action[group["action"]].append(clip_id)
    return stretched, by_action


def action_score(motion, action, clip_id, stretched, by_action, action_of, rng, cap):
    """Whether real clips of this action rank closer to the motion than other actions do."""
    produced = resample_frames(motion, 64)
    same = [c for c in by_action[action] if c != clip_id]
    other = [c for c in stretched if action_of.get(c) != action]
    if len(other) > cap:
        other = [other[i] for i in rng.choice(len(other), cap, replace=False)]
    if not same or not other:
        return None
    return rank_auc([trajectory_distance(produced, stretched[c]) for c in same],
                    [trajectory_distance(produced, stretched[c]) for c in other])


def realism_and_action(reader, groups, bank, threshold, stretched, by_action, action_of,
                       cap=100, seed=42):
    """Realism and action-level AUC for one method, over every clip.

    The action-level AUC is averaged over clips, with a 95% interval from drawing whole
    action groups again with repeats.
    """
    rng = np.random.RandomState(seed)
    scores, distances = [], []
    for group in groups:
        for clip_id in group["clip_ids"]:
            motion = reader(clip_id, group)
            if motion is None:
                continue
            if motion.shape[1] == joints_in_bank(bank):
                distances.append(distance_to_nearest(pose_shape(motion), bank))
            score = action_score(motion, group["action"], clip_id, stretched, by_action,
                                 action_of, rng, cap)
            if score is not None:
                scores.append((score, group["action"]))
    if distances:
        values = np.array(distances)
        realistic = float((values <= threshold).mean() * 100)
        mean_distance = float(values.mean())
    else:
        realistic = mean_distance = None
    low, mean, high = grouped_interval(scores)
    return {"realistic_pct": realistic, "mean_distance_to_nearest_real_clip": mean_distance,
            "realistic_threshold": threshold, "action_auc_mean": mean,
            "action_auc_ci": [low, high], "n_auc": len(scores)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--length", default="raw", choices=["raw", "length_controlled"])
    ap.add_argument("--data_dir", default=str(DEFAULT_DATA))
    ap.add_argument("--generated_dir", default=None)
    ap.add_argument("--min_clips", type=int, default=2)
    ap.add_argument("--max_other_actions", type=int, default=100,
                    help="how many clips of other actions each comparison may use")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    generated = Path(args.generated_dir) if args.generated_dir else data_dir / "generated"
    groups = action_groups(load_clip_list(data_dir), data_dir, min_clips=args.min_clips)
    action_of = {c: g["action"] for g in groups for c in g["clip_ids"]}
    print(f"{len(groups)} action groups, comparing motions {args.length.replace('_', ' ')}")

    bank, threshold, held_out_median, n_held_out = realism_bank(groups, data_dir)
    print(f"realistic below {threshold:.4f} (held-out real clips: median "
          f"{held_out_median:.4f}, {n_held_out} clips)")
    stretched, by_action = references(groups, data_dir)

    rows = []
    for method in [m.strip() for m in args.methods.split(",")]:
        reader = method_reader(method, data_dir, generated)
        sif_row = summarize(method, score_rows(groups, data_dir, reader, args.length))
        extra = realism_and_action(reader, groups, bank, threshold, stretched, by_action,
                                   action_of, args.max_other_actions)
        rows.append({**sif_row, **extra})
        print(f"  {method:20s} SIF={sif_row.get('sif', float('nan')):+.3f} "
              f"variation={sif_row.get('variation', float('nan')):.3f} "
              f"p={sif_row.get('p_value', float('nan')):.4f} | "
              f"action AUC={extra['action_auc_mean']:.3f}"
              f"[{extra['action_auc_ci'][0]:.3f},{extra['action_auc_ci'][1]:.3f}] | "
              f"realistic={extra['realistic_pct']}%")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"description": DESCRIPTIONS[args.length], "n_action_groups": len(groups),
                   "realistic_threshold": threshold, "rows": rows}, f, indent=2)
    print(f"  written to {args.out}")


if __name__ == "__main__":
    main()
