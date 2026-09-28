"""Score the random same-group clip's action-level AUC fairly, for every robot.

The random-clip row hands back the retarget of some other clip of the same action group.
If that clip is also left among the references it is compared against, it matches itself
exactly and the score is flattered. Removing it gives the number the paper reports.

Runs in the main environment, on the processor, in about ten minutes:
    python -m robots.lafan1_to_six_robots.random_clip_auc \
        --out output/robots/six_robots_random_clip_auc.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

from robots.action_auc import rank_auc, resample_frames, trajectory_distance
from robots.lafan1_to_six_robots.data import DEFAULT_DATA, SETTINGS, load_clip_list, load_tensor

DESCRIPTION = ("Action-level AUC of the random same-group clip for the six robots, with "
               "the clip being scored removed from its own reference set. Keys read "
               "robot|setting.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=str(DEFAULT_DATA))
    ap.add_argument("--max_other_actions", type=int, default=100)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    clip_list = load_clip_list(data_dir)
    action_of_group = {g["name"]: g["action"] for g in clip_list["action_groups"]}

    scores = {}
    for robot in clip_list["meta"]["robots"]:
        for setting in SETTINGS:
            setting_clips = clip_list["settings"][setting]
            group_of_clip = {c: group for group, clips in setting_clips.items() for c in clips}
            clip_ids = sorted(c for c in group_of_clip
                              if load_tensor(data_dir, robot, c) is not None)
            action_of = {c: action_of_group[group_of_clip[c]] for c in clip_ids}
            stretched = {c: resample_frames(load_tensor(data_dir, robot, c), 64)
                         for c in clip_ids}

            draws = np.random.RandomState(42)
            rng_other = np.random.RandomState(42)
            values = []
            for clip_id in clip_ids:
                siblings = setting_clips[group_of_clip[clip_id]]
                others = [x for x in siblings if x != clip_id] or siblings
                chosen = others[draws.randint(len(others))]
                same = [x for x in clip_ids if x != clip_id and x != chosen
                        and action_of[x] == action_of[clip_id]]
                other = [x for x in clip_ids if action_of[x] != action_of[clip_id]]
                if len(other) > args.max_other_actions:
                    other = [other[i] for i in rng_other.choice(
                        len(other), args.max_other_actions, replace=False)]
                if same and other:
                    produced = resample_frames(load_tensor(data_dir, robot, chosen), 64)
                    values.append(rank_auc(
                        [trajectory_distance(produced, stretched[x]) for x in same],
                        [trajectory_distance(produced, stretched[x]) for x in other]))
            scores[f"{robot}|{setting}"] = float(np.mean(values))
            print(f"{robot:18s} {setting:6s} own clip excluded: "
                  f"{scores[f'{robot}|{setting}']:.3f}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"description": DESCRIPTION, "own_clip_excluded": scores}, f, indent=2)
    print("written to", args.out)


if __name__ == "__main__":
    main()
