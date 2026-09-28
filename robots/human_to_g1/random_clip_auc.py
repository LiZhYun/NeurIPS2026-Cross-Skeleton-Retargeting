"""Score the random same-action clip's action-level AUC fairly.

The random-clip row hands back some other real robot clip of the same action. When that
clip is also left in the set of references it is compared against, it finds an exact copy
of itself and the score is flattered. Removing it is the fair rule, and the number the
paper reports. Both numbers are produced here so the difference is visible.

The random draws are the ones the main scoring run made, so this number belongs to that
run rather than to a fresh draw.

Runs in the main environment, on the processor, in about five minutes:
    python -m robots.human_to_g1.random_clip_auc --out output/robots/g1_random_clip_auc.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

from robots.action_auc import rank_auc, resample_frames, trajectory_distance
from robots.human_to_g1.data import DEFAULT_DATA, action_groups, load_clip_list, load_tensor
from robots.human_to_g1.score_four_measures import references

DESCRIPTION = ("Action-level AUC of the random same-action clip on the Unitree G1. Leaving "
               "the clip being scored inside its own reference set flatters the score; "
               "\"own_clip_excluded\" removes it, and that is the number the paper reports.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=str(DEFAULT_DATA))
    ap.add_argument("--max_other_actions", type=int, default=100)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    groups = action_groups(load_clip_list(data_dir), data_dir, min_clips=2)
    action_of = {c: g["action"] for g in groups for c in g["clip_ids"]}
    stretched, _ = references(groups, data_dir)
    all_clips = list(stretched)

    draws = np.random.RandomState(42)
    drawn_for = {}

    def draw(clip_id, group):
        others = [c for c in group["clip_ids"] if c != clip_id] or group["clip_ids"]
        chosen = others[draws.randint(len(others))]
        drawn_for[clip_id] = chosen
        return load_tensor(data_dir, "g1", chosen)

    # The SIF pass of the scoring run draws one clip per query before the action pass does,
    # so it is repeated here to leave the draws where the scoring run left them.
    for group in groups:
        for clip_id in group["clip_ids"]:
            draw(clip_id, group)

    rng = np.random.RandomState(42)
    included, excluded = [], []
    for group in groups:
        for clip_id in group["clip_ids"]:
            motion = draw(clip_id, group)
            if motion is None:
                continue
            chosen = drawn_for[clip_id]
            other = [c for c in all_clips if action_of.get(c) != group["action"]]
            if len(other) > args.max_other_actions:
                other = [other[i] for i in
                         rng.choice(len(other), args.max_other_actions, replace=False)]
            same = [c for c in stretched
                    if action_of.get(c) == group["action"] and c != clip_id]
            produced = resample_frames(motion, 64)
            other_distances = [trajectory_distance(produced, stretched[c]) for c in other]
            included.append(rank_auc([trajectory_distance(produced, stretched[c])
                                      for c in same], other_distances))
            without_self = [c for c in same if c != chosen]
            if without_self:
                excluded.append(rank_auc([trajectory_distance(produced, stretched[c])
                                          for c in without_self], other_distances))

    result = {"description": DESCRIPTION,
              "own_clip_excluded": float(np.mean(excluded)),
              "own_clip_included": float(np.mean(included)),
              "n_queries": len(included)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"own clip excluded {result['own_clip_excluded']:.4f}, "
          f"included {result['own_clip_included']:.4f}, "
          f"{result['n_queries']} queries -> {args.out}")


if __name__ == "__main__":
    main()
