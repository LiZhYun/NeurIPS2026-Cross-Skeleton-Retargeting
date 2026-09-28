"""Choose the actions and recordings this study works with, and write clips.json.

The BONES-SEED collection is far larger than the animal data the paper's main study uses,
and most of it is walking. Using it whole would answer an easier question than the one the
paper asks. So the study draws a small, deliberately varied sample that looks like the
animal data in shape: ninety actions, at most thirty per cent of them locomotion and the
rest spread over the collection's other packages by fixed quotas. Each action keeps four
recordings, or three when only three of a comfortable length exist. Within an action the
recordings are picked to come from different performers where possible.

The draw is fixed by a seed, so the same metadata table always gives the same clips.json.
Run on the collection's metadata table `seed_metadata_v004.parquet`, it rebuilds exactly
the selection the paper used: 90 actions and 355 clips. The list itself is not shipped,
because it copies parts of that metadata, which the collection's licence treats as
confidential.

Runs in the robot environment (robots/environment.yml), which has pandas and pyarrow for
reading the metadata table:
    python -m robots.human_to_g1.select_clips --metadata <seed_metadata_v004.parquet>
"""
import argparse
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

from robots.human_to_g1.data import DEFAULT_DATA, clip_list_path

SEED = 42
FPS_IN, FPS_OUT = 120, 30
DURATION_LOW, DURATION_HIGH = 300, 1200   # preferred recording length, frames at 120 fps
TARGET_ACTIONS = 90
CLIPS_PER_ACTION_TARGET = 4
CLIPS_PER_ACTION_MIN = 3
CLIPS_PER_ACTION_MAX = 5
CREDIT = ("Training data includes Motion Data by Bones Studio (https://bones.studio/). Use "
          "of the underlying dataset is subject to the BONES Motion Capture Dataset License "
          "Agreement.")

# How many actions to take from each package. Locomotion is capped at thirty per cent so a
# plain random draw cannot fill the sample with walking; the rest follow the size of each
# package's pool of eligible actions.
QUOTAS = OrderedDict([("Locomotion", 27), ("Interactions", 15), ("Dances", 11),
                      ("Communication", 9), ("Everyday", 9), ("Gaming", 8),
                      ("Sport", 6), ("Other", 5)])
assert sum(QUOTAS.values()) == TARGET_ACTIONS


def select_clips_for_action(frame, rng):
    """Pick a few recordings of one action, favouring different performers.

    Recordings are ranked by whether their length falls in the preferred band and then by
    how close they are to the middle of it, with a seeded tie-break. The first recording of
    each performer is taken in that order; if performers run out, the best remaining
    recordings fill the rest.
    """
    middle = 0.5 * (DURATION_LOW + DURATION_HIGH)
    df = frame.copy()
    df["in_band"] = ((df["move_duration_frames"] >= DURATION_LOW) &
                     (df["move_duration_frames"] <= DURATION_HIGH)).astype(int)
    df["dist_mid"] = (df["move_duration_frames"] - middle).abs()
    df["jit"] = rng.rand(len(df))
    df = df.sort_values(["in_band", "dist_mid", "jit"],
                        ascending=[False, True, True]).reset_index(drop=True)

    wanted = min(CLIPS_PER_ACTION_MAX, max(CLIPS_PER_ACTION_TARGET, CLIPS_PER_ACTION_MIN))
    wanted = min(wanted, len(df))
    picked, performers = [], set()
    for i, row in df.iterrows():
        if len(picked) >= wanted:
            break
        if row["take_actor"] not in performers:
            picked.append(i)
            performers.add(row["take_actor"])
    for i, row in df.iterrows():
        if len(picked) >= max(CLIPS_PER_ACTION_MIN, wanted):
            break
        if i not in picked:
            picked.append(i)
    return df.iloc[sorted(picked)].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True,
                    help="the collection's metadata table (parquet)")
    ap.add_argument("--data_dir", default=str(DEFAULT_DATA))
    ap.add_argument("--out", default=None, help="default <data_dir>/clips.json")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    table = pd.read_parquet(args.metadata)
    recordings = table[table["is_mirror"] == False].copy()  # noqa: E712

    packages = recordings.groupby("content_name")["package"].nunique()
    assert (packages > 1).sum() == 0, "an action name belongs to more than one package"

    recordings["in_band"] = ((recordings["move_duration_frames"] >= DURATION_LOW) &
                             (recordings["move_duration_frames"] <= DURATION_HIGH))
    in_band = recordings[recordings["in_band"]]
    sizes = in_band.groupby(["package", "content_name"]).size()
    eligible = sizes[sizes >= CLIPS_PER_ACTION_MIN].reset_index(name="n")

    actions = []
    for package, quota in QUOTAS.items():
        pool = sorted(eligible[eligible["package"] == package]["content_name"].tolist())
        if len(pool) < quota:
            raise RuntimeError(f"{package}: only {len(pool)} actions available, need {quota}")
        drawn = rng.choice(len(pool), size=quota, replace=False)
        for name in [pool[i] for i in sorted(drawn)]:
            chosen = select_clips_for_action(in_band[in_band["content_name"] == name], rng)
            clips = [OrderedDict([
                ("clip_id", str(r["filename"])),
                ("human_path", str(r["move_soma_uniform_path"])),
                ("robot_path", str(r["move_g1_path"])),
                ("duration_frames", int(r["move_duration_frames"])),
                ("performer", str(r["take_actor"])),
                ("recording_date", int(r["take_date"])),
            ]) for _, r in chosen.iterrows()]
            actions.append(OrderedDict([
                ("action", name), ("package", package),
                ("category", str(chosen.iloc[0]["category"])),
                ("n_clips", len(clips)), ("clips", clips)]))

    durations = [c["duration_frames"] for a in actions for c in a["clips"]]
    per_package = OrderedDict()
    for a in actions:
        entry = per_package.setdefault(a["package"], {"actions": 0, "clips": 0})
        entry["actions"] += 1
        entry["clips"] += a["n_clips"]
    counts = [a["n_clips"] for a in actions]
    all_distinct = sum(1 for a in actions
                       if len({c["performer"] for c in a["clips"]}) == a["n_clips"])

    summary = OrderedDict([
        ("n_actions", len(actions)),
        ("n_clips", len(durations)),
        ("actions_per_package", {k: v["actions"] for k, v in per_package.items()}),
        ("clips_per_package", {k: v["clips"] for k, v in per_package.items()}),
        ("clips_per_action", {"min": int(min(counts)), "max": int(max(counts)),
                              "mean": float(np.mean(counts))}),
        ("actions_with_all_distinct_performers", all_distinct),
        ("duration_frames_at_120fps", {
            "min": int(min(durations)), "max": int(max(durations)),
            "mean": float(np.mean(durations)), "median": float(np.median(durations)),
            "pct_in_band": float(np.mean([DURATION_LOW <= d <= DURATION_HIGH
                                          for d in durations]))}),
    ])
    meta = OrderedDict([
        ("source", "BONES-SEED"),
        ("credit", CREDIT),
        ("seed", args.seed), ("target_actions", TARGET_ACTIONS), ("quotas", dict(QUOTAS)),
        ("fps_in", FPS_IN), ("fps_out", FPS_OUT),
        ("downsample_stride", FPS_IN // FPS_OUT),
        ("duration_band_frames", [DURATION_LOW, DURATION_HIGH]),
        ("clips_per_action", [CLIPS_PER_ACTION_MIN, CLIPS_PER_ACTION_MAX]),
    ])
    out = Path(args.out) if args.out else clip_list_path(args.data_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(OrderedDict([("meta", meta), ("summary", summary),
                               ("actions", actions)]), f, indent=2)

    print(f"{summary['n_actions']} actions, {summary['n_clips']} clips -> {out}")
    print(f"{'package':16s} {'actions':>8s} {'clips':>6s}  (available)")
    for package in QUOTAS:
        available = int((eligible["package"] == package).sum())
        entry = per_package.get(package, {"actions": 0, "clips": 0})
        print(f"{package:16s} {entry['actions']:8d} {entry['clips']:6d}  ({available})")
    duration = summary["duration_frames_at_120fps"]
    print(f"clips per action: {summary['clips_per_action']['min']} to "
          f"{summary['clips_per_action']['max']}, "
          f"mean {summary['clips_per_action']['mean']:.2f}")
    print(f"actions where every clip has a different performer: "
          f"{all_distinct}/{summary['n_actions']}")
    print(f"length at 120 fps: median {duration['median']:.0f} frames, "
          f"{duration['pct_in_band'] * 100:.0f}% in the preferred band")
    print(f"locomotion share of actions: "
          f"{per_package['Locomotion']['actions'] / summary['n_actions'] * 100:.0f}%")


if __name__ == "__main__":
    main()
