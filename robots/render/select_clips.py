"""Measure which clips the figure should show, and record the measurements.

The figure has two halves. The first shows one human motion on all six robots, so it wants
the most expressive clip that exists for every robot: a lot of limb movement and a lot of
change in posture. The second shows two people performing the same routine on one robot,
so it wants a pair whose true retargets are far apart while the averaging model's outputs
for them are close, since that is the difference the models are supposed to preserve.

Nothing is drawn here. The measurements are written to `reports/selection.json` in the
output folder, so the choice can be checked; the clips the figure shows (dance2, performers
1 and 2) are fixed in the two drawing steps. The record needs the six-robot study's clip
list, joint positions and trained models' motions.

Runs in either environment, in a few seconds:
    python -m robots.render.select_clips
"""
import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from robots.lafan1_to_six_robots.data import DEFAULT_DATA, HORIZON, ROBOTS, load_clip_list
from robots.lafan1_to_six_robots.methods import METHODS
from robots.render.paths import DEFAULT_OUT

MODELS = [m for m in METHODS if m not in ("true_retarget", "random_clip")]


def root_center(motion):
    """Move each frame so the root joint sits at the origin, as the models were trained."""
    return motion - motion[:, 0:1, :]


def load_produced(data_dir, robot, setting, model, clip_id):
    """A trained model's motion for one clip, or None if it is missing."""
    path = Path(data_dir) / "generated" / robot / setting / model / f"{clip_id}.npy"
    return np.load(path).astype(np.float64) if path.exists() else None


def expressiveness(motion):
    """How much a motion moves, and how much its posture changes."""
    centred = root_center(motion)
    speed = np.linalg.norm(np.diff(centred, axis=0), axis=-1).mean() * 30.0
    posture = centred.reshape(len(centred), -1).std(0).mean()
    highest = centred[:, :, 2].max(1)
    return {"limb_speed": float(speed), "postural_std": float(posture),
            "reach_z_range": float(highest.max() - highest.min()),
            "travel": float(np.linalg.norm(motion[-1, 0, :2] - motion[0, 0, :2]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=str(DEFAULT_DATA))
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    tensors = data_dir / "tensors"
    reports = Path(args.out_dir) / "reports"
    clip_list = load_clip_list(data_dir)
    groups = {g["name"]: g for g in clip_list["action_groups"]}
    dense = clip_list["settings"]["dense"]

    expressive = []
    for group in groups.values():
        if group["action"] not in ("dance", "fight", "fightAndSports"):
            continue
        for clip in group["clips"]:
            clip_id = clip["clip_id"]
            on_all = all((tensors / r / f"{clip_id}.npy").exists() for r in ROBOTS)
            human_file = tensors / "human" / f"{clip_id}.npy"
            if not (on_all and human_file.exists()):
                continue
            robot = np.load(tensors / "unitree_g1" / f"{clip_id}.npy").astype(np.float64)
            entry = expressiveness(robot)
            entry["human"] = expressiveness(np.load(human_file).astype(np.float64))
            entry.update(clip_id=clip_id, action=group["action"],
                         performer=clip["performer"], start_frame=clip["start_frame"])
            entry["score"] = entry["limb_speed"] * entry["postural_std"]
            expressive.append(entry)
    expressive.sort(key=lambda e: -e["score"])

    pairs = []
    for name, clip_ids in dense.items():
        by_window = {}
        for clip_id in clip_ids:
            by_window.setdefault(clip_id.rsplit("_", 1)[1], []).append(clip_id)
        for window, same_window in by_window.items():
            for a, b in combinations(sorted(same_window), 2):
                file_a = tensors / "unitree_g1" / f"{a}.npy"
                file_b = tensors / "unitree_g1" / f"{b}.npy"
                if not (file_a.exists() and file_b.exists()):
                    continue
                motion_a = root_center(np.load(file_a).astype(np.float64))[:HORIZON]
                motion_b = root_center(np.load(file_b).astype(np.float64))[:HORIZON]
                row = {"action_group": name, "window": window, "a": a, "b": b,
                       "action": groups[name]["action"],
                       "gap_true_retarget": float(
                           np.linalg.norm(motion_a - motion_b, axis=-1).mean())}
                for model in MODELS:
                    produced_a = load_produced(data_dir, "unitree_g1", "dense", model, a)
                    produced_b = load_produced(data_dir, "unitree_g1", "dense", model, b)
                    row[f"gap_{model}"] = (
                        float(np.linalg.norm(produced_a - produced_b, axis=-1).mean())
                        if produced_a is not None and produced_b is not None else None)
                row["motion_a"] = float(
                    np.linalg.norm(np.diff(motion_a, axis=0), axis=-1).mean() * 30)
                row["motion_b"] = float(
                    np.linalg.norm(np.diff(motion_b, axis=0), axis=-1).mean() * 30)
                if row["gap_averaging_objective"]:
                    row["contrast"] = (row["gap_true_retarget"]
                                       / max(row["gap_averaging_objective"], 1e-6))
                    row["contrast_unpaired"] = (row["gap_true_retarget"]
                                                / max(row["gap_unpaired_objective"], 1e-6))
                pairs.append(row)
    pairs.sort(key=lambda r: -(r.get("contrast") or 0))

    reports.mkdir(parents=True, exist_ok=True)
    (reports / "selection.json").write_text(json.dumps({
        "expressive_clips": expressive[:15],
        "pairs_by_contrast": pairs[:25],
        "pairs_by_gap": sorted(pairs, key=lambda r: -r["gap_true_retarget"])[:15],
        "n_pairs": len(pairs)}, indent=2))

    print("most expressive clips present on every robot")
    for entry in expressive[:10]:
        print(f"{entry['clip_id']:24s} {entry['action']:14s} score={entry['score']:.5f} "
              f"speed={entry['limb_speed']:.3f} posture={entry['postural_std']:.3f} "
              f"travel={entry['travel']:.2f}")
    print("\nperformer pairs, furthest apart relative to the averaging model")
    print(f"{'routine':18s} {'a':22s} {'b':22s} {'true':>7s} {'unpaired':>9s} "
          f"{'averaging':>10s} {'truepair':>9s} {'ratio':>6s}")
    for row in pairs[:18]:
        print(f"{row['action_group']:18s} {row['a']:22s} {row['b']:22s} "
              f"{row['gap_true_retarget']:7.4f} {row['gap_unpaired_objective']:9.4f} "
              f"{row['gap_averaging_objective']:10.4f} {row['gap_true_pair_model']:9.4f} "
              f"{row['contrast']:6.2f}")
    print(f"\nwritten to {reports / 'selection.json'}")


if __name__ == "__main__":
    main()
