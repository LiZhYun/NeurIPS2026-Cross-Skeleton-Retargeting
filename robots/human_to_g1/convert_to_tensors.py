"""Turn the extracted clips into joint positions for both skeletons.

For every clip in clips.json this reads the human recording and the matching robot
recording, works out where each joint is in every frame, keeps twenty-nine points on the
human and thirty on the robot, and drops three frames out of every four so both sides run
at 30 frames per second. Both sides are thinned from the same starting frame, so frame
number 100 of a clip means the same instant on the human and on the robot. Everything
later in the study relies on that.

The step also writes a report of checks that need no pictures: that the two sides have the
same number of frames, that the human's bones keep a constant length, that feet sit near
the ground, and that walking clips travel a sensible distance.

The robot description must be fetched first (fetch_g1_model.sh). Runs in the robot
environment (robots/environment.yml), which has MuJoCo; needs no GPU and takes about
two minutes:
    python -m robots.human_to_g1.convert_to_tensors
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from robots.human_to_g1 import g1_positions, human_positions
from robots.human_to_g1.data import DEFAULT_DATA, load_clip_list


def bone_length_variation(positions, skeleton):
    """Largest relative wobble in a human bone's length over a clip.

    A rigid skeleton gives nearly zero. Joints that carry a position channel are skipped,
    because the distance to their parent is a movement rather than a bone.
    """
    moving = {i for i, channels in enumerate(skeleton.channels)
              if any(c.endswith("position") for c in channels)}
    worst = 0.0
    for j in range(skeleton.n_joints):
        parent = skeleton.parents[j]
        if parent < 0 or j in moving:
            continue
        length = np.linalg.norm(positions[:, j] - positions[:, parent], axis=1)
        if length.mean() > 1e-6:
            worst = max(worst, float(length.std() / length.mean()))
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=str(DEFAULT_DATA))
    ap.add_argument("--g1_model", default=str(g1_positions.DEFAULT_MODEL))
    ap.add_argument("--max_clips", type=int, default=0,
                    help="convert only the first few clips")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    extracted = data_dir / "extracted"
    tensors = data_dir / "tensors"
    clip_list = load_clip_list(data_dir)
    stride = clip_list["meta"]["downsample_stride"]
    clips = [(a, c) for a in clip_list["actions"] for c in a["clips"]]
    if args.max_clips:
        clips = clips[:args.max_clips]

    (tensors / "human").mkdir(parents=True, exist_ok=True)
    (tensors / "g1").mkdir(parents=True, exist_ok=True)

    converted, failures = [], []
    human_names = robot_names = body_idx = None
    started = time.time()
    for i, (action, clip) in enumerate(clips):
        clip_id = clip["clip_id"]
        human_file = extracted / clip["human_path"]
        robot_file = extracted / clip["robot_path"]
        try:
            if not human_file.exists():
                raise FileNotFoundError(f"missing human recording {human_file}")
            if not robot_file.exists():
                raise FileNotFoundError(f"missing robot recording {robot_file}")
            human_all, skeleton, _ = human_positions.load_positions_m(str(human_file))
            if body_idx is None:
                body_idx = skeleton.body_joint_indices
                human_names = [skeleton.names[k] for k in body_idx]
            robot_all, names = g1_positions.load_positions_m(str(robot_file), args.g1_model)
            if robot_names is None:
                robot_names = names

            n_human, n_robot = human_all.shape[0], robot_all.shape[0]
            same_length = (n_human == n_robot == clip["duration_frames"])
            wobble = bone_length_variation(human_all, skeleton)

            human = human_all[:, body_idx][::stride]
            robot = robot_all[::stride]
            frames = min(human.shape[0], robot.shape[0])
            human, robot = human[:frames], robot[:frames]

            np.save(tensors / "human" / f"{clip_id}.npy", human.astype(np.float32))
            np.save(tensors / "g1" / f"{clip_id}.npy", robot.astype(np.float32))

            toe = min(human[:, human_names.index("LeftToeEnd"), 1].min(),
                      human[:, human_names.index("RightToeEnd"), 1].min())
            ankle = min(robot[:, robot_names.index("left_ankle_roll_link"), 2].min(),
                        robot[:, robot_names.index("right_ankle_roll_link"), 2].min())
            hips = human[:, human_names.index("Hips")]
            pelvis = robot[:, robot_names.index("pelvis")]

            converted.append({
                "clip_id": clip_id, "action": action["action"],
                "package": action["package"], "frames_out": int(frames),
                "frames_before_thinning": [int(n_human), int(n_robot)],
                "same_length": bool(same_length),
                "human_bone_length_variation": wobble,
                "human_lowest_toe_m": float(toe),
                "robot_lowest_ankle_m": float(ankle),
                "robot_mean_pelvis_height_m": float(pelvis[:, 2].mean()),
                "human_travel_m": float(np.linalg.norm(hips[-1, [0, 2]] - hips[0, [0, 2]])),
                "robot_travel_m": float(np.linalg.norm(pelvis[-1, :2] - pelvis[0, :2])),
            })
        except Exception as error:  # noqa: BLE001
            failures.append({"clip_id": clip_id, "error": repr(error)})
        if (i + 1) % 25 == 0:
            print(f"  [{i + 1}/{len(clips)}] converted {len(converted)}, "
                  f"failed {len(failures)} ({time.time() - started:.0f}s)")

    if human_names is not None:
        with open(tensors / "joint_names.json", "w") as f:
            json.dump({"human": human_names, "g1": robot_names}, f, indent=2)

    def spread(key):
        values = [c[key] for c in converted]
        if not values:
            return None
        return {"min": float(np.min(values)), "median": float(np.median(values)),
                "max": float(np.max(values)), "mean": float(np.mean(values))}

    walking = [c for c in converted if "walk" in c["action"].lower()]
    report = {
        "n_clips": len(clips), "n_converted": len(converted), "n_failed": len(failures),
        "downsample_stride": stride,
        "human_joints": len(human_names) if human_names else 0,
        "robot_points": len(robot_names) if robot_names else 0,
        "all_same_length": all(c["same_length"] for c in converted),
        "n_length_mismatch": sum(1 for c in converted if not c["same_length"]),
        "overall": {key: spread(key) for key in
                    ("human_bone_length_variation", "human_lowest_toe_m",
                     "robot_lowest_ankle_m", "robot_mean_pelvis_height_m", "frames_out")},
        "walking_clips_travel": [
            {"clip_id": c["clip_id"], "action": c["action"],
             "human_travel_m": c["human_travel_m"], "robot_travel_m": c["robot_travel_m"]}
            for c in walking],
        "failures": failures,
        "clips": converted,
    }
    out = data_dir / "conversion_report.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nconverted {len(converted)} of {len(clips)} clips, {len(failures)} failed")
    print(f"  human joints {report['human_joints']}, robot points {report['robot_points']}")
    print(f"  both sides same length: {report['all_same_length']} "
          f"({report['n_length_mismatch']} mismatched)")
    if converted:
        overall = report["overall"]
        print(f"  human bone length wobble, worst: "
              f"{overall['human_bone_length_variation']['max']:.2e} (a rigid skeleton gives 0)")
        print(f"  lowest toe, median: {overall['human_lowest_toe_m']['median']:.3f} m")
        print(f"  lowest ankle, median: {overall['robot_lowest_ankle_m']['median']:.3f} m")
        print(f"  pelvis height, median: "
              f"{overall['robot_mean_pelvis_height_m']['median']:.3f} m")
    if walking:
        print(f"  {len(walking)} walking clips, first travels "
              f"{walking[0]['human_travel_m']:.2f} m as a human and "
              f"{walking[0]['robot_travel_m']:.2f} m as a robot")
    if failures:
        print(f"  failed: {[f['clip_id'] for f in failures[:5]]}")
    print(f"  report written to {out}")
    if not converted:
        first = failures[0]["error"] if failures else "clips.json lists no clips"
        sys.exit(f"no clip could be converted; the first error was:\n  {first}\n"
                 f"Check that the clips were extracted into {extracted} and that the robot "
                 f"description exists at {args.g1_model}.")


if __name__ == "__main__":
    main()
