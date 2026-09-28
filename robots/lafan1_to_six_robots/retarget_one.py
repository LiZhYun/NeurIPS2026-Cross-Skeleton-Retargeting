"""Retarget one recording to one robot, in a process of its own.

This does exactly what `retarget.py` does for a single recording. It exists so that a
recording which crashes the solver takes only its own process down and the caller can see
what happened.

Runs in the robot environment (robots/environment.yml), with the retargeting tool on the
import path. retarget_retry.py calls it; it can also be run by hand:
    PYTHONPATH=external/GMR python -m robots.lafan1_to_six_robots.retarget_one \
        --robot unitree_g1 --source_file dance2_subject2.bvh --routine dance2 \
        --performer 2 --n_frames <frames> \
        --bvh_dir data/robots/lafan1_to_six_robots/lafan1_bvh \
        --out_dir data/robots/lafan1_to_six_robots/retargeted
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", required=True)
    ap.add_argument("--source_file", required=True)
    ap.add_argument("--routine", required=True, help="the routine's name, such as dance2")
    ap.add_argument("--performer", type=int, required=True)
    ap.add_argument("--n_frames", type=int, required=True)
    ap.add_argument("--bvh_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out = Path(args.out_dir) / args.robot / f"{args.routine}_s{args.performer}.npz"
    if out.exists():
        print(json.dumps({"status": "already done"}))
        return

    from general_motion_retargeting import GeneralMotionRetargeting
    from general_motion_retargeting.utils.lafan1 import load_bvh_file

    started = time.time()
    frames, human_height = load_bvh_file(str(Path(args.bvh_dir) / args.source_file),
                                         format="lafan1")
    assert len(frames) == args.n_frames, (len(frames), args.n_frames)
    retargeter = GeneralMotionRetargeting(src_human="bvh_lafan1", tgt_robot=args.robot,
                                          actual_human_height=human_height, verbose=False)
    pose = np.empty((len(frames), retargeter.configuration.model.nq), np.float32)
    for i in range(len(frames)):
        pose[i] = retargeter.retarget(frames[i])
        if i % 1000 == 0:
            print(f"frame {i}/{len(frames)}", file=sys.stderr, flush=True)
    if not np.isfinite(pose).all():
        print(json.dumps({"status": "failed", "error": "missing values in the result"}))
        sys.exit(3)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, qpos=pose, n_frames=len(frames), fps=30, robot=args.robot,
                        source_file=args.source_file, seconds=time.time() - started)
    print(json.dumps({"status": "done", "frames": len(frames),
                      "seconds": round(time.time() - started, 1)}))


if __name__ == "__main__":
    main()
