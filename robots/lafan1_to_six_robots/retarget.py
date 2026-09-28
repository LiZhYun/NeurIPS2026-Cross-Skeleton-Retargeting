"""Retarget every recording to one robot.

Retargeting is what turns a human performance into a robot performance: it solves, frame by
frame, for the robot joint angles that put the robot's body where the person's body was,
as closely as the robot's build allows. This is the study's ground truth, the thing a model
would have to reproduce.

One robot per run, several recordings at a time. Results already on disk are left alone, so
an interrupted run can simply be started again.

Runs in the robot environment (robots/environment.yml), on the processor, with the
retargeting tool on the import path. About forty seconds of processor time per recording:
    PYTHONPATH=external/GMR python -m robots.lafan1_to_six_robots.retarget \
        --robot unitree_g1 --takes data/robots/lafan1_to_six_robots/takes.json \
        --bvh_dir data/robots/lafan1_to_six_robots/lafan1_bvh \
        --out_dir data/robots/lafan1_to_six_robots/retargeted
"""
import argparse
import json
import time
import traceback
from multiprocessing import Pool, set_start_method
from pathlib import Path

import numpy as np


def retarget_take(job):
    """Retarget one recording to one robot and save the joint angles of every frame."""
    robot, take, bvh_dir, out_dir = job
    out = Path(out_dir) / robot / f"{take['routine']}_s{take['performer']}.npz"
    if out.exists():
        try:
            if np.load(out)["qpos"].shape[0] == take["n_frames"]:
                return {"take": take["source_file"], "status": "already done"}
        except Exception:  # noqa: BLE001  unreadable file, so do it again
            pass
    try:
        from general_motion_retargeting import GeneralMotionRetargeting
        from general_motion_retargeting.utils.lafan1 import load_bvh_file

        started = time.time()
        frames, human_height = load_bvh_file(str(Path(bvh_dir) / take["source_file"]),
                                             format="lafan1")
        if len(frames) != take["n_frames"]:
            return {"take": take["source_file"], "status": "failed",
                    "error": f"{len(frames)} frames in the file, "
                             f"{take['n_frames']} expected"}
        retargeter = GeneralMotionRetargeting(src_human="bvh_lafan1", tgt_robot=robot,
                                              actual_human_height=human_height,
                                              verbose=False)
        pose = np.empty((len(frames), retargeter.configuration.model.nq), np.float32)
        for i in range(len(frames)):
            pose[i] = retargeter.retarget(frames[i])
        if not np.isfinite(pose).all():
            return {"take": take["source_file"], "status": "failed",
                    "error": "missing values in the result"}
        seconds = time.time() - started
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, qpos=pose, n_frames=len(frames), fps=30, robot=robot,
                            source_file=take["source_file"], seconds=seconds)
        return {"take": take["source_file"], "status": "done", "frames": len(frames),
                "seconds": round(seconds, 1)}
    except Exception:  # noqa: BLE001
        return {"take": take["source_file"], "status": "failed",
                "error": traceback.format_exc()[-800:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", required=True)
    ap.add_argument("--takes", required=True, help="takes.json written by the choose_windows step")
    ap.add_argument("--bvh_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--processes", type=int, default=16)
    args = ap.parse_args()

    takes = json.loads(Path(args.takes).read_text())["takes"]
    started = time.time()
    jobs = [(args.robot, t, args.bvh_dir, args.out_dir) for t in takes]
    with Pool(args.processes) as pool:
        results = pool.map(retarget_take, jobs, chunksize=1)
    elapsed = time.time() - started

    done = [r for r in results if r["status"] == "done"]
    skipped = [r for r in results if r["status"] == "already done"]
    failed = [r for r in results if r["status"] == "failed"]
    summary = {"robot": args.robot, "n_takes": len(takes), "done": len(done),
               "skipped": len(skipped), "failed": len(failed),
               "elapsed_seconds": round(elapsed, 1),
               "processor_seconds": round(sum(r.get("seconds", 0) for r in done), 1),
               "failures": failed}
    out = Path(args.out_dir) / args.robot / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "failures"}))
    for f in failed:
        print("failed:", f["take"], "|",
              f["error"].splitlines()[-1] if f.get("error") else "unknown")


if __name__ == "__main__":
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass
    main()
