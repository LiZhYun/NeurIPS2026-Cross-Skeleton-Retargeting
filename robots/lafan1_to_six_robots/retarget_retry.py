"""Redo any recording that is still missing, one process at a time.

Running many retargetings in one pool is fast, but a solver crash can take a worker down
and silently lose its share of the work. This step finds what is missing and runs each one
in its own process, so a crash is visible and costs only that recording. Each is tried
twice before being given up on, and the per-robot summary is rewritten from the files that
are actually on disk.

Runs in the robot environment (robots/environment.yml), on the processor, with the
retargeting tool on the import path:
    PYTHONPATH=external/GMR python -m robots.lafan1_to_six_robots.retarget_retry
"""
import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from robots.lafan1_to_six_robots.data import DEFAULT_DATA, ROBOTS


def missing_takes(takes, robots, out_dir):
    """Every recording and robot whose retargeted result is not on disk."""
    return [{"robot": robot, **take} for robot in robots for take in takes
            if not (Path(out_dir) / robot /
                    f"{take['routine']}_s{take['performer']}.npz").exists()]


def run_one(task, bvh_dir, out_dir, timeout=1200):
    """Retarget one recording in its own process, trying twice; True if it succeeded."""
    command = [sys.executable, "-m", "robots.lafan1_to_six_robots.retarget_one",
               "--robot", task["robot"], "--source_file", task["source_file"],
               "--routine", task["routine"], "--performer", str(task["performer"]),
               "--n_frames", str(task["n_frames"]), "--bvh_dir", str(bvh_dir),
               "--out_dir", str(out_dir)]
    label = f"{task['robot']} {task['routine']}_s{task['performer']}"
    for attempt in (1, 2):
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            print(f"[timed out, try {attempt}] {label}", flush=True)
            continue
        if result.returncode == 0:
            print(f"[done, try {attempt}] {label} "
                  f"{result.stdout.strip().splitlines()[-1]}", flush=True)
            return True
        print(f"[failed, try {attempt}] {label} exit {result.returncode}\n"
              f"  {result.stderr[-400:]}", flush=True)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=str(DEFAULT_DATA))
    ap.add_argument("--bvh_dir", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--robots", default=",".join(ROBOTS))
    ap.add_argument("--parallel", type=int, default=4)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    bvh_dir = Path(args.bvh_dir) if args.bvh_dir else data_dir / "lafan1_bvh"
    out_dir = Path(args.out_dir) if args.out_dir else data_dir / "retargeted"
    robots = args.robots.split(",")
    takes = json.loads((data_dir / "takes.json").read_text())["takes"]

    tasks = missing_takes(takes, robots, out_dir)
    print(f"{len(tasks)} recordings still missing")
    started = time.time()
    with ThreadPoolExecutor(args.parallel) as pool:
        results = list(pool.map(lambda t: run_one(t, bvh_dir, out_dir), tasks))
    print(f"{sum(results)}/{len(results)} recovered in {time.time() - started:.0f}s")

    for robot in sorted({t["robot"] for t in tasks}):
        done, seconds, failures = 0, 0.0, []
        for take in takes:
            path = out_dir / robot / f"{take['routine']}_s{take['performer']}.npz"
            if path.exists():
                done += 1
                seconds += float(np.load(path)["seconds"])
            else:
                failures.append({"take": take["source_file"], "status": "failed",
                                 "error": "still missing after two tries"})
        (out_dir / robot / "summary.json").write_text(json.dumps(
            {"robot": robot, "n_takes": len(takes), "done": done, "skipped": 0,
             "failed": len(failures), "elapsed_seconds": None,
             "processor_seconds": round(seconds, 1), "failures": failures}, indent=1))
        print(f"{robot}: {done}/{len(takes)} present")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
