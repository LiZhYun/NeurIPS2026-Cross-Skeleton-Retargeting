"""Check the downloaded recordings and choose the two windows to take from each.

Every recording is several minutes long. The study uses two ten-second windows from each,
one starting a quarter of the way in and one starting about two thirds of the way in
(65 per cent), so the clips come from settled motion rather than from the start or the end
of a recording. This step reads only the headers, confirms every recording has the
expected twenty-two joints and runs at 30 frames per second, and checks that both windows
fit without overlapping. It writes the list of recordings and windows to
`<data_dir>/takes.json`, which the later steps read.

Runs in the robot environment in a few seconds, without a GPU:
    python -m robots.lafan1_to_six_robots.choose_windows
"""
import argparse
import json
import re
import sys
from pathlib import Path

from robots.lafan1_to_six_robots.data import DEFAULT_DATA

WINDOW_FRAMES = 300
EXPECTED_JOINTS = 22


def read_header(path):
    """Joint count, frame count and frame duration, without parsing the motion."""
    joints, frames, frame_time = 0, None, None
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s.startswith(("ROOT ", "JOINT ")):
                joints += 1
            elif s.startswith("Frames:"):
                frames = int(s.split(":")[1])
            elif s.startswith("Frame Time:"):
                frame_time = float(s.split(":")[1])
                break
    return joints, frames, frame_time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=str(DEFAULT_DATA))
    ap.add_argument("--bvh_dir", default=None,
                    help="folder of downloaded recordings (default <data_dir>/lafan1_bvh)")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    bvh_dir = Path(args.bvh_dir) if args.bvh_dir else data_dir / "lafan1_bvh"
    takes, problems = [], []
    for path in sorted(bvh_dir.glob("*.bvh")):
        named = re.match(r"^(.+?)_subject(\d+)\.bvh$", path.name)
        if not named:
            problems.append(f"cannot read a routine and a performer from {path.name}")
            continue
        routine, performer = named.group(1), int(named.group(2))
        split = re.match(r"^([A-Za-z]+)(\d+)$", routine)
        action, number = (split.group(1), int(split.group(2))) if split else (routine, 0)
        joints, frames, frame_time = read_header(path)
        starts = [int(0.25 * frames), int(0.65 * frames)]
        if not (starts[1] + WINDOW_FRAMES <= frames and starts[0] + WINDOW_FRAMES <= starts[1]):
            problems.append(f"{path.name}: {frames} frames, the two windows do not fit")
        if joints != EXPECTED_JOINTS:
            problems.append(f"{path.name}: {joints} joints, expected {EXPECTED_JOINTS}")
        if abs(frame_time - 1 / 30) > 1e-4:
            problems.append(f"{path.name}: {frame_time} s per frame, expected 1/30")
        takes.append({"routine": routine, "action": action, "routine_number": number,
                      "performer": performer, "source_file": path.name,
                      "n_frames": frames, "window_starts": starts})

    routines = {}
    for take in takes:
        routines.setdefault(take["routine"], []).append(take["performer"])
    print(f"{len(takes)} recordings over {len(routines)} routines")
    print(f"length: shortest {min(t['n_frames'] for t in takes)}, "
          f"longest {max(t['n_frames'] for t in takes)}, "
          f"total {sum(t['n_frames'] for t in takes)} frames")
    thin = {r: sorted(p) for r, p in sorted(routines.items()) if len(p) < 3}
    print(f"routines with fewer than three performers, where the sparse setting keeps "
          f"everyone: {thin}")
    if problems:
        print("problems:")
        for p in problems:
            print("  " + p)
        sys.exit(1)

    out = data_dir / "takes.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"window_frames": WINDOW_FRAMES, "takes": takes}, indent=1))
    print(f"written to {out}")


if __name__ == "__main__":
    main()
