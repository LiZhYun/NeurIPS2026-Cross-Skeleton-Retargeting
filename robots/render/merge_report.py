"""Gather what the figure is made of into one file.

The clip choice, the two drawing steps and the fit errors each leave their own record. This
puts them together, so a reader can check which clips the figure shows, how the two halves
were framed, and how closely the models' body positions could be put back onto the robot.

Paths in the record are relative to the output folder. Runs in either environment, in a
second:
    python -m robots.render.merge_report
"""
import argparse
import json
from pathlib import Path

from robots.render.paths import DEFAULT_OUT, VIDEO_SUFFIX


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    reports = out_dir / "reports"
    with open(reports / "selection.json") as f:
        selection = json.load(f)
    with open(reports / "six_robots.json") as f:
        six = json.load(f)
    with open(reports / "two_performers.json") as f:
        two = json.load(f)

    shown = {p["clip_id"] for p in two["performers"]}
    merged = {
        "figure": {"png": "panels/robot_figure.png",
                   "pdf": "panels/robot_figure.pdf",
                   "first_half_png": "panels/six_robots.png",
                   "second_half_png": "panels/two_performers.png"},
        "six_robots": six,
        "two_performers": two,
        "fit_error_cm": {k: round(v["fit_error_m"]["rms_m"] * 100, 2)
                         for k, v in two["tiles"].items() if "fit_error_m" in v},
        "why_these_clips": {
            "most_expressive": selection["expressive_clips"][:5],
            "chosen_pair": [r for r in selection["pairs_by_contrast"]
                            if {r["a"], r["b"]} == shown],
        },
        "videos": sorted(str(p.relative_to(out_dir))
                         for p in out_dir.glob(f"videos/**/*{VIDEO_SUFFIX}")),
        "how_it_was_drawn": {
            "renderer": "MuJoCo, drawn off screen on the graphics card, edges smoothed with "
                        "8 samples per pixel, the robots' real shapes",
            "robot_descriptions": "the retargeting tool's own robot descriptions",
            "fit": "joint angles found to put each body at its predicted position, "
                   "orientations ignored, joint limits respected, each frame starting from "
                   "the previous one, 30 solver steps per frame",
        },
    }
    (reports / "report.json").write_text(json.dumps(merged, indent=2))
    print(json.dumps(merged["fit_error_cm"], indent=2))
    print("written to", reports / "report.json")


if __name__ == "__main__":
    main()
