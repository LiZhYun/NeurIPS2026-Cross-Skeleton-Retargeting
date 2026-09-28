"""Record the clip measurements, draw both halves of the figure, lay them out and gather
the record, in that order.

Runs in the robot environment (robots/environment.yml) on a machine with a graphics card.
It needs the six-robot study's data folder, with its trained models' motions, and the
retargeting tool in external/GMR for the robot descriptions. On one graphics card it takes
about three minutes with videos, and under a minute with --no_video (stills only):
    python -m robots.render.run_panels
"""
import argparse
import runpy
import sys

from robots.lafan1_to_six_robots.data import DEFAULT_DATA, ROOT
from robots.render.paths import DEFAULT_OUT

SELECTION = ["robots.render.select_clips"]
DRAWING = ["robots.render.panel_six_robots", "robots.render.panel_two_performers"]
LAYOUT = ["robots.render.compose", "robots.render.merge_report"]


def run(steps, arguments):
    """Run each step as if from the command line, with the arguments meant for it."""
    for step in steps:
        print(f"{step.rsplit('.', 1)[1]}", flush=True)
        sys.argv = [step] + arguments[step]
        runpy.run_module(step, run_name="__main__")


def parse():
    """The command-line arguments of every step, from this step's own options."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=str(DEFAULT_DATA))
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT))
    ap.add_argument("--gmr_dir", default=str(ROOT / "external/GMR"))
    ap.add_argument("--no_video", action="store_true", help="draw the stills only")
    args = ap.parse_args()
    draw = ["--data_dir", args.data_dir, "--out_dir", args.out_dir,
            "--gmr_dir", args.gmr_dir] + (["--no_video"] if args.no_video else [])
    arguments = {step: draw for step in DRAWING}
    arguments.update({step: ["--out_dir", args.out_dir] for step in LAYOUT})
    arguments.update({step: ["--data_dir", args.data_dir, "--out_dir", args.out_dir]
                      for step in SELECTION})
    return arguments


if __name__ == "__main__":
    run(SELECTION + DRAWING + LAYOUT, parse())
