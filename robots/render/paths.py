"""Where the figure's pictures, videos and records are written, and the video format.

Everything the renderer produces goes under one folder, which keeps the drawn output apart
from the study's data. The default sits inside the repository and can be changed with
`--out_dir` on any of the steps.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "paper/output/robot_figure"
VIDEO_SUFFIX = ".mp4"
