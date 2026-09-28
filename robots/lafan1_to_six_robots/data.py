"""Where this study's data lives, and the clip list every step reads.

The motion capture set used here records a handful of people performing numbered routines:
dance1, fight1, aiming2 and so on. A routine performed by several different people is what
makes the paper's question answerable, because the several performances of one routine are
exactly the instances whose differences should survive retargeting. So in this study an
action group, the set of clips scored together, is one routine: one or two clips per
performer of it.

Two settings are scored side by side. In the sparse setting an action group keeps at most
three performers and one window each, which is about as much data per group as the animal
set has. In the dense setting it keeps everything.

The clip list (`<data_dir>/clips.json`) and the joint positions are written by
`build_tensors.py`; neither is shipped with this repository, because the motion capture set
may not be redistributed, nor anything made from it.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data/robots/lafan1_to_six_robots"
ROBOTS = ["unitree_g1", "booster_t1_29dof", "fourier_n1", "stanford_toddy",
          "engineai_pm01", "pal_talos"]
SETTINGS = ["sparse", "dense"]
# Every row is scored over the same first 256 frames, which is the length the models
# produce (four whole 64-frame windows of a 300-frame clip), so no row is judged over more
# of a clip than another.
HORIZON = 256


def clip_list_path(data_dir):
    return Path(data_dir) / "clips.json"


def load_clip_list(data_dir):
    """Read clips.json from the data folder."""
    path = clip_list_path(data_dir)
    if not path.exists():
        raise SystemExit(f"{path} does not exist. Build it first with\n"
                         f"    python -m robots.lafan1_to_six_robots.build_tensors")
    with open(path) as f:
        return json.load(f)


def tensor_path(data_dir, side, clip_id):
    return Path(data_dir) / "tensors" / side / f"{clip_id}.npy"


def load_tensor(data_dir, side, clip_id, horizon=HORIZON):
    """Joint positions for one clip, cut to the common length, or None if missing."""
    path = tensor_path(data_dir, side, clip_id)
    return np.load(path).astype(np.float32)[:horizon] if path.exists() else None
