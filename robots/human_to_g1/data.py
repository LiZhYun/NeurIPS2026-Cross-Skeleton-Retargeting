"""Where this study's data lives and how its clips are grouped.

`clips.json` lists the actions the study uses and, for each action, the recordings chosen
for it. The recordings of one action form an action group, the set of clips scored
together. The file is the one every later step reads, so a step never has to guess which
clips belong together. It is not shipped with this repository, because it copies parts of
the collection's metadata, which the collection's licence treats as confidential;
`select_clips.py` rebuilds exactly the paper's selection from the metadata table.

Everything the study reads or writes sits in one data folder, `data/robots/human_to_g1`
by default. Joint positions are stored one file per clip under `<data>/tensors/human` and
`<data>/tensors/g1`, in metres, at 30 frames per second, and the two sides of a clip are
frame for frame the same take.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data/robots/human_to_g1"


def clip_list_path(data_dir):
    return Path(data_dir) / "clips.json"


def load_clip_list(data_dir):
    """Read clips.json from the data folder."""
    path = clip_list_path(data_dir)
    if not path.exists():
        raise SystemExit(f"{path} does not exist. Build it first with\n"
                         f"    python -m robots.human_to_g1.select_clips "
                         f"--metadata <seed_metadata_v004.parquet>")
    with open(path) as f:
        return json.load(f)


def tensor_path(data_dir, side, clip_id):
    return Path(data_dir) / "tensors" / side / f"{clip_id}.npy"


def load_tensor(data_dir, side, clip_id):
    """Joint positions for one clip, or None if that clip was never converted."""
    path = tensor_path(data_dir, side, clip_id)
    return np.load(path).astype(np.float32) if path.exists() else None


def action_groups(clip_list, data_dir, min_clips=2):
    """The action groups that have enough converted human clips to be scored."""
    groups = []
    for entry in clip_list["actions"]:
        clip_ids = [c["clip_id"] for c in entry["clips"]
                    if tensor_path(data_dir, "human", c["clip_id"]).exists()]
        if len(clip_ids) >= min_clips:
            groups.append({"action": entry["action"], "package": entry["package"],
                           "clip_ids": clip_ids})
    return groups


def load_corpus(data_dir, max_groups=0):
    """Every clip that has both sides converted, ready for training.

    Returns the clips keyed by clip name, each holding the human motion, the robot motion
    and the action it belongs to, together with the clip names grouped by action.
    """
    clips, groups = {}, {}
    for entry in load_clip_list(data_dir)["actions"]:
        kept = []
        for c in entry["clips"]:
            clip_id = c["clip_id"]
            human = tensor_path(data_dir, "human", clip_id)
            robot = tensor_path(data_dir, "g1", clip_id)
            if human.exists() and robot.exists():
                clips[clip_id] = {"human": np.load(human), "g1": np.load(robot),
                                  "action": entry["action"]}
                kept.append(clip_id)
        if kept:
            groups[entry["action"]] = kept
        if max_groups and len(groups) >= max_groups:
            break
    return clips, groups
