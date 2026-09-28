"""The rows of this study's tables, and how each one produces a robot motion for a clip.

true_retarget   the clip's own retarget, produced by the retargeting tool
random_clip     the retarget of a different clip of the same action group, drawn at random
the three models the motions written by the training step (unpaired_objective,
                averaging_objective, true_pair_model), read from
                <data_dir>/generated/<robot>/<setting>/<model>
"""
from pathlib import Path

import numpy as np

from robots.lafan1_to_six_robots.data import HORIZON, load_tensor

METHODS = ["true_retarget", "random_clip", "unpaired_objective", "averaging_objective",
           "true_pair_model"]


def method_reader(name, data_dir, robot, setting, generated_dir, seed=42):
    """A function that returns one robot motion for a clip, or None if there is none."""
    if name == "true_retarget":
        return lambda clip_id, clip_ids: load_tensor(data_dir, robot, clip_id)

    if name == "random_clip":
        rng = np.random.RandomState(seed)

        def read_random(clip_id, clip_ids):
            others = [c for c in clip_ids if c != clip_id] or clip_ids
            return load_tensor(data_dir, robot, others[rng.randint(len(others))])

        return read_random

    folder = Path(generated_dir) / robot / setting / name

    def read_produced(clip_id, clip_ids):
        path = folder / f"{clip_id}.npy"
        return np.load(path).astype(np.float32)[:HORIZON] if path.exists() else None

    return read_produced
