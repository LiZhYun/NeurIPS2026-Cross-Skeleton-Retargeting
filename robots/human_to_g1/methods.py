"""The rows of the study's tables, and how each one produces a robot motion for a clip.

true_retarget   the clip's own robot counterpart, the G1 version of the same take
random_clip     the real robot counterpart of a different clip of the same action, drawn at
                random; it shows what knowing only the action name is worth
the four models the motions written by the training steps (unpaired_objective,
                averaging_objective, true_pair_model, adversarial_objective), read from
                their folders under <data_dir>/generated
"""
from pathlib import Path

import numpy as np

from robots.human_to_g1.data import load_tensor

METHODS = ["true_retarget", "random_clip", "unpaired_objective", "averaging_objective",
           "true_pair_model", "adversarial_objective"]


def method_reader(name, data_dir, generated_dir, seed=42):
    """A function that returns one robot motion for a clip, or None if there is none."""
    if name == "true_retarget":
        return lambda clip_id, group: load_tensor(data_dir, "g1", clip_id)

    if name == "random_clip":
        rng = np.random.RandomState(seed)

        def read_random(clip_id, group):
            others = [c for c in group["clip_ids"] if c != clip_id] or group["clip_ids"]
            return load_tensor(data_dir, "g1", others[rng.randint(len(others))])

        return read_random

    folder = Path(generated_dir) / name

    def read_produced(clip_id, group):
        path = folder / f"{clip_id}.npy"
        return np.load(path).astype(np.float32) if path.exists() else None

    return read_produced
