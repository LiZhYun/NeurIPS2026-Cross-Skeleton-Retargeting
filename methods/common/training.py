"""Small helpers the training scripts share: settings files, seeding and the learning-rate schedule."""
import json
import math
import sys

import numpy as np
import torch


def apply_config(args, config_path, argv=None):
    """Fill args from a configuration file, without overriding options typed on the command line.

    Keys that start with an underscore are notes about the run and are ignored.
    """
    with open(config_path) as f:
        config = json.load(f)
    config = {k: v for k, v in config.items() if not k.startswith('_')}
    argv = sys.argv[1:] if argv is None else argv
    given = {tok[2:].split('=')[0].replace('-', '_') for tok in argv if tok.startswith('--')}
    unknown = [k for k in config if not hasattr(args, k)]
    if unknown:
        raise ValueError(f"config {config_path} has unknown settings: {sorted(unknown)}")
    for key, value in config.items():
        if key not in given:
            setattr(args, key, value)
    return args


def set_seed(seed):
    """Seed NumPy and PyTorch, on the CPU and on every GPU."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def lr_lambda(step, warmup, total):
    """Learning-rate multiplier: a linear rise over `warmup` steps, then a cosine fall to zero
    at step `total`."""
    if warmup > 0 and step < warmup:
        return float(step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
