"""Read a folder of latent codes written by latents/dump_latents.py.

Every other script in this folder starts here. One folder holds one trained model's code for
each query of a benchmark set, and index.json says which query each file belongs to.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np


def read_run(run_dir, load=True):
    """Return the folder's index, in query order, with each code attached.

    Each entry has query_id, source_skeleton, target_skeleton, source_clip and, unless load
    is off, code: the saved array for that query.
    """
    run_dir = Path(run_dir)
    index_path = run_dir / 'index.json'
    if not index_path.exists():
        raise FileNotFoundError(
            f'{index_path} is missing, so this folder holds no latent codes. '
            f'Run python -m latents.dump_latents to write it.')
    index = json.loads(index_path.read_text())
    if not index:
        raise ValueError(f'{index_path} lists no queries.')
    index = sorted(index, key=lambda e: e['query_id'])
    if load:
        for entry in index:
            entry['code'] = np.load(run_dir / f"z_query_{entry['query_id']:04d}.npy")
    return index


def pooled_matrix(run_dir):
    """Stack every code of one folder into a single table of rows by width.

    AnyTop saves a code per joint and moment, the others save one per moment; both are laid
    out flat so that each row is one latent vector.
    """
    blocks = [e['code'].reshape(-1, e['code'].shape[-1]) for e in read_run(run_dir)]
    return np.concatenate(blocks, axis=0)


def group_by_target_skeleton(run_dir, min_vectors=20):
    """Stack the codes of each target skeleton, in query order.

    Skeletons with fewer than min_vectors latent vectors in total are left out, because a
    covariance over so few vectors says little.
    """
    by_skel = {}
    for entry in read_run(run_dir):
        by_skel.setdefault(entry['target_skeleton'], []).append(entry)
    out = {}
    for skel, entries in by_skel.items():
        stacked = np.concatenate([e['code'].reshape(-1, e['code'].shape[-1])
                                  for e in entries], axis=0)
        if stacked.shape[0] < min_vectors:
            continue
        out[skel] = (stacked, [e['query_id'] for e in entries])
    return out
