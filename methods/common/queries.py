"""Helpers for answering the queries of a benchmark set.

Each query lists the clips its answer will be compared against. Methods that take an example
clip from the target animal must never pick one of those, and every method cuts or stretches
its answer to the typical length of the clips it will be compared against.
"""
import hashlib

import numpy as np

# The lists in a query that name clips the answer will be compared against
REFERENCE_KEYS = ('positives_cluster', 'positives_exact',
                  'adversarials_easy', 'adversarials_hard',
                  'distractors_same_target_skel')


def per_query_seed(tag, qid, skel_b):
    """A random seed that depends only on the query and the target animal, so a draw made
    with it does not depend on the order the queries are worked through."""
    s = f"{tag}_{qid}_{skel_b}".encode()
    return int(hashlib.md5(s).hexdigest()[:8], 16)


def reference_length(q, default):
    """The median length of the clips the query's answer will be compared against, or the
    source clip's length when the query lists none; `default` if that is missing too."""
    pos_T = [p['T'] for p in q.get('positives_cluster', [])]
    return int(np.median(pos_T)) if pos_T else q.get('src_T', default)


def stretch_to_length(motion, target_len):
    """Resample a [T, J, C] motion linearly in time to target_len frames (float32)."""
    if motion.shape[0] == target_len:
        return motion
    from scipy.interpolate import interp1d
    xs_in = np.linspace(0, 1, motion.shape[0])
    xs_out = np.linspace(0, 1, target_len)
    out = np.zeros((target_len,) + motion.shape[1:], dtype=np.float32)
    for j in range(motion.shape[1]):
        for c in range(motion.shape[2]):
            f_interp = interp1d(xs_in, motion[:, j, c], kind='linear', assume_sorted=True)
            out[:, j, c] = f_interp(xs_out)
    return out
