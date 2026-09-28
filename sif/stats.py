"""How much to trust an average SIF: a shuffle test and a confidence interval."""
from collections import OrderedDict

import numpy as np


def _block_members(blocks):
    members = OrderedDict()
    for i, b in enumerate(blocks):
        members.setdefault(b, []).append(i)
    return members


def shuffle_test(shuffled, blocks, n_shuffles=10000, seed=42):
    """How often an average SIF this high would appear if the outputs ignored their sources.

    `shuffled[i]` holds group i's SIF for every way of pairing its outputs with its source clips,
    the true pairing first (see score.shuffled_correlations). We repeat many times: each group
    picks one pairing at random, the true one included, and we average over groups. Groups in the
    same block pick together, because they reuse related source clips. The result is the share of
    repeats whose average reaches the observed one, counting the observed average itself, so it is
    never exactly zero.
    """
    full = [np.nan_to_num(np.asarray(r, dtype=float), nan=0.0) for r in shuffled]
    observed = float(np.mean([r[0] for r in full]))
    rng = np.random.RandomState(seed)
    null = np.zeros(n_shuffles)
    members = _block_members(blocks)
    draws = rng.rand(n_shuffles, len(members))
    for k, idx in enumerate(members.values()):
        u = draws[:, k]
        for i in idx:
            n = len(full[i])
            null += full[i][np.minimum((u * n).astype(int), n - 1)]
    null /= len(full)
    return (int(np.sum(null >= observed)) + 1) / (n_shuffles + 1)


def bootstrap_interval(values, blocks, n_boot=10000, seed=42, level=0.95):
    """Range that holds the average of `values` in `level` of resamples.

    Each resample draws whole blocks at random, with repeats allowed, so related groups stay together.
    """
    values = np.asarray(values, dtype=float)
    members = list(_block_members(blocks).values())
    rng = np.random.RandomState(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        drawn = rng.choice(len(members), len(members), replace=True)
        idx = [i for k in drawn for i in members[k]]
        means[b] = values[idx].mean()
    means.sort()
    tail = (1.0 - level) / 2.0
    return float(means[int(tail * n_boot)]), float(means[int((1.0 - tail) * n_boot)])
