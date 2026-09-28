"""Two further measures used by both robot studies.

Action-level AUC asks whether the action survives retargeting: given one produced robot
motion, do real robot clips of the same action come out closer than clips of other
actions? Clips are compared after stretching both to 64 frames, aligning them in time,
and aligning them in space, and the answer is read off as the chance that a same-action
clip ranks ahead of a different-action one. Half means the produced motion carries no
information about which action it came from.

Realism asks whether a produced motion looks like real robot motion at all: how far each
of its poses is from the nearest pose in a bank of real robot poses. The cut-off is set
from real clips that were kept out of the bank, before any produced motion is measured.
"""
from collections import defaultdict

import numpy as np


def _align(a, b):
    """Rotation, uniform scale and the two centroids that best map `a` onto `b`."""
    centre_a = a.mean(axis=0, keepdims=True)
    centre_b = b.mean(axis=0, keepdims=True)
    at, bt = a - centre_a, b - centre_b
    u, s, vt = np.linalg.svd(at.T @ bt)
    flip = np.sign(np.linalg.det(u @ vt))
    if flip == 0:
        flip = 1.0
    rotation = u @ np.diag([1.0, 1.0, flip]) @ vt
    singular = s.copy()
    singular[-1] *= flip
    scale = singular.sum() / ((at ** 2).sum() + 1e-9)
    return rotation, float(scale), centre_a, centre_b


def _time_alignment(a, b):
    """Indices that pair up the frames of two clips by dynamic time warping."""
    n_a, n_b = a.shape[0], b.shape[0]
    if n_a == 0 or n_b == 0:
        return [], []
    cost = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    total = np.full((n_a + 1, n_b + 1), np.inf, dtype=np.float64)
    total[0, 0] = 0.0
    for i in range(1, n_a + 1):
        for j in range(1, n_b + 1):
            total[i, j] = cost[i - 1, j - 1] + min(total[i - 1, j - 1], total[i - 1, j],
                                                   total[i, j - 1])
    path_a, path_b = [], []
    i, j = n_a, n_b
    while i > 0 and j > 0:
        path_a.append(i - 1)
        path_b.append(j - 1)
        step = sorted([(total[i - 1, j - 1], i - 1, j - 1), (total[i - 1, j], i - 1, j),
                       (total[i, j - 1], i, j - 1)])
        _, i, j = step[0]
    return list(reversed(path_a)), list(reversed(path_b))


def trajectory_distance(predicted, reference, time_align=True, root_relative=True):
    """Average distance between two motions after matching them in time and space."""
    if predicted.shape[1] != reference.shape[1]:
        joints = min(predicted.shape[1], reference.shape[1])
        predicted, reference = predicted[:, :joints], reference[:, :joints]
    if root_relative:
        predicted = predicted - predicted[:, :1, :]
        reference = reference - reference[:, :1, :]
    if time_align:
        idx_a, idx_b = _time_alignment(predicted.reshape(predicted.shape[0], -1),
                                       reference.reshape(reference.shape[0], -1))
        if not idx_a:
            return float("inf")
        a = predicted[idx_a].reshape(-1, 3)
        b = reference[idx_b].reshape(-1, 3)
    else:
        frames = min(predicted.shape[0], reference.shape[0])
        a = predicted[:frames].reshape(-1, 3)
        b = reference[:frames].reshape(-1, 3)
    rotation, scale, centre_a, centre_b = _align(a, b)
    aligned = scale * (a - centre_a) @ rotation + centre_b
    return float(np.mean(np.linalg.norm(aligned - b, axis=1)))


def rank_auc(same_action, other_action):
    """Chance that a same-action clip is closer than a different-action one."""
    same = [d for d in same_action if d is not None and np.isfinite(d)]
    other = [d for d in other_action if d is not None and np.isfinite(d)]
    if not same or not other:
        return None
    scores = np.concatenate([-np.asarray(same), -np.asarray(other)])
    labels = np.concatenate([np.ones(len(same)), np.zeros(len(other))])
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    sorted_scores = scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n_same = int(labels.sum())
    n_other = len(labels) - n_same
    return float((ranks[labels == 1].sum() - n_same * (n_same + 1) / 2.0)
                 / (n_same * n_other))


def grouped_interval(values_with_groups, n_resamples=2000, level=0.95, seed=42):
    """Average of the values, with an interval from drawing whole action groups again.

    `values_with_groups` pairs each value with its action group. Returns the lower end of
    the interval, the average and the upper end, in that order.
    """
    values = [(v, g) for v, g in values_with_groups if v is not None]
    if not values:
        return (0.0, 0.0, 0.0)
    rng = np.random.RandomState(seed)
    by_group = defaultdict(list)
    for value, group in values:
        by_group[group].append(value)
    groups = list(by_group.keys())
    means = []
    for _ in range(n_resamples):
        drawn = rng.choice(len(groups), len(groups), replace=True)
        pooled = []
        for g in drawn:
            pooled.extend(by_group[groups[g]])
        means.append(np.mean(pooled) if pooled else 0.0)
    means = np.array(means)
    return (float(np.percentile(means, (1 - level) / 2 * 100)),
            float(np.mean([v for v, _ in values])),
            float(np.percentile(means, (1 + level) / 2 * 100)))


def resample_frames(clip, frames=64):
    """Stretch or squeeze a motion of shape (frames, joints, 3) to exactly `frames`."""
    length = clip.shape[0]
    if length == frames:
        return clip
    grid = np.linspace(0, length - 1, frames)
    low = np.floor(grid).astype(int)
    high = np.minimum(low + 1, length - 1)
    weight = (grid - low)[:, None, None]
    return (1 - weight) * clip[low] + weight * clip[high]


def pose_shape(clip):
    """A size-free description of each pose: the joints' relative arrangement."""
    centred = clip - clip.mean(1, keepdims=True)
    scale = np.sqrt((centred ** 2).sum(-1).mean(-1, keepdims=True))[:, :, None] + 1e-6
    centred = centred / scale
    gram = np.einsum("tjd,tkd->tjk", centred, centred)
    upper = np.triu_indices(gram.shape[1])
    return gram[:, upper[0], upper[1]]


def distance_to_nearest(poses, bank, n_samples=40):
    """Average distance from a clip's poses to the closest pose in a bank of real ones."""
    picks = np.linspace(0, len(poses) - 1, min(n_samples, len(poses))).astype(int)
    return float(np.mean([np.sqrt(((bank - poses[i]) ** 2).sum(-1)).min() for i in picks]))


def joints_in_bank(bank):
    """How many joints a pose bank was built from."""
    entries = bank.shape[1]
    return int((np.sqrt(8 * entries + 1) - 1) / 2)
