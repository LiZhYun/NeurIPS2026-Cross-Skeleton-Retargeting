"""Distance between two motion clips on the same skeleton.

A clip is an array of joint positions with shape (frames, joints, 3). Arrays with more than
three channels per joint are accepted; only the first three (the positions) are used.
"""
import numpy as np

FIXED_FRAMES = 64


def resample(clip, frames=FIXED_FRAMES):
    """Linearly resample a clip to a fixed number of frames."""
    t = clip.shape[0]
    if t == frames:
        return clip
    grid = np.linspace(0.0, t - 1.0, frames)
    lo = np.floor(grid).astype(int)
    hi = np.minimum(lo + 1, t - 1)
    w = (grid - lo)[:, None, None].astype(clip.dtype)
    return clip[lo] * (1.0 - w) + clip[hi] * w


def _aligned_error(a, b):
    """Mean squared error after aligning each frame of `a` onto `b`.

    Each frame is centered, then rotated and uniformly scaled onto the matching frame of `b`
    (least-squares Procrustes). The alignment runs from `a` to `b`, so the measure is not
    exactly symmetric; `motion_distance(a, b)` always aligns its first argument.
    """
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    errors = []
    for at, bt in zip(a, b):
        h = at.T @ bt
        try:
            u, s, vt = np.linalg.svd(h)
        except np.linalg.LinAlgError:
            continue
        d = np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))])
        rotation = u @ d @ vt
        scale = np.trace(np.diag(s) @ d) / max(1e-8, np.sum(at ** 2))
        errors.append(np.mean((scale * at @ rotation - bt) ** 2))
    return float(np.mean(errors)) if errors else np.nan


def motion_distance(a, b, length="raw", frames=FIXED_FRAMES):
    """Distance between clips `a` and `b` of shape (frames, joints, >=3).

    length="raw":   both clips are cropped to their common number of frames.
    length="fixed": both clips are first resampled to `frames` frames, so clip duration
                    carries no information.
    Returns NaN if the clips have different joint counts or fewer than four common frames.
    """
    a = np.asarray(a)[..., :3]
    b = np.asarray(b)[..., :3]
    if a.shape[1] != b.shape[1]:
        return np.nan
    if length == "raw":
        t = min(a.shape[0], b.shape[0])
        if t < 4:
            return np.nan
        a, b = a[:t], b[:t]
    elif length == "fixed":
        a, b = resample(a, frames), resample(b, frames)
    else:
        raise ValueError(f"length must be 'raw' or 'fixed', got {length!r}")
    return _aligned_error(a, b)


def distance_matrix(clips, length="raw", frames=FIXED_FRAMES):
    """Pairwise distances between clips; entry (i, j) for i < j aligns clip i onto clip j."""
    n = len(clips)
    d = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i + 1, n):
            d[i, j] = d[j, i] = motion_distance(clips[i], clips[j], length, frames)
    return d
