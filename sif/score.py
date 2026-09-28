"""Source-Instance Fidelity (SIF) for groups of clips.

A group holds several source clips that share a source skeleton and an action, and the outputs
a method produced for them on one target skeleton, in the same order. SIF asks whether the
outputs differ from one another the way their sources do: it is the correlation between the
pairwise source distances and the pairwise output distances. A method that ignores the source
scores near zero; a method that keeps the relative differences between source clips scores near
one.
"""
from dataclasses import dataclass, field
from itertools import permutations
from typing import List, Optional

import numpy as np

from .distance import FIXED_FRAMES, distance_matrix
from .stats import bootstrap_interval, shuffle_test


def _pairs(d_src, d_out):
    n = d_src.shape[0]
    mask = ~np.eye(n, dtype=bool) & ~np.isnan(d_src) & ~np.isnan(d_out)
    return d_src[mask], d_out[mask]


def correlation(d_src, d_out):
    """Pearson correlation between two distance matrices (off-diagonal entries).

    Returns 0 when either set of distances is constant (for example, identical outputs) and
    NaN when fewer than two pairs are available.
    """
    s, o = _pairs(d_src, d_out)
    if s.size < 2:
        return np.nan
    if s.std() < 1e-8 or o.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(s, o)[0, 1])


def variation(d_src, d_out):
    """Average output distance divided by average source distance.

    Near 1 when outputs differ as much as their sources, near 0 when they collapse to a single
    motion.
    """
    s, o = _pairs(d_src, d_out)
    return float(o.mean() / s.mean()) if s.size and s.mean() > 1e-8 else 0.0


def shuffled_correlations(d_src, d_out):
    """SIF for every way of assigning outputs to sources, the true assignment first."""
    n = d_src.shape[0]
    return np.array([correlation(d_src, d_out[np.ix_(p, p)])
                     for p in map(list, permutations(range(n)))], dtype=float)


@dataclass
class GroupScore:
    sif: float
    variation: float
    n_clips: int
    block: Optional[str] = None
    d_src: np.ndarray = field(default=None, repr=False)
    d_out: np.ndarray = field(default=None, repr=False)


@dataclass
class Result:
    """Scores for a set of groups under one length setting."""
    length: str
    groups: List[GroupScore]
    sif: float            # average SIF over groups
    ci: tuple             # 95% interval, resampling whole blocks
    p: float              # one-sided shuffle-test p-value (blocks shuffled together)
    variation: float      # median variation over groups

    @property
    def n_groups(self):
        return len(self.groups)


def score_group(sources, outputs, length="raw", frames=FIXED_FRAMES, block=None):
    """Score one group. `sources` and `outputs` are equal-length lists of clips."""
    if len(sources) != len(outputs):
        raise ValueError("each source clip needs exactly one output")
    d_src = distance_matrix(sources, length, frames)
    d_out = distance_matrix(outputs, length, frames)
    return GroupScore(correlation(d_src, d_out), variation(d_src, d_out), len(sources),
                      block, d_src, d_out)


def score_groups(groups, length="raw", frames=FIXED_FRAMES, min_clips=3, max_clips=6,
                 n_shuffles=10000, n_boot=10000, seed=42):
    """Score a list of groups and summarize them.

    Each group is a dict with 'sources' and 'outputs' (lists of clips of shape
    (frames, joints, 3), outputs in the same order as sources) and an optional 'block' name,
    usually the source skeleton. Groups that share a block are shuffled together in the shuffle
    test and resampled together for the interval, because they reuse related source clips.

    length may be "raw", "fixed" or "both" (a dict with both results is returned).
    Groups with fewer than `min_clips` clips are skipped (a two-clip group has a single
    distance and no correlation); groups are cut to their first `max_clips` clips so the
    shuffle test can enumerate every assignment exactly.
    """
    if length == "both":
        return {mode: score_groups(groups, mode, frames, min_clips, max_clips,
                                   n_shuffles, n_boot, seed) for mode in ("raw", "fixed")}
    scored = []
    for g in groups:
        src, out = list(g["sources"]), list(g["outputs"])
        if len(src) < min_clips:
            continue
        src, out = src[:max_clips], out[:max_clips]
        gs = score_group(src, out, length, frames, g.get("block"))
        if not np.isnan(gs.sif):
            scored.append(gs)
    if not scored:
        raise ValueError("no group could be scored")
    blocks = [g.block if g.block is not None else i for i, g in enumerate(scored)]
    sifs = np.array([g.sif for g in scored])
    p = shuffle_test([shuffled_correlations(g.d_src, g.d_out) for g in scored],
                     blocks, n_shuffles, seed)
    ci = bootstrap_interval(sifs, blocks, n_boot, seed)
    return Result(length, scored, float(sifs.mean()), ci, p,
                  float(np.median([g.variation for g in scored])))
