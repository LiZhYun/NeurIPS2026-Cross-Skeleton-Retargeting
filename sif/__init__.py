"""Source-Instance Fidelity (SIF): does a retargeted motion keep what made its source clip
different from other clips of the same action?

Quick use:
    from sif import score_groups
    result = score_groups(groups, length="both")
    result["raw"].sif, result["raw"].ci, result["raw"].p, result["raw"].variation
"""
from .distance import motion_distance, distance_matrix, resample
from .score import correlation, variation, score_group, score_groups, GroupScore, Result
from .stats import shuffle_test, bootstrap_interval

__all__ = ["motion_distance", "distance_matrix", "resample", "correlation", "variation",
           "score_group", "score_groups", "GroupScore", "Result", "shuffle_test",
           "bootstrap_interval"]
__version__ = "1.0.0"
