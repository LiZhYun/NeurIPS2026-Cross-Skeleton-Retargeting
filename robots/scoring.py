"""Source-Instance Fidelity for the two robot studies.

Both studies ask the same question the paper asks of the animal data: when several
people perform the same action, do the retargeted robot motions differ from one another
the way the human recordings do? That is measured by Source-Instance Fidelity (SIF).

An action group is the set of clips scored together: several human recordings of one
action (in the six-robot study, several people performing one numbered routine), together
with the robot motion a method produced for each of them, in the same order. SIF is the
correlation between the distances among the human clips and the distances among the robot
motions. Variation is how far apart the robot motions are relative to their sources; it
falls to zero when a method returns nearly the same motion whatever it is given.

The distance between two motions comes from the installable `sif` package, so the robot
studies and the animal benchmark measure motions the same way.

The shuffle test below differs from the one in `sif.stats` in one respect: each
action group draws its own shuffle independently, and shuffles whose correlation is
undefined are left out of that group's pool. Action groups here never share source clips,
so drawing them independently is the right comparison, and it is the test behind the
paper's robot numbers.
"""
from itertools import permutations

import numpy as np

from sif import bootstrap_interval, correlation, distance_matrix, variation

# The two ways of comparing motions of unequal duration. "raw" compares over the frames
# the two clips share; "length_controlled" first stretches both to the same number of
# frames, so duration itself carries no information.
LENGTH_SETTINGS = {"raw": "raw", "length_controlled": "fixed"}


def score_group(name, sources, outputs, length="raw"):
    """One action group's SIF and variation, plus the distance matrices the tests need.

    Returns None when fewer than two clips have both a source and an output, or when the
    correlation is undefined.
    """
    pairs = [(s, o) for s, o in zip(sources, outputs) if s is not None and o is not None]
    if len(pairs) < 2:
        return None
    setting = LENGTH_SETTINGS[length]
    d_src = distance_matrix([p[0] for p in pairs], setting)
    d_out = distance_matrix([p[1] for p in pairs], setting)
    sif = correlation(d_src, d_out)
    if np.isnan(sif):
        return None
    return {"name": name, "n_clips": len(pairs), "sif": sif,
            "variation": variation(d_src, d_out), "d_src": d_src, "d_out": d_out}


def shuffled_correlations(d_src, d_out):
    """SIF for every way of handing the outputs to the sources, the true one first."""
    n = d_src.shape[0]
    return np.array([correlation(d_src, d_out[np.ix_(np.asarray(p), np.asarray(p))])
                     for p in permutations(range(n))], dtype=float)


def shuffle_test(groups, n_draws=10000, seed=42):
    """How often an average SIF this high appears when outputs ignore their sources.

    Each draw gives every action group one of its possible pairings, the true pairing
    included, and averages over action groups. The p-value counts the observed average
    itself, so it is never exactly zero.
    """
    pools, observed = [], []
    for g in groups:
        shuffled = shuffled_correlations(g["d_src"], g["d_out"])
        others = shuffled[1:]
        others = others[~np.isnan(others)]
        observed.append(float(shuffled[0]))
        pools.append(np.concatenate([[shuffled[0]], others]))
    average = float(np.mean(observed))
    rng = np.random.RandomState(seed)
    null = np.zeros(n_draws)
    for pool in pools:
        null += pool[rng.randint(0, len(pool), size=n_draws)]
    null /= len(pools)
    spread = float(null.std(ddof=1))
    return {"observed": average, "shuffled_mean": float(null.mean()),
            "z": (average - null.mean()) / spread if spread > 0 else float("nan"),
            "p_value": (int(np.sum(null >= average)) + 1) / (n_draws + 1)}


def summarize(method, groups, n_draws=10000, n_resamples=10000, seed=42):
    """One table row: SIF, variation, the shuffle test and a 95% interval for SIF.

    The interval comes from drawing whole action groups again with repeats.
    """
    if not groups:
        return {"method": method, "n_action_groups": 0}
    sifs = np.array([g["sif"] for g in groups], dtype=float)
    variations = np.array([g["variation"] for g in groups], dtype=float)
    test = shuffle_test(groups, n_draws, seed)
    low, high = bootstrap_interval(sifs, list(range(len(groups))), n_resamples, seed)
    return {
        "method": method,
        "n_action_groups": len(groups),
        "sif": float(sifs.mean()),
        "sif_median": float(np.median(sifs)),
        "variation_mean": float(variations.mean()),
        "variation": float(np.median(variations)),
        "variation_middle_half": [float(np.percentile(variations, 25)),
                                  float(np.percentile(variations, 75))],
        "shuffled_mean": test["shuffled_mean"],
        "z": test["z"],
        "p_value": test["p_value"],
        "sif_ci95": [low, high],
    }


def resample_mean(values, n_resamples=10000, seed=42, levels=(95, 90)):
    """Average of `values` with intervals from drawing the values again with repeats."""
    values = np.asarray(list(values), dtype=float)
    rng = np.random.RandomState(seed)
    drawn = np.sort([values[rng.randint(0, len(values), len(values))].mean()
                     for _ in range(n_resamples)])
    out = {"mean": float(values.mean())}
    for level in levels:
        tail = (100 - level) / 2.0
        out[f"ci{level}"] = [float(np.percentile(drawn, tail)),
                             float(np.percentile(drawn, 100 - tail))]
    return out


def variation_ratio(method_by_group, reference_by_group, n_resamples=10000, seed=42):
    """How much of the true retarget's variation a method keeps, per action group.

    Both arguments map an action group's name to its variation. The two are paired by name,
    so the ratio and its interval use the same action groups on both sides. The ratio is the
    median of the method's variations divided by the median of the true retarget's; the
    paper calls it Q.
    """
    shared = sorted(set(method_by_group) & set(reference_by_group))
    if len(shared) < 3:
        return None
    method = np.array([method_by_group[g] for g in shared], float)
    reference = np.array([reference_by_group[g] for g in shared], float)
    point = float(np.median(method) / max(np.median(reference), 1e-9))
    rng = np.random.RandomState(seed)
    drawn = np.empty(n_resamples)
    for b in range(n_resamples):
        pick = rng.randint(0, len(shared), len(shared))
        drawn[b] = np.median(method[pick]) / max(np.median(reference[pick]), 1e-9)
    return {"ratio": point,
            "ci95": [float(np.percentile(drawn, 2.5)), float(np.percentile(drawn, 97.5))],
            "n_action_groups": len(shared)}
