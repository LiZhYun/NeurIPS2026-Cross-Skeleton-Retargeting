import json
from pathlib import Path

import numpy as np
import pytest

from sif import motion_distance, score_group, score_groups

DATA = Path(__file__).parent / "data"


def random_rotation(rng):
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    return q if np.linalg.det(q) > 0 else -q


def toy_groups(rng, n_groups=8, n_clips=4, frames=50, joints=10):
    groups = []
    for g in range(n_groups):
        sources = [rng.normal(size=(frames, joints, 3)).cumsum(0) * 0.1 for _ in range(n_clips)]
        groups.append({"sources": sources, "block": f"b{g % 4}"})
    return groups


def test_distance_is_zero_for_identical_clips():
    clip = np.random.default_rng(0).normal(size=(30, 8, 3))
    assert motion_distance(clip, clip) == pytest.approx(0.0, abs=1e-12)


def test_distance_ignores_rotation_scale_and_translation():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=(30, 8, 3)), rng.normal(size=(30, 8, 3))
    moved = 2.5 * a @ random_rotation(rng) + rng.normal(size=3)
    assert motion_distance(moved, b) == pytest.approx(motion_distance(a, b), rel=1e-6)
    assert motion_distance(moved, a) == pytest.approx(0.0, abs=1e-10)


def test_distance_uses_positions_only():
    rng = np.random.default_rng(2)
    a, b = rng.normal(size=(30, 8, 13)), rng.normal(size=(30, 8, 13))
    assert motion_distance(a, b) == motion_distance(a[..., :3], b[..., :3])


def test_distance_needs_matching_joints_and_enough_frames():
    rng = np.random.default_rng(3)
    assert np.isnan(motion_distance(rng.normal(size=(30, 8, 3)), rng.normal(size=(30, 9, 3))))
    assert np.isnan(motion_distance(rng.normal(size=(3, 8, 3)), rng.normal(size=(30, 8, 3))))
    assert not np.isnan(motion_distance(rng.normal(size=(3, 8, 3)), rng.normal(size=(30, 8, 3)),
                                        length="fixed"))


def test_outputs_equal_to_sources_score_one():
    rng = np.random.default_rng(4)
    g = toy_groups(rng, n_groups=1)[0]
    s = score_group(g["sources"], g["sources"])
    assert s.sif == pytest.approx(1.0)
    assert s.variation == pytest.approx(1.0)


def test_collapsed_outputs_score_zero():
    rng = np.random.default_rng(5)
    groups = [dict(g, outputs=[g["sources"][0]] * len(g["sources"])) for g in toy_groups(rng)]
    r = score_groups(groups, n_shuffles=500, n_boot=500)
    assert r.sif == 0.0 and r.p == 1.0
    assert r.variation == pytest.approx(0.0, abs=1e-12)


def test_faithful_beats_unrelated():
    rng = np.random.default_rng(6)
    groups = toy_groups(rng)
    faithful = [dict(g, outputs=[s[:, :6] for s in g["sources"]]) for g in groups]
    unrelated = [dict(g, outputs=[rng.normal(size=s[:, :6].shape).cumsum(0) * 0.1
                                  for s in g["sources"]]) for g in groups]
    rf = score_groups(faithful, n_shuffles=2000, n_boot=500)
    ru = score_groups(unrelated, n_shuffles=2000, n_boot=500)
    assert rf.sif > 0.5 and rf.p < 0.01
    assert abs(ru.sif) < 0.3 and ru.p > 0.01


def test_small_groups_are_skipped_and_large_groups_cut():
    rng = np.random.default_rng(7)
    groups = toy_groups(rng, n_groups=4, n_clips=8)
    groups[0]["sources"] = groups[0]["sources"][:2]
    for g in groups:
        g["outputs"] = g["sources"]
    r = score_groups(groups, n_shuffles=100, n_boot=100)
    assert r.n_groups == 3
    assert all(g.n_clips == 6 for g in r.groups)


def test_results_are_deterministic():
    rng = np.random.default_rng(8)
    groups = [dict(g, outputs=[s[::-1] for s in g["sources"]]) for g in toy_groups(rng)]
    a = score_groups(groups, length="both", n_shuffles=300, n_boot=300)
    b = score_groups(groups, length="both", n_shuffles=300, n_boot=300)
    for mode in ("raw", "fixed"):
        assert (a[mode].sif, a[mode].ci, a[mode].p) == (b[mode].sif, b[mode].ci, b[mode].p)


def test_matches_the_code_used_for_the_paper():
    """Numbers in reference_case.json were computed with the scripts behind the paper."""
    spec = json.loads((DATA / "reference_case.json").read_text())
    arrays = np.load(DATA / "reference_case.npz")
    groups = [{"block": g["block"],
               "sources": [arrays[f"g{i}_sources_{c}"] for c in range(g["n"])],
               "outputs": [arrays[f"g{i}_outputs_{c}"] for c in range(g["n"])]}
              for i, g in enumerate(spec["groups"])]
    for mode, want in spec["expected"].items():
        r = score_groups(groups, length=mode, n_shuffles=spec["n_shuffles"],
                         n_boot=spec["n_boot"], seed=spec["seed"])
        np.testing.assert_allclose([g.sif for g in r.groups], want["group_sif"], atol=1e-12)
        np.testing.assert_allclose([g.variation for g in r.groups], want["group_variation"],
                                   atol=1e-12)
        assert r.sif == pytest.approx(want["sif"], abs=1e-12)
        assert r.p == pytest.approx(want["p"], abs=1e-12)
        np.testing.assert_allclose(r.ci, want["ci"], atol=1e-12)
        assert r.variation == pytest.approx(want["variation_median"], abs=1e-12)
