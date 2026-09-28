"""A short example: what SIF reports for three kinds of retargeting.

We make toy source clips on a 20-joint skeleton, grouped by action, and three "methods" that
produce outputs on a different, 12-joint skeleton:

  faithful   keeps what makes each source clip different (the target body follows a fixed
             subset of the source joints),
  collapsed  returns the same motion for every clip of an action,
  unrelated  returns plausible target motions that have nothing to do with the given clip.

Run from the repository root:  python examples/quickstart.py
"""
import numpy as np

from sif import score_groups

rng = np.random.default_rng(0)
FRAMES, SRC_JOINTS, TGT_JOINTS = 60, 20, 12
target_joints = rng.choice(SRC_JOINTS, TGT_JOINTS, replace=False)


def source_clip(action_pose, style):
    """A clip = an action-specific motion plus a clip-specific style, with a little noise."""
    t = np.linspace(0, 2 * np.pi, FRAMES)[:, None, None]
    return (action_pose * np.sin(t) + style * np.sin(2 * t + style)
            + 0.02 * rng.normal(size=(FRAMES, SRC_JOINTS, 3)))


def faithful(clip):
    return clip[:, target_joints]


def main():
    groups = {"faithful": [], "collapsed": [], "unrelated": []}
    for skeleton in range(6):                   # six source skeletons (blocks)
        for action in range(5):                  # five actions each
            pose = rng.normal(size=(SRC_JOINTS, 3))
            sources = [source_clip(pose, rng.normal(scale=0.5, size=(SRC_JOINTS, 3)))
                       for _ in range(4)]         # four clips per group
            same_motion = faithful(sources[0])
            for name, outputs in [
                ("faithful", [faithful(s) for s in sources]),
                ("collapsed", [same_motion for _ in sources]),
                ("unrelated", [faithful(source_clip(pose, rng.normal(scale=0.5, size=(SRC_JOINTS, 3))))
                               for _ in sources]),
            ]:
                groups[name].append({"sources": sources, "outputs": outputs,
                                     "block": f"skeleton_{skeleton}"})

    print("method      SIF     95% CI              p       variation")
    for name, g in groups.items():
        r = score_groups(g, length="raw")
        print(f"{name:10s} {r.sif:+.3f}  [{r.ci[0]:+.3f}, {r.ci[1]:+.3f}]  {r.p:.4f}  {r.variation:.3f}")


if __name__ == "__main__":
    main()
