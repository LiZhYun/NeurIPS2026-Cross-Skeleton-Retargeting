"""Read a human motion capture file and work out where every joint is, frame by frame.

A BVH file stores a skeleton and, for each frame, an angle for every joint. Turning that
into joint positions means walking down the skeleton and applying each joint's rotation to
everything below it. The result is what the rest of this study uses: one point per joint,
in metres, with the up direction along Y.

This study reads the files itself rather than through a library because the human rig in
this collection puts position channels on two joints instead of one, which most readers
refuse. Each joint's own channel list is honoured here, which is what the file format
actually allows.

The conversion step uses this module; it can also be run on a single file to see what the
file contains:
    python -m robots.human_to_g1.human_positions <file.bvh>
"""
import argparse
import re
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

_ROTATION = {"Xrotation": "X", "Yrotation": "Y", "Zrotation": "Z"}
_POSITION = {"Xposition": 0, "Yposition": 1, "Zposition": 2}
_FINGER = re.compile(r"(Thumb|Index|Middle|Ring|Pinky)")


class Skeleton:
    """A skeleton's joint names, parents, rest offsets and channel lists."""

    def __init__(self, names, parents, offsets, channels):
        self.names = names
        self.parents = parents
        self.offsets = np.asarray(offsets, np.float64)
        self.channels = channels
        self.n_joints = len(names)

    @property
    def body_joint_indices(self):
        """The joints kept for this study: no fingers, and no fixed world origin.

        The robot has no fingers, and the rig's outermost joint never moves: the body's
        travel through the world is carried by the hips.
        """
        return [i for i, n in enumerate(self.names)
                if not _FINGER.search(n) and n != "Root"]


def parse_bvh(path):
    """Read a BVH file into a skeleton, the raw per-frame channel values and the timestep."""
    names, parents, offsets, channels = [], [], [], []
    stack = [-1]
    in_motion = False
    frame_time = None
    rows = []
    declared_frames = None

    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if in_motion:
                if s.startswith("Frames:"):
                    declared_frames = int(s.split(":")[1])
                elif s.startswith("Frame Time:"):
                    frame_time = float(s.split(":")[1])
                else:
                    rows.append([float(x) for x in s.split()])
                continue

            token = s.split()
            key = token[0]
            if key in ("ROOT", "JOINT"):
                names.append(token[1])
                parents.append(stack[-1])
                offsets.append([0.0, 0.0, 0.0])
                channels.append([])
            elif key == "End":                 # a bare tip, carrying no motion of its own
                stack.append(-2)
            elif key == "{":
                if stack[-1] != -2:
                    stack.append(len(names) - 1)
            elif key == "}":
                stack.pop()
            elif key == "OFFSET":
                if stack[-1] != -2:
                    offsets[stack[-1]] = [float(token[1]), float(token[2]), float(token[3])]
            elif key == "CHANNELS":
                channels[stack[-1]] = token[2:]
            elif key == "MOTION":
                in_motion = True

    skeleton = Skeleton(names, parents, offsets, channels)
    motion = np.asarray(rows, np.float64)
    if declared_frames is not None and motion.shape[0] != declared_frames:
        raise ValueError(f"{path}: file says {declared_frames} frames, found {motion.shape[0]}")
    expected = sum(len(c) for c in channels)
    if motion.shape[1] != expected:
        raise ValueError(f"{path}: {motion.shape[1]} values per frame, expected {expected}")
    return skeleton, motion, frame_time


def forward_kinematics(skeleton, motion):
    """Joint positions in the file's own units, one point per joint per frame.

    Rotation channels are applied in the order the file lists them. A joint that carries
    position channels uses them in place of its rest offset.
    """
    n_frames, n_joints = motion.shape[0], skeleton.n_joints
    spans, start = [], 0
    for c in skeleton.channels:
        spans.append((start, start + len(c)))
        start += len(c)

    world_rotation = np.empty((n_frames, n_joints, 3, 3))
    world_position = np.empty((n_frames, n_joints, 3))
    for j in range(n_joints):
        lo, hi = spans[j]
        values = motion[:, lo:hi]
        translation = np.tile(skeleton.offsets[j], (n_frames, 1)).astype(np.float64)
        order, angles = "", []
        for k, channel in enumerate(skeleton.channels[j]):
            if channel in _POSITION:
                translation[:, _POSITION[channel]] = values[:, k]
            elif channel in _ROTATION:
                order += _ROTATION[channel]
                angles.append(values[:, k])
        if order:
            local = Rotation.from_euler(order, np.stack(angles, axis=1),
                                        degrees=True).as_matrix()
        else:
            local = np.tile(np.eye(3), (n_frames, 1, 1))

        parent = skeleton.parents[j]
        if parent < 0:
            world_rotation[:, j] = local
            world_position[:, j] = translation
        else:
            world_rotation[:, j] = world_rotation[:, parent] @ local
            world_position[:, j] = world_position[:, parent] + np.einsum(
                "tij,tj->ti", world_rotation[:, parent], translation)
    return world_position


def load_positions_m(path):
    """Joint positions in metres, together with the skeleton and the timestep."""
    skeleton, motion, frame_time = parse_bvh(path)
    return forward_kinematics(skeleton, motion) / 100.0, skeleton, frame_time


def main():
    ap = argparse.ArgumentParser(description="Report what one human recording contains.")
    ap.add_argument("bvh", help="a human recording from the collection (.bvh)")
    args = ap.parse_args()
    positions, skeleton, frame_time = load_positions_m(args.bvh)
    print(f"{Path(args.bvh).name}: {skeleton.n_joints} joints, {positions.shape[0]} frames, "
          f"{frame_time} s per frame")
    print(f"joints kept for this study: {len(skeleton.body_joint_indices)}")
    print(f"hips at the first frame (m): {positions[0, skeleton.names.index('Hips')]}")


if __name__ == "__main__":
    main()
