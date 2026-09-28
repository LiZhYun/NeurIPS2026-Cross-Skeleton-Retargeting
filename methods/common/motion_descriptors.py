"""A few measurements taken from a short piece of motion.

Two of the methods need to say what a motion is doing without saying which animal is doing
it. These five measurements are how they say it. MoReFlow uses them twice: to decide which
source clip goes with which target clip during training, and to state what is wanted when
generating.

  root_vel   how fast the body travels and turns, averaged over the window
  EE_local   where each foot or wing tip sits relative to the body, divided by the length
             of the limb it is on, so a long-legged animal and a short-legged one are
             comparable
  EE_world   where each foot or wing tip sits in the world
  root_XY    how far across the ground the body moves during the window
  root_Z     how high the body is held

A foot or wing tip is the last joint of a contact group, which is a limb or body segment
that can touch the ground; benchmark/contact_groups.json lists them per skeleton. Each
measurement is returned padded to the same length, so the model always receives the same
shaped input.

The motion format stores 13 numbers per joint per frame, which makes most of these direct
reads:

  [0:3]   position relative to the body, already in the body's own frame for every joint
          but the root
  [3:9]   rotation
  [9:12]  velocity in the body's own frame
  [12]    whether the joint is touching the ground

so root_vel's straight-line part is the average of the root's channels 9 to 11, EE_local is
the stored position of the end joints, and root_Z is the average of root channel 1. Only
EE_world and root_XY need the motion to be rebuilt in world space.

methods/common/motion_descriptors_torch.py computes the same numbers in a form a model can
be trained through. The two must agree, and the trainers check that before they start.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

from core.truebones.motion_process import recover_from_bvh_ric_np
from core.anytop.utils.rotation_conversions import rotation_6d_to_matrix_np

CONDITIONS = ['root_vel', 'EE_local', 'EE_world', 'root_XY', 'root_Z']
G_MAX = 8                      # max end-effector groups per skeleton (padded)
D_COND_PADDED = 24             # universal pad-to size for the cond_vec model input

CONTACT_GROUPS_PATH = Path(__file__).resolve().parents[2] / 'benchmark/contact_groups.json'


# -------------------------------------------------------------------- helpers


def load_contact_groups():
    with open(CONTACT_GROUPS_PATH) as f:
        raw = json.load(f)
    return {k: v for k, v in raw.items() if not k.startswith('_')}


def select_ee_joints(skel_contact_groups):
    if not skel_contact_groups:
        return np.array([], dtype=np.int64)
    ee_indices = []
    for gname in sorted(skel_contact_groups.keys()):
        joint_list = skel_contact_groups[gname]
        if joint_list:
            ee_indices.append(joint_list[-1])  # deepest = last
    return np.array(ee_indices, dtype=np.int64)


def compute_bone_chain_length(offsets, parents, ee_joint_idx, anchor_idx=0):
    path = []
    j = int(ee_joint_idx)
    seen = set()
    while j != anchor_idx and j != -1:
        if j in seen:
            break
        path.append(j)
        seen.add(j)
        j = int(parents[j])
    return float(sum(np.linalg.norm(offsets[k]) for k in path) + 1e-6)


def precompute_skel_descriptors(cond_dict, contact_groups_dict):
    out = {}
    for skel, c in cond_dict.items():
        cg = contact_groups_dict.get(skel, {})
        ee = select_ee_joints(cg)
        bls = np.ones(G_MAX, dtype=np.float64)
        ee_padded = -np.ones(G_MAX, dtype=np.int64)
        for i, j in enumerate(ee[:G_MAX]):
            ee_padded[i] = j
            bls[i] = compute_bone_chain_length(c['offsets'], c['parents'], j, anchor_idx=0)
        out[skel] = {
            'ee_joints': ee_padded,
            'bone_lengths': bls,
            'parents': np.asarray(c['parents'], dtype=np.int64),
            'offsets': np.asarray(c['offsets'], dtype=np.float64),
            'n_ee': min(len(ee), G_MAX),
        }
    return out


# -------------------------------------------------------------------- log SO(3)


def _log_so3_np(R):
    """Log map of SO(3) to axis-angle. R: [..., 3, 3] -> [..., 3]."""
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_theta = np.clip((trace - 1.0) / 2.0, -1.0 + 1e-7, 1.0 - 1e-7)
    theta = np.arccos(cos_theta)
    sin_theta = np.sin(theta)
    small = np.abs(sin_theta) < 1e-5
    factor = np.where(small, 0.5 + theta**2 / 12.0, theta / (2.0 * sin_theta + 1e-20))
    skew_x = R[..., 2, 1] - R[..., 1, 2]
    skew_y = R[..., 0, 2] - R[..., 2, 0]
    skew_z = R[..., 1, 0] - R[..., 0, 1]
    return np.stack([factor * skew_x, factor * skew_y, factor * skew_z], axis=-1)


# -------------------------------------------------------------------- descriptors


def phi(motion_13d, c_type, skel_desc):
    """Take one of the five measurements on a single motion window.

    motion_13d: [T, J, 13] np float
    c_type: one of CONDITIONS
    Returns: 1-D np.float64 array of length D_COND_PADDED, with the descriptor's own
    numbers at the front (root_vel: 6, EE_local/world: 24, root_XY: 2, root_Z: 1).
    """
    motion_13d = motion_13d.astype(np.float64)
    T = motion_13d.shape[0]
    ee_idx = skel_desc['ee_joints']
    n_ee = skel_desc['n_ee']
    bone_lengths = skel_desc['bone_lengths']

    if c_type == 'root_vel':
        # Linear: channels 9,10,11 of the root row are root-aligned-frame velocity.
        # Per the encoding, features = [pos[:-1], rot[:-1], vel, foot], all of length T-1,
        # so every channel is aligned and all frames are valid.
        lin_vel = motion_13d[:, 0, 9:12].mean(axis=0)                    # [3]
        # Angular: per-frame root rotation matrix; relative rotation R_t^T @ R_{t+1}
        R_world = rotation_6d_to_matrix_np(motion_13d[:, 0, 3:9])         # [T, 3, 3]
        R_rel = np.einsum('tji,tjk->tik', R_world[:-1], R_world[1:])      # R_t^T @ R_{t+1}
        ang_vel = _log_so3_np(R_rel).mean(axis=0)                         # [3]
        out = np.zeros(D_COND_PADDED, dtype=np.float64)
        out[:6] = np.concatenate([lin_vel, ang_vel])
        return out

    elif c_type == 'EE_local':
        # Channels 0:3 are already in the root-anchored frame.
        out = np.zeros(D_COND_PADDED, dtype=np.float64)
        for g in range(n_ee):
            j = int(ee_idx[g])
            ee_pos = motion_13d[:, j, :3].mean(axis=0) / bone_lengths[g]  # [3]
            out[g*3 : g*3 + 3] = ee_pos
        return out

    elif c_type == 'EE_world':
        joint_world = recover_from_bvh_ric_np(motion_13d)                 # [T, J, 3]
        out = np.zeros(D_COND_PADDED, dtype=np.float64)
        for g in range(n_ee):
            j = int(ee_idx[g])
            out[g*3 : g*3 + 3] = joint_world[:, j, :].mean(axis=0)
        return out

    elif c_type == 'root_XY':
        joint_world = recover_from_bvh_ric_np(motion_13d)                 # [T, J, 3]
        root_xy = joint_world[:, 0, [0, 2]]                                # [T, 2]
        out = np.zeros(D_COND_PADDED, dtype=np.float64)
        out[:2] = (root_xy - root_xy[0:1]).mean(axis=0)
        return out

    elif c_type == 'root_Z':
        out = np.zeros(D_COND_PADDED, dtype=np.float64)
        out[0] = motion_13d[:, 0, 1].mean()
        return out

    else:
        raise ValueError(f"Unknown c_type: {c_type}")


# -------------------------------------------------------------------- self-test


if __name__ == '__main__':
    import argparse
    from core.truebones.param_utils import DATASET_DIR

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--skel', default='Horse')
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    data_root = root / DATASET_DIR
    cond = np.load(data_root / 'cond.npy', allow_pickle=True).item()
    contact_groups = load_contact_groups()
    skel_descs = precompute_skel_descriptors(cond, contact_groups)
    desc = skel_descs[args.skel]
    print(f"[{args.skel}] n_ee={desc['n_ee']}, "
          f"ee_joints={desc['ee_joints'][:desc['n_ee']]}, "
          f"bone_lengths={desc['bone_lengths'][:desc['n_ee']]}")

    motion_dir = data_root / 'motions'
    clips = sorted([f for f in motion_dir.iterdir() if f.name.startswith(args.skel + '___')])
    motion = np.load(clips[0]).astype(np.float32)[:32]
    print(f"  Sample clip: {clips[0].name}, shape={motion.shape}")

    for c in CONDITIONS:
        v = phi(motion, c, desc)
        print(f"  {c}: shape={v.shape}, mean={v.mean():.4f}, "
              f"std={v.std():.4f}, max={np.abs(v).max():.4f}")
