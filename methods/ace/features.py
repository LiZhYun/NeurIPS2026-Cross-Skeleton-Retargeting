"""ACE's per-frame motion feature (NumPy reference).

The feature loss compares source and target motion through a 37-number per-frame
descriptor, body-length normalized wherever it has a length dimension:

  height (1)               root world height, divided by body length
  6D rotation (6)          taken from the root's world rotation matrix, which avoids the
                           quaternion sign ambiguity
  root linear velocity (3) channels 9-11 of the root row, divided by body length
  root angular velocity (3) log_so3(R_t^T @ R_{t+1}); radians, so not normalized
  end-effector positions (24 = 3 x 8) world positions in contact-group order, divided by
                           body length

The loss is the per-frame norm of the difference, averaged over time; averaging the
feature first and taking a norm afterwards would throw the timing away.

References:
  arXiv:2305.14792, sections 5.2-5.3, 6.3 and 7
  core/truebones/motion_process.py, recover_root_quat_and_pos_np
"""
from __future__ import annotations
from pathlib import Path

import numpy as np

from core.truebones.motion_process import (
    recover_from_bvh_ric_np, recover_root_quat_and_pos_np,
)
from methods.common.motion_descriptors import (
    G_MAX, load_contact_groups, select_ee_joints, compute_bone_chain_length, _log_so3_np,
)


def _recover_root_world(motion_13d):
    """Wraps the canonical NumPy root recovery.
    Returns (pos_world [T, 3], R_world [T, 3, 3]). R_world is column-stacked.
    """
    root_row = motion_13d[:, 0, :]
    r_rot_quat, r_pos = recover_root_quat_and_pos_np(root_row)
    R_world = r_rot_quat.transforms()
    return r_pos.astype(np.float64), R_world.astype(np.float64)

D_FEATURE = 1 + 6 + 3 + 3 + 3 * G_MAX   # 1 height + 6 rotation + 3 linear + 3 angular + 24 EE = 37


def matrix_to_rotation_6d_np(R):
    """Inverse of rotation_6d_to_matrix_np (column-stacked convention).

    R: [..., 3, 3]. Returns 6D = first 2 columns flattened: [..., 6].
    """
    return np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1)


def precompute_ace_descriptors(cond_dict, contact_groups_dict):
    """Work out once, per skeleton, what the feature needs: its overall body length and
    which joints are its end effectors.
    Returns dict skeleton -> {body_length, ee_joints, n_ee, parents, offsets}."""
    out = {}
    for skel, c in cond_dict.items():
        offsets = np.asarray(c['offsets'], dtype=np.float64)
        parents = np.asarray(c['parents'], dtype=np.int64)
        cg = contact_groups_dict.get(skel, {})
        ee = select_ee_joints(cg)
        # Body length: sqrt(mean(||offsets[1:]||^2)), the same scale MoReFlow uses.
        offset_norms = np.linalg.norm(offsets[1:], axis=-1)
        body_length = float(np.sqrt(np.mean(offset_norms**2)) + 1e-6)
        # End-effector indices padded to G_MAX
        ee_padded = -np.ones(G_MAX, dtype=np.int64)
        ee_padded[:min(len(ee), G_MAX)] = ee[:G_MAX]
        out[skel] = {
            'body_length': body_length,
            'ee_joints': ee_padded,
            'n_ee': min(len(ee), G_MAX),
            'parents': parents,
            'offsets': offsets,
        }
    return out


def ace_feature_per_frame(motion_13d, body_length, ee_joints, n_ee):
    """Compute the ACE feature per frame on ONE motion window.

    motion_13d: [T, J, 13] np float (physical units, after un-normalizing)
    body_length: float scalar (per skeleton)
    ee_joints: np.int64 [G_MAX] (-1 = unused)
    n_ee: int

    Returns: [T, D_FEATURE=37] np.float64.
    """
    motion_13d = motion_13d.astype(np.float64)
    T, J, _ = motion_13d.shape

    # 1. Recover the root in world space (canonical NumPy reference)
    pos_world_root, R_world_quat = _recover_root_world(motion_13d)       # [T, 3], [T, 3, 3]

    # 2. Height: root world height, body-length normalized
    height = pos_world_root[:, 1:2] / body_length                         # [T, 1]

    # 3. 6D rotation from the world rotation matrix (column convention)
    rot_6d = matrix_to_rotation_6d_np(R_world_quat)                       # [T, 6]

    # 4. Root linear velocity (channels 9, 10, 11), body-length normalized
    lin_vel_local = motion_13d[:, 0, 9:12] / body_length                  # [T, 3]

    # 5. Root angular velocity per frame: log_so3(R_t^T @ R_{t+1})
    R_rel = np.einsum('tji,tjk->tik', R_world_quat[:-1], R_world_quat[1:])  # [T-1, 3, 3]
    ang_vel = _log_so3_np(R_rel)                                          # [T-1, 3]
    ang_vel = np.concatenate([np.zeros((1, 3), dtype=np.float64), ang_vel], axis=0)  # [T, 3]
    # Radians are dimensionless, so this one is not body-length normalized

    # 6. End-effector world positions, body-length normalized, contact-group ordered
    joint_world = recover_from_bvh_ric_np(motion_13d)                     # [T, J, 3]
    ee_flat = np.zeros((T, 3 * G_MAX), dtype=np.float64)
    for g in range(n_ee):
        j = int(ee_joints[g])
        ee_flat[:, g*3 : g*3+3] = joint_world[:, j, :] / body_length

    return np.concatenate([height, rot_6d, lin_vel_local, ang_vel, ee_flat], axis=-1)  # [T, 37]


def compute_feature_loss_np(x_src_phys, x_tgt_phys, src_desc, tgt_desc):
    """Per-frame feature loss: the norm of the feature difference, averaged over time.

    Only the first min(n_ee_src, n_ee_tgt) end effectors are compared; the rest are masked,
    since two skeletons need not have the same number of them.
    """
    psi_src = ace_feature_per_frame(x_src_phys, src_desc['body_length'],
                                    src_desc['ee_joints'], src_desc['n_ee'])  # [T, 37]
    psi_tgt = ace_feature_per_frame(x_tgt_phys, tgt_desc['body_length'],
                                    tgt_desc['ee_joints'], tgt_desc['n_ee'])  # [T, 37]
    n_ee_min = min(src_desc['n_ee'], tgt_desc['n_ee'])
    # The first 13 numbers are always compared; the end-effector block only up to n_ee_min
    mask = np.ones(D_FEATURE, dtype=np.float64)
    mask[13 + 3*n_ee_min:] = 0.0
    diff = (psi_src - psi_tgt) * mask[None, :]
    per_frame_norm = np.linalg.norm(diff, axis=-1)                        # [T]
    return float(per_frame_norm.mean())


if __name__ == '__main__':
    import argparse
    from core.truebones.param_utils import DATASET_DIR

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--skel', default='Horse')
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    data_root = root / DATASET_DIR
    cond = np.load(data_root / 'cond.npy', allow_pickle=True).item()
    cg = load_contact_groups()
    descs = precompute_ace_descriptors(cond, cg)
    desc = descs[args.skel]
    print(f"[{args.skel}] body_length={desc['body_length']:.4f}, n_ee={desc['n_ee']}, "
          f"ee_joints={desc['ee_joints'][:desc['n_ee']]}")

    motion_dir = data_root / 'motions'
    clips = sorted([f for f in motion_dir.iterdir() if f.name.startswith(args.skel + '___')])
    motion = np.load(clips[0]).astype(np.float32)[:32]
    print(f"  Sample clip: {clips[0].name}, shape={motion.shape}")

    psi = ace_feature_per_frame(motion, desc['body_length'], desc['ee_joints'], desc['n_ee'])
    print(f"  Feature per frame: {psi.shape} (expect [32, 37])")
    print(f"  Components of the first frame:")
    print(f"    height={psi[0, 0]:.4f}, rot_6d={psi[0, 1:7]}")
    print(f"    lin_vel={psi[0, 7:10]}, ang_vel={psi[0, 10:13]}")
    print(f"    end effectors (first): {psi[0, 13:16]}")

    # The loss of a motion against itself should be about zero
    loss = compute_feature_loss_np(motion, motion, desc, desc)
    print(f"  feature loss(self, self) = {loss:.6e} (expect ~0)")
