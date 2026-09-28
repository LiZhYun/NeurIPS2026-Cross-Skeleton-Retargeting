"""Motion2Motion, working straight on the motion files. The paper calls this the direct variant.

Motion2Motion (Chen et al., arXiv:2508.13139) copies a motion onto a new animal by example.
It is given the motion to copy, one clip of the target animal to take its style from, and a
handful of bones that are declared to correspond, such as this animal's left front foot and
that one's. It then builds the answer a short window at a time, repeatedly replacing each
window with the window of the example clip it resembles most, working from a coarse version of
the motion up to the full one.

The authors' own program reads and writes motion files in the BVH format;
methods/motion2motion/generate_bvh.py runs that program. This file runs the same algorithm
directly on the clips as the dataset stores them, which is what the paper reports as the
direct variant.

Three differences from the published method, all of them noted in the paper:

  - the dataset stores thirteen numbers per joint per frame, covering position, rotation,
    velocity and ground contact, rather than the rotations and root position the authors'
    format carries. The extra numbers make matching noisier, not cleaner, so this is not a
    setting that flatters the method;
  - the corresponding bones are read off the contact groups of each skeleton by matching group
    names, rather than being chosen by hand, with fall-backs for pairs whose groups do not
    line up at all, such as a snake and a two-legged animal;
  - how strongly the source motion is imposed, and the sequence of coarse-to-fine resolutions,
    are fixed rather than adjusted as the algorithm proceeds.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from core.truebones.param_utils import DATASET_DIR

ROOT = Path(__file__).resolve().parents[2]
CONTACT_GROUPS_PATH = ROOT / 'benchmark/contact_groups.json'

PATCH_SIZE = 11
NUM_STEPS = 3
NOISE_SIGMA = 1.0  # the paper uses 10.0, which suits the numbers in its own format; the
                   # numbers in this one are smaller to begin with
COARSE_RATIO = 0.4  # the coarsest version is this fraction of the full length
PYR_FACTOR = 0.75
MATCHING_ALPHA = 0.9


def load_assets():
    """Load the skeleton descriptions, the contact groups and the path to the clips."""
    cond = np.load(os.path.join(DATASET_DIR, 'cond.npy'), allow_pickle=True).item()
    with open(CONTACT_GROUPS_PATH) as f:
        contact_groups = json.load(f)
    return cond, contact_groups, os.path.join(DATASET_DIR, 'motions')


# ---------------------------------------------------------------------------------------
# Finding, for each short window of one motion, the window of another that resembles it most
# ---------------------------------------------------------------------------------------
def extract_patches(x, patch_size, stride=1, loop=False):
    """Cut a motion into overlapping windows of a few frames each."""
    B, C, T = x.shape
    if loop:
        # loop padding
        x = torch.cat([x, x[:, :, :patch_size - 1]], dim=-1)
    patches = x.unfold(-1, patch_size, stride)  # [B, C, N, patch_size]
    N = patches.shape[2]
    patches = patches.permute(0, 2, 1, 3).reshape(B, N, C * patch_size)
    return patches


def combine_patches(x_shape, patches, patch_size, stride=1, loop=False):
    """Put the windows back together into one motion, averaging where they overlap."""
    B, C, T = x_shape
    N = patches.shape[1]
    patches = patches.view(B, N, C, patch_size).permute(0, 2, 3, 1)  # [B, C, patch_size, N]
    out = torch.zeros(B, C, T, device=patches.device, dtype=patches.dtype)
    count = torch.zeros(B, 1, T, device=patches.device, dtype=patches.dtype)
    for i in range(patch_size):
        # patches[:, :, i, :] contributes to frames [i, i+stride, ...]
        idx = torch.arange(N, device=patches.device) * stride + i
        # valid idx < T
        valid = idx < T
        if loop:
            idx = idx % T
            valid = torch.ones_like(valid)
        if not valid.any():
            continue
        v_idx = idx[valid]
        v_vals = patches[:, :, i, valid]
        out.index_add_(-1, v_idx, v_vals)
        count.index_add_(-1, v_idx, torch.ones_like(v_vals[:, :1]))
    return out / count.clamp_min(1e-6)


def efficient_cdist(X, Y, chunk=1024):
    """Distance from every window of one set to every window of the other."""
    # ||x-y||^2 = ||x||^2 + ||y||^2 - 2 x.y
    X2 = (X ** 2).sum(-1, keepdim=True)
    Y2 = (Y ** 2).sum(-1, keepdim=True).t()
    out = torch.empty(X.shape[0], Y.shape[0], device=X.device)
    for i in range(0, X.shape[0], chunk):
        Xi = X[i:i+chunk]
        Xi2 = X2[i:i+chunk]
        out[i:i+chunk] = (Xi2 + Y2 - 2 * Xi @ Y.t()).clamp_min(0.0).sqrt()
    return out


def get_NNs_Dists(dist_fn, X_patches, Y_patches, alpha):
    """For each window of the first set, find the window of the second it resembles most.

    Distances are first divided by how close each candidate is to its own best match, so that
    a window of the example that is a good match for everything does not get reused for
    everything.
    """
    D = dist_fn(X_patches, Y_patches)  # [Nx, Ny]
    if alpha is not None and alpha > 0:
        # Discount candidates that match everything equally well
        d_min = D.min(dim=0, keepdim=True).values + alpha  # [1, Ny]
        D = D / d_min
    nnf = D.argmin(dim=1)
    return nnf, D.gather(1, nnf.unsqueeze(-1)).squeeze(-1)


# ---------------------------------------------------------------------------------------
# Deciding which bones of the two animals correspond
# ---------------------------------------------------------------------------------------
def author_sparse_mapping(src_skel, tgt_skel, cond, contact_groups, max_pairs=6):
    """Decide which bones of the two animals correspond, and say in words what was paired.

    The two roots are always paired. After that, in order of preference:

      - groups the two animals have in common, such as a left front foot on both sides,
        paired at the outermost joint of each;
      - failing that, a left group with any left group and a right with any right;
      - failing that, the groups of the two animals paired off in the order they are listed,
        which is the best that can be done for, say, a spider and a two-legged animal.

    If even that yields nothing, one joint partway down each skeleton is paired, so that the
    algorithm has something to work with.
    """
    src_g = contact_groups.get(src_skel, {})
    tgt_g = contact_groups.get(tgt_skel, {})
    src_joints = cond[src_skel]['joints_names']
    tgt_joints = cond[tgt_skel]['joints_names']

    pairs = [(0, 0)]  # root-root
    description = [f'root({src_joints[0]})↔root({tgt_joints[0]})']

    # Body parts both animals have
    shared = [k for k in src_g if k in tgt_g and not k.startswith('_')]
    for g in shared:
        if len(pairs) >= max_pairs:
            break
        src_idx = src_g[g][-1]  # most distal joint in group
        tgt_idx = tgt_g[g][-1]
        pairs.append((src_idx, tgt_idx))
        description.append(f'{g}:{src_joints[src_idx]}↔{tgt_joints[tgt_idx]}')

    if len(pairs) >= 3:
        return pairs, description

    # Failing that, match left to left and right to right
    src_L = [k for k in src_g if k.startswith('L') and not k.startswith('_')]
    src_R = [k for k in src_g if k.startswith('R') and not k.startswith('_')]
    tgt_L = [k for k in tgt_g if k.startswith('L') and not k.startswith('_')]
    tgt_R = [k for k in tgt_g if k.startswith('R') and not k.startswith('_')]

    for s_groups, t_groups, side in [(src_L, tgt_L, 'left side'), (src_R, tgt_R, 'right side')]:
        for sg in s_groups:
            if len(pairs) >= max_pairs:
                break
            if not t_groups:
                continue
            tg = t_groups[0]  # pick first available
            src_idx = src_g[sg][-1]
            tgt_idx = tgt_g[tg][-1]
            existing = [(s, t) for (s, t) in pairs]
            if (src_idx, tgt_idx) in existing:
                continue
            pairs.append((src_idx, tgt_idx))
            description.append(f'{side}:{src_joints[src_idx]}↔{tgt_joints[tgt_idx]}')

    if len(pairs) >= 3:
        return pairs[:max_pairs], description[:max_pairs]

    # Failing that, pair the groups off in the order they are listed
    tgt_all = list(tgt_g.values())
    for i, (g_name, s_idxs) in enumerate(src_g.items()):
        if len(pairs) >= max_pairs:
            break
        if i >= len(tgt_all):
            break
        s_idx = s_idxs[-1]
        t_idx = tgt_all[i][-1]
        if (s_idx, t_idx) not in pairs:
            pairs.append((s_idx, t_idx))
            description.append(f'in listed order:{src_joints[s_idx]}↔{tgt_joints[t_idx]}')

    if len(pairs) < 2:
        # Nothing lined up at all, so pair one joint partway down each skeleton
        s_far = min(len(src_joints) - 1, len(src_joints) // 2)
        t_far = min(len(tgt_joints) - 1, len(tgt_joints) // 2)
        pairs.append((s_far, t_far))
        description.append(f'partway down each skeleton:{src_joints[s_far]}↔{tgt_joints[t_far]}')

    return pairs[:max_pairs], description[:max_pairs]


# ---------------------------------------------------------------------------------------
# Building the answer, coarse version first
# ---------------------------------------------------------------------------------------
def retarget(source_motion, example_motion, src_j_idxs, tgt_j_idxs, tgt_n_joints,
             device='cpu', seed=42):
    """Copy a motion onto the target animal, using one of its clips as the example to follow.

    The paired bones carry the source motion across. The rest of the target animal's body is
    filled in from the example clip, window by window, so that it keeps moving in a way that
    animal plausibly could. The answer has the same length as the source motion.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    J_t = tgt_n_joints
    assert example_motion.shape[1] == J_t
    assert len(src_j_idxs) == len(tgt_j_idxs)

    # The whole example clip, used to fill in the bones that were not paired
    ex_full = example_motion                        # [T_e, J_t, 13]

    # The sequence of lengths to work at, coarsest first
    T_s = source_motion.shape[0]
    T_e = example_motion.shape[0]
    final_len = T_s

    lengths = [max(PATCH_SIZE + 2, int(round(final_len * COARSE_RATIO)))]
    while lengths[-1] < final_len:
        nxt = int(round(lengths[-1] / PYR_FACTOR))
        if nxt <= lengths[-1]:
            nxt = lengths[-1] + 1
        lengths.append(nxt)
    lengths[-1] = final_len

    # Start from the example clip, stretched to the coarsest length
    def interp_1d(x, new_T):
        # x: [T, J, C] -> [new_T, J, C]
        x_t = torch.tensor(x, device=device, dtype=torch.float32)
        x_t = x_t.permute(1, 2, 0).unsqueeze(0)  # [1, J, C, T]
        J, C = x_t.shape[1], x_t.shape[2]
        x_t = x_t.reshape(1, J*C, x_t.shape[-1])
        x_t = F.interpolate(x_t, size=new_T, mode='linear', align_corners=True)
        return x_t.reshape(1, J, C, new_T).squeeze(0).permute(2, 0, 1).cpu().numpy()

    synth = interp_1d(ex_full, lengths[0])  # [L0, J_t, 13] initial
    # Start from something rough rather than the example itself
    rng = np.random.RandomState(seed)
    noise = rng.randn(*synth.shape).astype(np.float32) * NOISE_SIGMA * np.std(synth)
    synth = synth + noise

    for lvl, L in enumerate(lengths):
        # Stretch what we have to this length
        if lvl > 0:
            synth = interp_1d(synth, L)

        # The example at a matching length
        lvl_ratio = L / final_len
        L_e = max(PATCH_SIZE + 2, min(T_e, int(round(T_e * lvl_ratio * 1.2))))
        ex_full_lvl = interp_1d(ex_full, L_e)             # [L_e, J_t, 13]
        ex_bnd_lvl = ex_full_lvl[:, tgt_j_idxs, :]        # [L_e, B, 13]

        # Impose the source motion on the paired bones
        src_lvl = interp_1d(source_motion, L)              # [L, J_s, 13]
        src_bnd_lvl = src_lvl[:, src_j_idxs, :]            # [L, B, 13]

        synth[:, tgt_j_idxs, :] = MATCHING_ALPHA * src_bnd_lvl + (1 - MATCHING_ALPHA) * synth[:, tgt_j_idxs, :]

        # Replace each window with its closest match in the example, a few times over
        for step in range(NUM_STEPS):
            # Windows of the paired bones
            s_ten = torch.tensor(synth[:, tgt_j_idxs, :], device=device, dtype=torch.float32)
            s_ten = s_ten.permute(1, 2, 0).reshape(1, len(tgt_j_idxs) * 13, L)
            e_ten = torch.tensor(ex_bnd_lvl, device=device, dtype=torch.float32)
            e_ten = e_ten.permute(1, 2, 0).reshape(1, len(tgt_j_idxs) * 13, L_e)

            x_patches = extract_patches(s_ten, PATCH_SIZE, 1, loop=False)  # [1, N_s, D]
            y_patches = extract_patches(e_ten, PATCH_SIZE, 1, loop=False)  # [1, N_e, D]

            nnf, _ = get_NNs_Dists(efficient_cdist, x_patches.squeeze(0), y_patches.squeeze(0), 0.01)
            matched = y_patches[:, nnf, :]  # [1, N_s, D]
            combined = combine_patches(s_ten.shape, matched, PATCH_SIZE, 1, loop=False)
            # Write the matched windows back
            combined_np = combined.reshape(1, len(tgt_j_idxs), 13, L).squeeze(0).permute(2, 0, 1).cpu().numpy()
            synth[:, tgt_j_idxs, :] = combined_np

            # Do the same for the whole body, so the bones that were not paired keep moving
            # together with the rest
            s_ten_full = torch.tensor(synth, device=device, dtype=torch.float32)
            s_ten_full = s_ten_full.permute(1, 2, 0).reshape(1, J_t * 13, L)
            e_ten_full = torch.tensor(ex_full_lvl, device=device, dtype=torch.float32)
            e_ten_full = e_ten_full.permute(1, 2, 0).reshape(1, J_t * 13, L_e)
            x_p_full = extract_patches(s_ten_full, PATCH_SIZE, 1, loop=False)
            y_p_full = extract_patches(e_ten_full, PATCH_SIZE, 1, loop=False)

            # Judge the match on the paired bones alone
            bnd_channels = []
            for b in tgt_j_idxs:
                for c in range(13):
                    bnd_channels.append(b * 13 + c)
            bnd_channels_t = torch.tensor(bnd_channels, device=device, dtype=torch.long)

            x_bnd_view = x_p_full.reshape(1, x_p_full.shape[1], J_t * 13, PATCH_SIZE)
            y_bnd_view = y_p_full.reshape(1, y_p_full.shape[1], J_t * 13, PATCH_SIZE)
            x_bnd_view = x_bnd_view.index_select(2, bnd_channels_t).reshape(1, x_p_full.shape[1], -1)
            y_bnd_view = y_bnd_view.index_select(2, bnd_channels_t).reshape(1, y_p_full.shape[1], -1)

            nnf_full, _ = get_NNs_Dists(efficient_cdist,
                                         x_bnd_view.squeeze(0), y_bnd_view.squeeze(0), 0.01)
            matched_full = y_p_full[:, nnf_full, :]
            combined_full = combine_patches(s_ten_full.shape, matched_full, PATCH_SIZE, 1, loop=False)
            synth = combined_full.reshape(1, J_t, 13, L).squeeze(0).permute(2, 0, 1).cpu().numpy()

            # Put the source motion back on the paired bones
            synth[:, tgt_j_idxs, :] = MATCHING_ALPHA * src_bnd_lvl + (1 - MATCHING_ALPHA) * synth[:, tgt_j_idxs, :]

    if synth.shape[0] != final_len:
        synth = interp_1d(synth, final_len)
    return synth.astype(np.float32)
