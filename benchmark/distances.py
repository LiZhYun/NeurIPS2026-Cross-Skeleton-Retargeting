"""How far apart two motions on the same skeleton are.

Three ways of measuring it. All three try to ignore differences that should not count, such
as which way the animal happens to be facing or how fast the clip is played.

  procrustes   Compare where the joints are. The two clips are first stretched in time onto
               each other, so that matching moments line up even if one clip is slower. The
               whole body is then rotated and rescaled to sit on top of the other as closely
               as it can. What is left over is the distance. This is the measure the paper
               reports.
  zscore_dtw   Compare the timing. Each body slot from benchmark/slot_encoder.py is put on a
               common scale and the two are stretched in time onto each other, so only the
               shape of the movement over time counts, not its size or place.
  q_component  Compare the clip features of benchmark/clip_features.py: the path of the body
               centre, the contact pattern, the step rate and how the movement is shared
               among the limbs, as four separate differences. The paper calls this Q-comp.

Stretching two sequences in time onto each other is dynamic time warping: it walks along both
clips at once, pairing up frames so the total difference between paired frames is as small as
it can be, and is allowed to linger on a frame of one clip while the other moves on.
"""
from __future__ import annotations

import numpy as np

from benchmark.slot_encoder import SLOT_COUNT, slot_type_to_idx


def _kabsch_3d(A, B):
    """Find the rotation, scale and shift that bring one cloud of points onto another.

    A and B are lists of points in space. The result is the rotation, the scale and the two
    centres for which scale * (A - tA) rotated by R and shifted to tB sits as close to B as it
    can. The rotation is a real rotation: the shape is never mirrored.
    """
    tA = A.mean(axis=0, keepdims=True)
    tB = B.mean(axis=0, keepdims=True)
    A0 = A - tA
    B0 = B - tB
    H = A0.T @ B0  # 3x3
    U, S, Vt = np.linalg.svd(H)
    # Never mirror: make the rotation a proper one
    d = np.sign(np.linalg.det(U @ Vt))
    if d == 0:
        d = 1.0
    D = np.diag([1.0, 1.0, d])
    R = U @ D @ Vt
    # The scale uses the singular values, the last one flipped if a mirror was removed
    S_corr = S.copy()
    S_corr[-1] *= d
    scale = S_corr.sum() / ((A0 ** 2).sum() + 1e-9)
    return R, float(scale), tA, tB


def _dtw_cost(a, b):
    """How different two sequences are once they have been stretched onto each other."""
    T_a, T_b = a.shape[0], b.shape[0]
    if T_a == 0 or T_b == 0:
        return 0.0
    # Vectorized cost matrix [T_a, T_b]
    diff = a[:, None, :] - b[None, :, :]  # [T_a, T_b, D]
    C = np.linalg.norm(diff, axis=-1)
    D = np.full((T_a + 1, T_b + 1), np.inf, dtype=np.float64)
    D[0, 0] = 0.0
    for i in range(1, T_a + 1):
        for j in range(1, T_b + 1):
            D[i, j] = C[i-1, j-1] + min(D[i-1, j-1], D[i-1, j], D[i, j-1])
    return float(D[T_a, T_b] / (T_a + T_b))


def _dtw_align_indices(a, b):
    """Pair up the frames of two sequences so that paired frames match as closely as possible.

    Returns two lists of the same length: the k-th frame of the first list is paired with the
    k-th frame of the second.
    """
    T_a, T_b = a.shape[0], b.shape[0]
    if T_a == 0 or T_b == 0:
        return [], []
    diff = a[:, None, :] - b[None, :, :]
    C = np.linalg.norm(diff, axis=-1)
    D = np.full((T_a + 1, T_b + 1), np.inf, dtype=np.float64)
    D[0, 0] = 0.0
    for i in range(1, T_a + 1):
        for j in range(1, T_b + 1):
            D[i, j] = C[i-1, j-1] + min(D[i-1, j-1], D[i-1, j], D[i, j-1])
    # Backtrack
    path_a, path_b = [], []
    i, j = T_a, T_b
    while i > 0 and j > 0:
        path_a.append(i - 1); path_b.append(j - 1)
        choices = [(D[i-1, j-1], i-1, j-1), (D[i-1, j], i-1, j), (D[i, j-1], i, j-1)]
        choices.sort()
        _, i, j = choices[0]
    return list(reversed(path_a)), list(reversed(path_b))


def dist_zscore_dtw_inv(inv_pred, inv_ref):
    """Timing distance between two motions written in body slots, averaged over the slots."""
    NULL_SLOT = slot_type_to_idx("null")

    def normalize(inv):
        out = np.zeros_like(inv[:, :, 0:3])
        for s in range(SLOT_COUNT):
            if s == NULL_SLOT:
                continue
            traj = inv[:, s, 0:3].copy()
            # A slot this skeleton does not use carries no signal, only noise
            if np.abs(traj).max() < 1e-4:
                continue
            m = traj.mean(axis=0, keepdims=True)
            sd = traj.std(axis=0, keepdims=True)
            # A slot that barely moves would be all noise once rescaled
            if sd.max() < 1e-4:
                continue
            sd = sd + 1e-6  # tiny floor for numerical stability
            out[:, s, :] = (traj - m) / sd
        return out

    norm_a = normalize(inv_pred)
    norm_b = normalize(inv_ref)
    dists = []
    for s in range(SLOT_COUNT):
        if s == NULL_SLOT:
            continue
        ta = norm_a[:, s, :]
        tb = norm_b[:, s, :]
        if np.all(ta == 0) and np.all(tb == 0):
            continue
        dists.append(_dtw_cost(ta, tb))
    return float(np.mean(dists)) if dists else 0.0


def dist_procrustes_trajectory(pos_pred, pos_ref, dtw_align=True,
                                use_root_relative=True):
    """How far the joints of one motion sit from the joints of another, at their best fit.

    Both motions are on the same skeleton, so their joints correspond one to one. Each frame
    is first measured relative to the root joint, so an animal that wanders off does not look
    wrong for that reason alone. The two clips are then lined up in time, the whole body is
    rotated and rescaled onto the other, and what remains is averaged.
    """
    if pos_pred.shape[1] != pos_ref.shape[1]:
        # Both motions should be on the same skeleton; compare what they have in common
        J = min(pos_pred.shape[1], pos_ref.shape[1])
        pos_pred = pos_pred[:, :J]
        pos_ref = pos_ref[:, :J]

    if use_root_relative:
        # Measure every joint relative to the root, so drifting away does not count
        pos_pred = pos_pred - pos_pred[:, :1, :]
        pos_ref = pos_ref - pos_ref[:, :1, :]

    if dtw_align:
        # Line the two clips up in time
        a_flat = pos_pred.reshape(pos_pred.shape[0], -1)
        b_flat = pos_ref.reshape(pos_ref.shape[0], -1)
        idx_a, idx_b = _dtw_align_indices(a_flat, b_flat)
        if not idx_a:
            return float('inf')
        # Collect the paired joint positions of every paired frame
        A = pos_pred[idx_a].reshape(-1, 3)  # [|path|*J, 3]
        B = pos_ref[idx_b].reshape(-1, 3)
    else:
        T = min(pos_pred.shape[0], pos_ref.shape[0])
        A = pos_pred[:T].reshape(-1, 3)
        B = pos_ref[:T].reshape(-1, 3)

    # Rotate and rescale one cloud of points onto the other
    R, scale, tA, tB = _kabsch_3d(A, B)
    A_aligned = scale * (A - tA) @ R + tB
    return float(np.mean(np.linalg.norm(A_aligned - B, axis=1)))


def q_component_distances(feat_pred, feat_ref):
    """Compare two clips on each clip feature separately.

    Four differences are returned, not added up: how far apart the two paths of the body
    centre are, how poorly the two contact patterns agree, how different the step rates are,
    and how differently the movement is shared out over the limbs.
    """
    com_p = np.asarray(feat_pred['com_path'])
    com_r = np.asarray(feat_ref['com_path'])
    T = min(com_p.shape[0], com_r.shape[0])
    com_diff = float(np.linalg.norm(com_p[:T] - com_r[:T]) /
                     (np.linalg.norm(com_r[:T]) + 1e-9))

    cs_p = np.asarray(feat_pred['contact_sched'])
    cs_r = np.asarray(feat_ref['contact_sched'])
    if cs_p.ndim > 1:
        cs_p = (cs_p.sum(axis=1) > 0).astype(np.float32)
    if cs_r.ndim > 1:
        cs_r = (cs_r.sum(axis=1) > 0).astype(np.float32)
    T = min(len(cs_p), len(cs_r))
    cs_p, cs_r = cs_p[:T] > 0.5, cs_r[:T] > 0.5
    tp = float(((cs_p == 1) & (cs_r == 1)).sum())
    fp = float(((cs_p == 1) & (cs_r == 0)).sum())
    fn = float(((cs_p == 0) & (cs_r == 1)).sum())
    pr = tp / (tp + fp + 1e-8)
    rc = tp / (tp + fn + 1e-8)
    cs_f1 = 2 * pr * rc / (pr + rc + 1e-8)

    cad_diff = abs(float(feat_pred['cadence']) - float(feat_ref['cadence']))

    lu_p = -np.sort(-np.asarray(feat_pred['limb_usage']))
    lu_r = -np.sort(-np.asarray(feat_ref['limb_usage']))
    K = max(len(lu_p), len(lu_r))
    lu_p = np.pad(lu_p, (0, K - len(lu_p)))
    lu_r = np.pad(lu_r, (0, K - len(lu_r)))
    lu_l2 = float(np.linalg.norm(lu_p - lu_r))

    return {
        'com_rel_l2': com_diff,
        'cs_one_minus_f1': float(1 - cs_f1),
        'cadence_abs': cad_diff,
        'limb_l2': lu_l2,
    }
