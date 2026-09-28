"""Clip features: a short description of what a clip does, free of the body that does it.

Two animals with different numbers of legs cannot be compared joint by joint, but they can be
compared on what their movement achieves. This module measures five such things for a clip:

  - the path the centre of the body travels,
  - how fast it moves forward, in its own frame of reference,
  - which limbs are touching the ground at each moment,
  - how many steps it takes per second,
  - how the movement is shared out among the limbs.

Sizes are divided out, so a mouse and an elephant walking the same way come out alike. The
retrieval baselines rank candidate clips by these numbers, and the action-level AUC evaluator
offers them as one of its measures (the paper calls that measure Q-comp).

Which joints of a skeleton can touch the ground, and which limb each belongs to, is read from
benchmark/contact_groups.json.

The features of every clip are measured once from the processed Truebones data and stored in
save/clip_features.npz, which ANCHOR, the random baselines, the AL-Flow models and the
action-level test read. To make that file, run this from the repository root (it takes a few
minutes):

    python -m benchmark.clip_features --build
"""
import argparse
import json
import os
from os.path import join as pjoin
from pathlib import Path

import numpy as np

from core.truebones.param_utils import DATASET_DIR

ROOT = Path(__file__).resolve().parents[1]
CONTACT_GROUPS_PATH = ROOT / 'benchmark/contact_groups.json'
CLIP_FEATURES_PATH = ROOT / 'save/clip_features.npz'
# The paper's whole-body step rate for the clips in which several step rates are equally strong
RHYTHM_TIES = json.loads((ROOT / 'benchmark/rhythm_ties.json').read_text())


def load_contact_groups(path=CONTACT_GROUPS_PATH):
    """Read the contact groups of every skeleton, leaving out the notes at the top of the file."""
    with open(path) as f:
        raw = json.load(f)
    return {k: v for k, v in raw.items() if not k.startswith('_')}


def load_cond(cond_path=None):
    if cond_path is None:
        cond_path = pjoin(DATASET_DIR, 'cond.npy')
    return np.load(cond_path, allow_pickle=True).item()


def load_motion_positions(fname, motion_dir=None, n_joints=None):
    """Read a clip and return where its joints are and which of them touch the ground."""
    from core.truebones.motion_process import recover_from_bvh_ric_np
    if motion_dir is None:
        motion_dir = pjoin(DATASET_DIR, 'motions')
    m = np.load(pjoin(motion_dir, fname))  # [T, J, 13]
    if n_joints is not None and m.shape[1] > n_joints:
        m = m[:, :n_joints]
    contacts = (m[..., 12] > 0.5).astype(np.int8)  # [T, J]
    positions = recover_from_bvh_ric_np(m.astype(np.float32))  # [T, J, 3]
    return positions, contacts


def open_clip_features(path=CLIP_FEATURES_PATH):
    """Open the stored features of every clip, or stop with a note on how to make them."""
    if not Path(path).exists():
        raise SystemExit(f"The clip features are missing ({path}). Make them once from the "
                         f"processed Truebones data with:\n"
                         f"    python -m benchmark.clip_features --build")
    return np.load(path, allow_pickle=True)


def load_clip_features(path=CLIP_FEATURES_PATH):
    """Open the stored features and return a function that looks a clip up by name."""
    qc = open_clip_features(path)
    fname_to_idx = {m['fname']: i for i, m in enumerate(qc['meta'])}

    def get_features(fname):
        i = fname_to_idx.get(fname)
        if i is None:
            return None
        return {
            'com_path': qc['com_path'][i], 'heading_vel': qc['heading_vel'][i],
            'contact_sched': qc['contact_sched'][i],
            'cadence': float(qc['cadence'][i]), 'limb_usage': qc['limb_usage'][i],
        }
    return get_features


def canonical_body_frame(positions, subtree_masses, smoothing=0.9):
    """Find where the body is and which way it faces, in every frame.

    The centre of the body is a weighted average of the joints, heavier where more of the
    animal hangs below. Forward is the smoothed direction that centre is travelling in along
    the ground; an animal that is standing still keeps the direction it last had, or, if it
    never moves, the long axis of its body.
    """
    T, J, _ = positions.shape
    w = subtree_masses / (subtree_masses.sum() + 1e-8)

    centroids = np.einsum('j,tjd->td', w, positions)  # [T, 3]

    # Facing direction: where the body centre is drifting along the ground, smoothed
    vel = np.zeros_like(centroids)
    vel[1:] = centroids[1:] - centroids[:-1]
    vel[0] = vel[1]
    vel_horiz = vel.copy()
    vel_horiz[:, 1] = 0  # zero out vertical

    forward = np.zeros((T, 3))
    forward[0] = vel_horiz[0]
    for t in range(1, T):
        forward[t] = smoothing * forward[t-1] + (1 - smoothing) * vel_horiz[t]

    norms = np.linalg.norm(forward, axis=1, keepdims=True)
    # Standing still: fall back to the long axis of the body
    static = norms.squeeze() < 1e-6
    if static.all():
        rest_centered = positions[0] - centroids[0]
        _, _, Vt = np.linalg.svd(rest_centered, full_matrices=False)
        fallback = Vt[0]
        fallback[1] = 0
        fallback = fallback / (np.linalg.norm(fallback) + 1e-8)
        forward[:] = fallback
    else:
        forward[~static] = forward[~static] / norms[~static]
        if static.any():
            last_good = forward[~static][0]
            for t in range(T):
                if static[t]:
                    forward[t] = last_good
                else:
                    last_good = forward[t]

    up = np.array([0.0, 1.0, 0.0])
    lateral = np.cross(up, forward)
    lat_norms = np.linalg.norm(lateral, axis=1, keepdims=True)
    lateral = lateral / (lat_norms + 1e-8)

    R = np.stack([forward, lateral, np.tile(up, (T, 1))], axis=-1)  # [T, 3, 3]

    return centroids, R


def compute_subtree_masses(parents, offsets):
    J = len(parents)
    bone_lengths = np.linalg.norm(offsets, axis=1)
    masses = bone_lengths.copy()
    for j in reversed(range(J)):
        p = parents[j]
        if 0 <= p < J and p != j:
            masses[p] += masses[j]
    return masses


def body_scale(offsets):
    """How big the animal is, used to divide sizes out of everything else."""
    # The total bone length stands in for the size of the animal. Only its ordering matters:
    # a bigger animal has to come out bigger.
    return float(np.linalg.norm(offsets, axis=1).sum() + 1e-6)


def compute_heading_velocity(centroids, R, fps=30):
    """How fast the body is moving in the direction it faces, frame by frame."""
    T = centroids.shape[0]
    vel = np.zeros_like(centroids)
    vel[1:] = (centroids[1:] - centroids[:-1]) * fps
    vel[0] = vel[1]
    # R columns: [forward, lateral, up]. Project onto forward axis.
    forward_speed = np.einsum('td,td->t', vel, R[:, :, 0])
    return forward_speed


def compute_contact_schedule_aggregate(contacts):
    """What fraction of the joints touch the ground in each frame."""
    T, J = contacts.shape
    return contacts.sum(axis=1).astype(np.float32) / max(J, 1)


def compute_contact_schedule_grouped(contacts, groups):
    """How much of each limb touches the ground in each frame, one column per limb."""
    T, J = contacts.shape
    names = sorted(groups.keys())
    C = len(names)
    sched = np.zeros((T, C), dtype=np.float32)
    for i, name in enumerate(names):
        idxs = [j for j in groups[name] if 0 <= j < J]
        if idxs:
            sched[:, i] = contacts[:, idxs].mean(axis=1)
    return sched, names


def compute_cadence(contact_schedule, fps=30, min_cycle=0.25, max_cycle=4.0):
    """How many steps per second, read off the rhythm of the footfalls.

    Rates outside a quarter of a step per second to four steps per second are ignored as
    implausible for an animal.
    """
    if contact_schedule.ndim > 1:
        s = contact_schedule.sum(axis=1)
    else:
        s = contact_schedule
    s = s - s.mean()
    if s.std() < 1e-6:
        return 0.0
    T = len(s)
    fft = np.fft.rfft(s)
    freqs = np.fft.rfftfreq(T, d=1.0 / fps)
    # Keep only plausible step rates
    lo = 1.0 / max_cycle  # e.g. 0.25 Hz
    hi = 1.0 / min_cycle  # e.g. 4 Hz
    mask = (freqs >= lo) & (freqs <= hi)
    if not mask.any():
        return 0.0
    power = np.abs(fft) ** 2
    power_mask = power[mask]
    freq_mask = freqs[mask]
    peak_idx = np.argmax(power_mask)
    return float(freq_mask[peak_idx])


def compute_limb_usage(positions, kinematic_chains, fps=30):
    """What share of the movement each limb accounts for. The shares add up to one."""
    T, J, _ = positions.shape
    vel = np.zeros_like(positions)
    vel[1:] = (positions[1:] - positions[:-1]) * fps
    vel[0] = vel[1]
    ke = 0.5 * (vel ** 2).sum(axis=-1)  # [T, J]
    ke_total_per_joint = ke.mean(axis=0)  # [J]
    K = len(kinematic_chains)
    energy = np.zeros(K)
    for k, chain in enumerate(kinematic_chains):
        idxs = [j for j in chain if 0 <= j < J]
        if idxs:
            energy[k] = ke_total_per_joint[idxs].mean()
    total = energy.sum() + 1e-12
    return energy / total


def extract_clip_features(fname, cond, contact_groups=None, n_joints=None,
                          motion_dir=None, fps=30):
    """Measure the five features of one stored clip.

    Returns the path of the body centre, the forward speed, which limbs are on the ground in
    each frame, the number of steps per second, the share of the movement taken by each limb,
    and the size of the animal that the first two were divided by.
    """
    parents = cond['parents']
    offsets = cond['offsets']
    chains = cond['kinematic_chains']
    J = offsets.shape[0]

    positions, contacts = load_motion_positions(fname, motion_dir=motion_dir, n_joints=J)
    T = positions.shape[0]

    subtree = compute_subtree_masses(parents, offsets)
    centroids, R = canonical_body_frame(positions, subtree)

    scale = body_scale(offsets)

    # Where the body centre travels, with size divided out
    com_path = (centroids - centroids[0:1]) / scale  # starts at zero in the first frame

    # How fast it goes forward, with size divided out
    heading_vel = compute_heading_velocity(centroids, R, fps=fps) / scale

    # Which limbs are on the ground, frame by frame
    if contact_groups is not None and cond['object_type'] in contact_groups:
        sched, names = compute_contact_schedule_grouped(contacts, contact_groups[cond['object_type']])
    else:
        sched = compute_contact_schedule_aggregate(contacts)
        names = None

    # How many steps per second. Where several step rates are equally strong, which one comes
    # out on top depends on the computer's rounding, so the paper's choice is stored for those clips.
    cadence = RHYTHM_TIES.get(fname) if names is None else None
    if cadence is None:
        cadence = compute_cadence(sched, fps=fps)

    # How the movement is shared out among the limbs
    limb_usage = compute_limb_usage(positions, chains, fps=fps)

    return {
        'com_path': com_path.astype(np.float32),
        'heading_vel': heading_vel.astype(np.float32),
        'contact_sched': sched.astype(np.float32),
        'contact_group_names': names,
        'cadence': float(cadence),
        'limb_usage': limb_usage.astype(np.float32),
        'body_scale': float(scale),
        'n_joints': int(J),
        'n_frames': int(T),
    }


def extract_features_from_array(motion, cond, contact_groups_for_skel, fps=30):
    """Measure the five features of a motion held in memory rather than a stored clip."""
    from core.truebones.motion_process import recover_from_bvh_ric_np

    parents = cond['parents']
    offsets = cond['offsets']
    chains = cond['kinematic_chains']
    J = offsets.shape[0]

    if motion.shape[1] > J:
        motion = motion[:, :J]
    contacts = (motion[..., 12] > 0.5).astype(np.int8)
    positions = recover_from_bvh_ric_np(motion.astype(np.float32))
    T = positions.shape[0]

    subtree = compute_subtree_masses(parents, offsets)
    centroids, R = canonical_body_frame(positions, subtree)
    scale = body_scale(offsets)
    com_path = (centroids - centroids[0:1]) / scale
    heading_vel = compute_heading_velocity(centroids, R, fps=fps) / scale

    if contact_groups_for_skel is not None:
        sched, names = compute_contact_schedule_grouped(contacts, contact_groups_for_skel)
    else:
        sched = compute_contact_schedule_aggregate(contacts)
        names = None

    cadence = compute_cadence(sched, fps=fps)
    limb_usage = compute_limb_usage(positions, chains, fps=fps)

    return {
        'com_path': com_path.astype(np.float32),
        'heading_vel': heading_vel.astype(np.float32),
        'contact_sched': sched.astype(np.float32),
        'contact_group_names': names,
        'cadence': float(cadence),
        'limb_usage': limb_usage.astype(np.float32),
        'body_scale': float(scale),
    }


def feature_signature(entry, k_limb=5):
    """Boil one clip's features down to a single vector, so clips can be compared.

    Clips differ in length, so each feature is summarised by a few numbers of its own: how far
    the body travelled and how fast, how much of the time it was on the ground, its step rate,
    and the share taken by its busiest limbs.
    """
    if isinstance(entry, np.ndarray) and entry.dtype == object:
        q = entry.item() if entry.ndim == 0 else entry
    else:
        q = entry

    # Body centre: distance travelled, top speed, how far it ranged in each direction,
    # and its average height
    com = np.asarray(q['com_path'])
    if com.size == 0:
        com_sum = np.zeros(6)
    else:
        disp = com[-1] - com[0]
        vel = np.diff(com, axis=0) if com.shape[0] > 1 else np.zeros((1, 3))
        peak_speed = np.linalg.norm(vel, axis=-1).max()
        com_sum = np.array([
            np.linalg.norm(disp),
            peak_speed,
            np.ptp(com[:, 0]),
            np.ptp(com[:, 1]),
            np.ptp(com[:, 2]),
            com[:, 1].mean(),  # mean height
        ])

    # Forward speed: average, spread, fastest and slowest
    hv = np.asarray(q['heading_vel'])
    hv_sum = np.array([hv.mean(), hv.std(), hv.max(), hv.min()]) if hv.size > 0 else np.zeros(4)

    # Contacts: average, spread, share of frames mostly in contact, and range
    cs = np.asarray(q['contact_sched'])
    if cs.ndim > 1:
        cs = cs.mean(axis=1)  # collapse to aggregate
    if cs.size > 0:
        cs_sum = np.array([cs.mean(), cs.std(), (cs > 0.5).mean(), np.ptp(cs)])
    else:
        cs_sum = np.zeros(4)

    # Steps per second
    cad = float(q['cadence'])

    # Share of the movement taken by the busiest limbs
    lu = np.asarray(q['limb_usage'])
    if lu.size > 0:
        top = -np.sort(-lu)[:k_limb]
        if top.size < k_limb:
            top = np.concatenate([top, np.zeros(k_limb - top.size)])
    else:
        top = np.zeros(k_limb)

    return np.concatenate([com_sum, hv_sum, cs_sum, [cad], top]).astype(np.float32)


def skeleton_of_clip(fname, skeletons):
    """The skeleton a processed clip belongs to, read from the start of its file name."""
    matches = [s for s in skeletons if fname.startswith(s + '_')]
    return max(matches, key=len) if matches else None


def build_cache(out_path, max_clips=None):
    """Measure the features of every processed clip and store them together.

    Clips are stored in order of their file names; everything that reads the file looks clips
    up by name.
    """
    cond = load_cond()
    motion_dir = pjoin(DATASET_DIR, 'motions')
    meta = []
    for fname in sorted(f for f in os.listdir(motion_dir) if f.endswith('.npy')):
        skeleton = skeleton_of_clip(fname, cond)
        if skeleton is None:
            print(f"  skip {fname}: no skeleton of that name in cond.npy")
            continue
        meta.append({'fname': fname, 'skeleton': skeleton,
                     'n_joints': int(cond[skeleton]['offsets'].shape[0])})

    if max_clips:
        meta = meta[:max_clips]

    all_com_path = []  # clips differ in length, so these are stored as object arrays
    all_heading_vel = []
    all_contact_sched = []
    all_cadence = []
    all_limb_usage = []
    kept_meta = []
    for i, m in enumerate(meta):
        if i % 100 == 0:
            print(f"  {i}/{len(meta)}")
        if m['skeleton'] not in cond:
            continue
        try:
            q = extract_clip_features(m['fname'], cond[m['skeleton']], motion_dir=motion_dir)
        except Exception as e:
            print(f"  skip {m['fname']}: {e}")
            continue
        all_com_path.append(q['com_path'])
        all_heading_vel.append(q['heading_vel'])
        all_contact_sched.append(q['contact_sched'])
        all_cadence.append(q['cadence'])
        all_limb_usage.append(q['limb_usage'])
        kept_meta.append(m)

    np.savez(
        out_path,
        com_path=np.array(all_com_path, dtype=object),
        heading_vel=np.array(all_heading_vel, dtype=object),
        contact_sched=np.array(all_contact_sched, dtype=object),
        cadence=np.array(all_cadence, dtype=np.float32),
        limb_usage=np.array(all_limb_usage, dtype=object),
        meta=np.array(kept_meta, dtype=object),
    )
    print(f"Saved: {out_path}  ({len(kept_meta)} clips)")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--build', action='store_true',
                   help='measure the features of every processed clip and store them')
    p.add_argument('--out', default=str(CLIP_FEATURES_PATH))
    p.add_argument('--max_clips', type=int, default=None,
                   help='measure only the first few clips, to try the command out')
    args = p.parse_args()
    if args.build:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        build_cache(args.out, max_clips=args.max_clips)
    else:
        p.print_help()


if __name__ == '__main__':
    main()
