"""Build the small made-up world the other scripts in this folder work on.

Real animal motion gives us no way to check an answer, because nobody knows what the one
correct motion for another animal would have been. This script builds a world where we do
know: eight stick-figure skeletons, six actions, and one rule that turns a clip on one
skeleton into the matching clip on another. Because that rule is written down here, every
method can be measured against the answer it should have given.

The skeletons are chains of 8, 10, 12, 14, 16, 18, 20 or 22 joints, each bone one unit
long, resting along the vertical axis. The six actions are simple joint-angle patterns over
64 frames: walking, fighting, standing still, falling, flying and swimming.

The rule that carries a clip from one skeleton to another spreads the joints of the source
evenly along the chain of the target, and then bends time by an amount that depends only on
which action it is. So the rule is the same for every clip of the same action, which is the
situation this experiment is about.

Four versions of the world are written, crossing two choices:

  how many clips per skeleton and action    one (sparse) or fifty (dense)
  whether true pairs are available          yes (paired) or no (unpaired)

With fifty clips, the clips of one skeleton and action differ in timing and in how large the
movement is, so the action is recognisable but each clip is its own.

Each version is written to its own folder:

  clips.npy       every clip, padded to the largest joint count
  meta.json       what each clip is: skeleton, action, joint count
  transport.npy   the correct target clip for each pair, in the paired versions
  pairs.json      which source clip each of those belongs to

    python -m synthetic.build_dataset
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

K_SKELETONS = 8
JOINTS_PER_SKEL = [8, 10, 12, 14, 16, 18, 20, 22]
T_FRAMES = 64
ACTIONS = ['locomotion', 'combat', 'idle', 'death', 'fly', 'swim']


def make_chain_rest_pose(J: int) -> np.ndarray:
    """[J, 3] chain along the vertical axis, one unit per bone."""
    p = np.zeros((J, 3), dtype=np.float32)
    p[:, 1] = np.arange(J, dtype=np.float32)
    return p


def action_trajectory(action: str, J: int, T: int, instance_seed: int,
                      add_instance_noise: bool) -> np.ndarray:
    """Generate one clip: [T, J, 3] joint positions for one action on a J-joint chain.

    The action name fixes the timing and the shape of the movement. When instance noise is
    on, the clip also gets its own phase and its own amplitude, so two clips of the same
    action are recognisably the same action but not the same motion.
    """
    rng = np.random.RandomState(instance_seed)
    rest = make_chain_rest_pose(J)             # [J, 3]
    pose = np.broadcast_to(rest, (T, J, 3)).astype(np.float32).copy()
    t = np.arange(T, dtype=np.float32) / T     # [T] in [0, 1)

    phase = rng.uniform(0, 2*np.pi) if add_instance_noise else 0.0
    amp = (1.0 + rng.uniform(-0.1, 0.1)) if add_instance_noise else 1.0

    if action == 'locomotion':
        # Every joint swings forward and back together
        delta_x = amp * 0.5 * np.sin(2*np.pi*t + phase)              # [T]
        pose[..., 0] += delta_x[:, None]                              # broadcast over joints
    elif action == 'combat':
        # The far half of the chain snaps from side to side
        delta_x = amp * np.sign(np.sin(4*np.pi*t + phase))            # [T]
        pose[:, J//2:, 0] += 0.6 * delta_x[:, None]
    elif action == 'idle':
        # A small quick wobble sideways
        delta_z = amp * 0.05 * np.sin(8*np.pi*t + phase)              # [T]
        pose[..., 2] += delta_z[:, None]
    elif action == 'death':
        # The chain sinks steadily towards the ground
        decay = amp * (1.0 - t)                                       # [T] from 1 to 0
        for j in range(J):
            pose[:, j, 1] = j * decay
    elif action == 'fly':
        # A large wave runs along the chain in two directions at once
        for j in range(J):
            pose[:, j, 0] += amp * 0.4 * np.sin(2*np.pi*t + phase + j*0.3)
            pose[:, j, 2] += amp * 0.4 * np.cos(2*np.pi*t + phase + j*0.3)
    elif action == 'swim':
        # The far half moves against the near half
        for j in range(J):
            sign = 1.0 if j < J//2 else -1.0
            pose[:, j, 0] += amp * sign * 0.3 * np.sin(2*np.pi*t + phase)
    else:
        raise ValueError(f"Unknown action: {action}")
    return pose


def true_transport(x_a: np.ndarray, J_b: int, action: str) -> np.ndarray:
    """The correct target clip for a source clip: spread the joints, then bend time.

    x_a: [T, J_a, 3]. Returns [T, J_b, 3].

    The joints of the source are spread evenly along the target's chain by straight-line
    interpolation. Time is then bent by an amount that depends only on the action, so every
    clip of one action is carried across in the same way.
    """
    T, J_a, _ = x_a.shape
    t_grid = np.arange(T, dtype=np.float32) / max(1, T-1)                # source time in [0, 1]
    if action == 'locomotion':   u = t_grid                              # unchanged
    elif action == 'combat':     u = t_grid                              # unchanged
    elif action == 'idle':       u = t_grid                              # unchanged
    elif action == 'death':      u = np.power(t_grid, 1.5)               # starts slowly
    elif action == 'fly':        u = t_grid                              # unchanged
    elif action == 'swim':       u = 0.5*(1 - np.cos(np.pi*t_grid))      # slow at both ends
    else: raise ValueError(action)
    src_idx = np.clip((u * (T-1)).astype(np.int64), 0, T-1)
    x_a_warped = x_a[src_idx]                                            # [T, J_a, 3]

    src_chain = np.linspace(0, 1, J_a, dtype=np.float32)
    tgt_chain = np.linspace(0, 1, J_b, dtype=np.float32)
    x_b = np.zeros((T, J_b, 3), dtype=np.float32)
    for d in range(3):
        for ti in range(T):
            x_b[ti, :, d] = np.interp(tgt_chain, src_chain, x_a_warped[ti, :, d])
    return x_b


def build_cell(out_dir: Path, density: str, supervision: str, seed: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    n_per_cell = 1 if density == 'sparse' else 50

    clips = []
    meta = []
    instance_id = 0
    for s_idx, J_s in enumerate(JOINTS_PER_SKEL):
        for a_idx, action in enumerate(ACTIONS):
            for k in range(n_per_cell):
                cs = seed * 1000003 + s_idx * 1009 + a_idx * 53 + k * 7
                clip = action_trajectory(action, J_s, T_FRAMES, cs,
                                         add_instance_noise=(n_per_cell > 1))
                clips.append(clip)
                meta.append({'instance_id': instance_id, 'skel': s_idx,
                             'action': a_idx, 'action_name': action,
                             'n_joints': J_s, 'cell_count': n_per_cell})
                instance_id += 1

    # Pad every clip to the largest joint count so they can be stacked into one array
    J_max = max(JOINTS_PER_SKEL)
    arr = np.zeros((len(clips), T_FRAMES, J_max, 3), dtype=np.float32)
    for i, clip in enumerate(clips):
        arr[i, :, :clip.shape[1], :] = clip
    np.save(out_dir / 'clips.npy', arr)

    if supervision == 'paired':
        # Give each source clip one target skeleton (sparse) or five (dense), and write the
        # correct target clip for each.
        rng = np.random.RandomState(seed + 42)
        pair_records = []
        pair_arr_list = []
        for src_idx, item in enumerate(meta):
            sa, ai = item['skel'], item['action']
            n_pairs_per_src = 1 if density == 'sparse' else min(5, K_SKELETONS - 1)
            for _ in range(n_pairs_per_src):
                sb = rng.choice([k for k in range(K_SKELETONS) if k != sa])
                xa = clips[src_idx]
                xb_target = true_transport(xa, JOINTS_PER_SKEL[sb], ACTIONS[ai])
                pair_records.append({
                    'src_instance_id': src_idx, 'skel_a': sa,
                    'skel_b': int(sb), 'action': ai,
                })
                pad = np.zeros((T_FRAMES, J_max, 3), dtype=np.float32)
                pad[:, :xb_target.shape[1], :] = xb_target
                pair_arr_list.append(pad)
        pair_arr = np.stack(pair_arr_list)
        np.save(out_dir / 'transport.npy', pair_arr)
        with open(out_dir / 'pairs.json', 'w') as f:
            json.dump(pair_records, f, indent=2)
        print(f"  paired transport: {pair_arr.shape}, {len(pair_records)} pair records")

    with open(out_dir / 'meta.json', 'w') as f:
        json.dump({
            'density': density, 'supervision': supervision,
            'seed': seed, 'n_clips': len(clips),
            'n_skeletons': K_SKELETONS, 'joints_per_skel': JOINTS_PER_SKEL,
            'n_actions': len(ACTIONS), 'actions': ACTIONS, 'T_frames': T_FRAMES,
            'J_max': J_max, 'clips': meta,
        }, f, indent=2)

    print(f"[build_cell] {density}_{supervision}: clips={arr.shape}, n={len(clips)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=str, default='save/synthetic_2x2',
                        help='where to write the four versions, relative to the repository root')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    out_root = ROOT / args.out
    out_root.mkdir(parents=True, exist_ok=True)

    cells = [
        ('sparse', 'unpaired'),
        ('sparse', 'paired'),
        ('dense', 'unpaired'),
        ('dense', 'paired'),
    ]
    for density, supervision in cells:
        cell_dir = out_root / f'{density}_{supervision}'
        build_cell(cell_dir, density, supervision, args.seed)

    print(f"\nAll 4 versions written to {out_root}")


if __name__ == '__main__':
    main()
