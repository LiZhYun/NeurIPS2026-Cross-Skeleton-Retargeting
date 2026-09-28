"""More data does not restore the variation a squared-error model throws away.

A model trained to minimise squared error is trained to answer with an average. When many
different source clips share one description, the average it settles on is the same for all
of them, so its answers stop depending on which clip was asked about. That is the
conditional-mean effect.

The natural objection is that this is only a shortage of data: give the model more clips for
each combination of source skeleton, target skeleton and action, and the effect should
disappear. This script tests that objection in the made-up world of build_dataset.py, where
the correct answer is known. It trains the same small model again and again with two, four,
eight, sixteen, thirty-two and fifty clips per combination. Each time it then asks the model
about source clips it never saw, and measures how much the answers vary from one source clip
to the next, against how much the correct answers vary.

A ratio of one would mean the model keeps all the variation that belongs to the source. The
ratio stays far below one at every rung of the ladder.

The --pairing option chooses which target each source clip is trained against:

- random_same_group (the default, and the setting the paper reports): at every training step,
  every source clip is trained against the correct target of a different source clip, picked at
  random from the same combination of source skeleton, target skeleton and action, and picked
  again at the next step. This is the setting of the paper's conditional-mean proposition: the
  model sees targets of the right skeleton and action, but never the one that belongs to the
  clip it was given. It needs at least two clips per combination.
- true_pairs: every source clip is trained against its own correct target, for comparison.

The paper's table comes from seeds 42, 43 and 44; with the seed fixed, a rerun repeats them
exactly.

    python -m synthetic.conditional_mean_ladder --seed 42
    python -m synthetic.conditional_mean_ladder --seed 42 --pairing true_pairs
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from synthetic.build_dataset import (
    K_SKELETONS, JOINTS_PER_SKEL, ACTIONS, T_FRAMES, action_trajectory, true_transport,
)

ROOT = Path(__file__).resolve().parents[1]

DESCRIPTION = (
    'A model trained to minimise squared error on true pairs learns the average answer, not '
    'the one that belongs to a particular source clip. In the same made-up world this file '
    'trains such a model with more and more true pairs per cell and records how much of the '
    'true variation its predictions keep. More pairs do not bring the variation back.'
)
RANDOM_PAIRING_DESCRIPTION = (
    'The same ladder, but at every training step each source clip is trained against the '
    'correct target of a different, randomly chosen source clip of the same source skeleton, '
    'target skeleton and action.'
)
PAIRINGS = ('true_pairs', 'random_same_group')


class TinyRegressor(nn.Module):
    """A small network that answers with a target clip directly, trained on squared error.

    It is given the source clip, the source and target skeletons and the action, and it
    returns a target clip. Average pooling, skeleton and action embeddings and a small stack
    of linear layers are enough here, because the correct answer in this world depends only
    on the source clip and those three labels.
    """
    def __init__(self, j_max, t_frames, n_skels, n_actions, d_emb=32, hidden=128):
        super().__init__()
        self.j_max = j_max
        self.t_frames = t_frames
        self.skel_emb = nn.Embedding(n_skels, d_emb)
        self.action_emb = nn.Embedding(n_actions, d_emb)
        flat = j_max * t_frames * 3
        self.encoder = nn.Sequential(
            nn.Linear(flat, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden + 3 * d_emb, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, flat),
        )

    def forward(self, x_a, sa, sb, action):
        b = x_a.shape[0]
        x = x_a.view(b, -1)
        h = self.encoder(x)
        cond = torch.cat([self.skel_emb(sa), self.skel_emb(sb), self.action_emb(action)], -1)
        out = self.head(torch.cat([h, cond], -1))
        return out.view(b, self.t_frames, self.j_max, 3)


def build_paired_data(n_pairs_per_cell: int, j_max: int, seed: int):
    """Make the training pairs: this many distinct source clips per combination.

    Each source clip is given exactly one target, the correct one.
    """
    src_list = []
    tgt_list = []
    sa_list = []
    sb_list = []
    a_list = []
    for sa in range(K_SKELETONS):
        for sb in range(K_SKELETONS):
            if sa == sb:
                continue
            for ai, action in enumerate(ACTIONS):
                for k in range(n_pairs_per_cell):
                    cs = seed * 100003 + sa * 1009 + sb * 53 + ai * 7 + k * 3
                    xa = action_trajectory(action, JOINTS_PER_SKEL[sa], T_FRAMES, cs,
                                           add_instance_noise=True)
                    xb = true_transport(xa, JOINTS_PER_SKEL[sb], action)
                    pad_a = np.zeros((T_FRAMES, j_max, 3), dtype=np.float32)
                    pad_a[:, :xa.shape[1], :] = xa
                    pad_b = np.zeros((T_FRAMES, j_max, 3), dtype=np.float32)
                    pad_b[:, :xb.shape[1], :] = xb
                    src_list.append(pad_a)
                    tgt_list.append(pad_b)
                    sa_list.append(sa)
                    sb_list.append(sb)
                    a_list.append(ai)
    return (
        np.stack(src_list), np.stack(tgt_list),
        np.array(sa_list), np.array(sb_list), np.array(a_list),
    )


def build_held_out_eval(n_eval_per_cell: int, j_max: int, seed: int):
    """Make the questions: source clips the training never saw, drawn the same way."""
    by_cell = {}
    for sa in range(K_SKELETONS):
        for sb in range(K_SKELETONS):
            if sa == sb:
                continue
            for ai, action in enumerate(ACTIONS):
                samples = []
                for k in range(n_eval_per_cell):
                    cs = (seed + 9999) * 100003 + sa * 1009 + sb * 53 + ai * 7 + k * 3
                    xa = action_trajectory(action, JOINTS_PER_SKEL[sa], T_FRAMES, cs,
                                           add_instance_noise=True)
                    xb = true_transport(xa, JOINTS_PER_SKEL[sb], action)
                    pad_a = np.zeros((T_FRAMES, j_max, 3), dtype=np.float32)
                    pad_a[:, :xa.shape[1], :] = xa
                    pad_b = np.zeros((T_FRAMES, j_max, 3), dtype=np.float32)
                    pad_b[:, :xb.shape[1], :] = xb
                    samples.append((pad_a, JOINTS_PER_SKEL[sa], pad_b, JOINTS_PER_SKEL[sb]))
                by_cell[(sa, sb, ai)] = samples
    return by_cell


def train_and_measure(n_pairs_per_cell: int, max_steps: int, batch_size: int,
                      device: str, seed: int, pairing: str = 'true_pairs'):
    """Train at one rung of the ladder and measure how much its answers vary."""
    j_max = max(JOINTS_PER_SKEL)
    # The seed fixes the starting weights as well as the data, so a rerun repeats exactly.
    torch.manual_seed(seed)
    src, tgt, sa, sb, ai = build_paired_data(n_pairs_per_cell, j_max, seed)
    held_eval = build_held_out_eval(n_eval_per_cell=8, j_max=j_max, seed=seed)
    print(f"  {n_pairs_per_cell} pairs per cell: {len(src)} training pairs, "
          f"{len(held_eval)} cells of questions")

    dev = torch.device(device)
    model = TinyRegressor(j_max, T_FRAMES, K_SKELETONS, len(ACTIONS)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    src_t = torch.from_numpy(src).to(dev)
    tgt_t = torch.from_numpy(tgt).to(dev)
    sa_t = torch.from_numpy(sa).long().to(dev)
    sb_t = torch.from_numpy(sb).long().to(dev)
    ai_t = torch.from_numpy(ai).long().to(dev)
    n = len(src)
    rng = np.random.RandomState(seed + 1)
    # build_paired_data stores the clips of one cell next to each other, so clip i belongs to
    # the cell that starts at i - i % M. The partner draws use their own random stream, so the
    # batches are the same under both pairings.
    m = n_pairs_per_cell
    shuffle_targets = pairing == 'random_same_group'
    partner_rng = np.random.RandomState(seed + 2)
    t0 = time.time()
    for step in range(max_steps):
        idx = rng.choice(n, size=min(batch_size, n), replace=False)
        tgt_idx = idx
        if shuffle_targets:
            # Move each clip by 1 to M-1 places within its cell: a different clip, uniformly.
            shift = partner_rng.randint(1, m, size=len(idx))
            tgt_idx = idx - idx % m + (idx % m + shift) % m
        x_pred = model(src_t[idx], sa_t[idx], sb_t[idx], ai_t[idx])
        loss = F.mse_loss(x_pred, tgt_t[tgt_idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % max(1, max_steps // 4) == 0:
            print(f"    step {step+1}/{max_steps}  loss={loss.item():.4f}  ({time.time()-t0:.0f}s)")

    model.eval()
    out_vars = []
    src_vars = []
    oracle_vars = []
    output_minus_oracle_mse = []
    cell_count = 0
    with torch.no_grad():
        for (sa_c, sb_c, ai_c), samples in held_eval.items():
            if len(samples) < 2:
                continue
            preds = []
            sources = []
            oracles = []
            for pad_a, real_J_a, pad_b_oracle, real_J_b in samples:
                xa_t = torch.from_numpy(pad_a).unsqueeze(0).to(dev)
                pred = model(xa_t,
                             torch.tensor([sa_c], device=dev).long(),
                             torch.tensor([sb_c], device=dev).long(),
                             torch.tensor([ai_c], device=dev).long())
                pred_np = pred[0, :, :real_J_b, :].cpu().numpy()  # cut to the real joints
                preds.append(pred_np)
                sources.append(pad_a[:, :real_J_a, :])
                oracles.append(pad_b_oracle[:, :real_J_b, :])
            # How much each set varies across the source clips of this cell
            pred_arr = np.stack(preds)        # [n_src, T, J_b, 3]
            src_arr = np.stack(sources)       # [n_src, T, J_a, 3]
            oracle_arr = np.stack(oracles)    # [n_src, T, J_b, 3]
            out_vars.append(float(np.mean(pred_arr.var(axis=0))))
            src_vars.append(float(np.mean(src_arr.var(axis=0))))
            oracle_vars.append(float(np.mean(oracle_arr.var(axis=0))))
            mse = float(np.mean((pred_arr - oracle_arr) ** 2))
            output_minus_oracle_mse.append(mse)
            cell_count += 1
    out_var = float(np.mean(out_vars)) if out_vars else float('nan')
    src_var = float(np.mean(src_vars)) if src_vars else float('nan')
    oracle_var = float(np.mean(oracle_vars)) if oracle_vars else float('nan')
    return {
        'pairs_per_cell': n_pairs_per_cell,
        'same_as_true_pairs': not shuffle_targets,
        'n_cells': cell_count,
        'predicted_variance_per_cell': out_var,
        'true_map_variance_per_cell': oracle_var,
        'source_variance_per_cell': src_var,
        'variance_ratio': out_var / max(oracle_var, 1e-9),
        'mean_squared_error': (float(np.mean(output_minus_oracle_mse))
                               if output_minus_oracle_mse else float('nan')),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', default=None,
                   help='where to write the result; the default names the file after the seed')
    p.add_argument('--pairs_per_cell', type=int, nargs='+',
                   default=[2, 4, 8, 16, 32, 50],
                   help='the rungs of the ladder')
    p.add_argument('--max_steps', type=int, default=2500,
                   help='training steps at each rung; the paper used 2500')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--pairing', choices=PAIRINGS, default='random_same_group',
                   help='true_pairs: each source clip is trained against its own correct '
                        'target; random_same_group: against the correct target of another '
                        'source clip of the same cell, drawn again at every step')
    args = p.parse_args()
    if args.pairing == 'random_same_group' and min(args.pairs_per_cell) < 2:
        p.error('random_same_group needs at least two clips per cell')

    name = ('conditional_mean_ladder' if args.pairing == 'random_same_group' else
            'conditional_mean_ladder_true_pairs')
    out_path = (Path(args.out) if args.out else
                ROOT / 'save/synthetic_2x2' / f'{name}_seed{args.seed}.json')

    print(f"device={args.device}  pairing={args.pairing}  ladder={args.pairs_per_cell}  "
          f"max_steps={args.max_steps}")
    rows = []
    for M in args.pairs_per_cell:
        print(f"\n=== {M} pairs per cell ===")
        row = train_and_measure(M, args.max_steps, args.batch_size, args.device, args.seed,
                                args.pairing)
        rows.append(row)
        print(f"  predicted variance={row['predicted_variance_per_cell']:.5f}  "
              f"true variance={row['true_map_variance_per_cell']:.5f}  "
              f"ratio={row['variance_ratio']:.4f}  error={row['mean_squared_error']:.4f}")

    out = {
        'description': (DESCRIPTION if args.pairing == 'true_pairs' else
                        RANDOM_PAIRING_DESCRIPTION),
        'settings': {
            'pairing': args.pairing,
            'max_steps': args.max_steps, 'batch_size': args.batch_size,
            'seed': args.seed, 'pairs_per_cell_values': args.pairs_per_cell,
        },
        'rows': rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1, default=float))
    print(f"\nsaved {out_path}")
    print("\nHow much of the true variation survives, by number of pairs per cell:")
    for r in rows:
        print(f"  {r['pairs_per_cell']:4d} pairs  ratio={r['variance_ratio']:.4f}  "
              f"error={r['mean_squared_error']:.4f}")


if __name__ == '__main__':
    main()
