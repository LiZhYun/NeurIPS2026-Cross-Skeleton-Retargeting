"""Train a small generator on each version of the made-up world and see what it recovers.

The four versions differ in how many clips there are per skeleton and action, and in whether
true pairs are available. One small generator is trained on each, then asked for the target
clip of source clips it never saw. Because the correct answer is known, the error can be
measured, and the four numbers say which of the two difficulties matters: too few clips to
tell apart, or no way to tell which target belongs to which source.

  sparse, unpaired   one clip per skeleton and action, no true pairs
  sparse, paired     one clip per skeleton and action, true pairs available
  dense, unpaired    fifty clips per skeleton and action, no true pairs
  dense, paired      fifty clips per skeleton and action, true pairs available

The generator is a small transformer that starts from noise and moves towards a motion,
told which skeleton and which action it is aiming for. Where true pairs are available it is
also asked to match the paired target directly.

Half the clips of each skeleton and action are held back from training and used to ask the
question. With one clip per skeleton and action there is nothing to hold back, so every clip
is used as a question instead, and what is being tested is whether the answer carries across
to skeleton pairs the training never showed.

    python -m synthetic.train_and_evaluate --all
    python -m synthetic.train_and_evaluate --cell dense_paired
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
    JOINTS_PER_SKEL, ACTIONS, T_FRAMES, K_SKELETONS, true_transport,
)

ROOT = Path(__file__).resolve().parents[1]


class SmallFlowGen(nn.Module):
    """A small transformer that predicts which way a motion should move at each moment.

    x_t: [B, T, J_max, 3] the motion part way between noise and a real clip
    t:   [B]              how far along that journey it is, from 0 to 1
    skel_id, action_id: [B] which skeleton and which action are being asked for
    Returns v: [B, T, J_max, 3], the direction to keep moving in.
    """
    def __init__(self, T=T_FRAMES, J_max=22, d_model=256, n_layers=6, n_heads=8,
                 n_skels=K_SKELETONS, n_actions=len(ACTIONS)):
        super().__init__()
        self.T = T; self.J_max = J_max; self.d_model = d_model
        self.proj_in = nn.Linear(J_max * 3, d_model)
        self.t_emb = nn.Sequential(nn.Linear(1, d_model), nn.SiLU(),
                                   nn.Linear(d_model, d_model))
        self.skel_emb = nn.Embedding(n_skels, d_model)
        self.action_emb = nn.Embedding(n_actions, d_model)
        self.pos_emb = nn.Parameter(torch.randn(T, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*2,
            dropout=0.0, batch_first=True, norm_first=True, activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.proj_out = nn.Linear(d_model, J_max * 3)

    def forward(self, x_t, t, skel_id, action_id):
        B, T, J, _ = x_t.shape
        h = self.proj_in(x_t.reshape(B, T, J*3))         # [B, T, d]
        h = h + self.pos_emb.unsqueeze(0)
        cond = (self.t_emb(t.unsqueeze(-1)) + self.skel_emb(skel_id) +
                self.action_emb(action_id)).unsqueeze(1)  # [B, 1, d]
        h = h + cond
        h = self.encoder(h)
        v = self.proj_out(h).reshape(B, T, J, 3)
        return v


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def split_train_eval(meta, eval_frac=0.5, seed=0):
    """Hold back half the clips of each skeleton and action. Returns two lists of indices.

    A skeleton and action with only one clip keeps it for training, because holding it back
    would leave that combination untrained.
    """
    rng = np.random.RandomState(seed)
    by_cell = {}
    for i, m in enumerate(meta):
        key = (m['skel'], m['action'])
        by_cell.setdefault(key, []).append(i)
    train_idx, eval_idx = [], []
    for key, idxs in by_cell.items():
        if len(idxs) == 1:
            train_idx.extend(idxs)
        else:
            shuffled = idxs.copy(); rng.shuffle(shuffled)
            n_eval = max(1, int(len(idxs) * eval_frac))
            eval_idx.extend(shuffled[:n_eval])
            train_idx.extend(shuffled[n_eval:])
    return train_idx, eval_idx


def set_seed(seed):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def train_one_cell(cell_dir: Path, max_steps=2000, batch_size=32, lr=1e-3,
                   w_paired=1.0, device='cuda', seed=0, log_interval=200):
    """Train the generator on one version of the world. Returns it with its bookkeeping."""
    print(f"\n=== Training on: {cell_dir.name} ===")
    set_seed(seed)
    dev = torch.device(device)

    clips = np.load(cell_dir / 'clips.npy')             # [N, T, J_max, 3]
    meta = json.load(open(cell_dir / 'meta.json'))['clips']
    print(f"  clips: {clips.shape}, n={len(meta)}")

    train_idx, eval_idx = split_train_eval(meta, eval_frac=0.5, seed=seed)
    print(f"  train={len(train_idx)}, held back={len(eval_idx)}")

    has_pairs = (cell_dir / 'transport.npy').exists()
    if has_pairs:
        transport = np.load(cell_dir / 'transport.npy')   # [N_pairs, T, J_max, 3]
        pairs = json.load(open(cell_dir / 'pairs.json'))
        # Only pairs whose source clip is in the training half may be used
        train_set = set(train_idx)
        pair_idx = [pi for pi, p in enumerate(pairs) if p['src_instance_id'] in train_set]
        print(f"  true pairs with a training source: {len(pair_idx)} / {len(pairs)}")
    else:
        transport = None; pairs = None; pair_idx = []

    G = SmallFlowGen().to(dev)
    print(f"  model parameters: {count_params(G):,}")
    opt = torch.optim.AdamW(G.parameters(), lr=lr, weight_decay=0.01)

    clips_t = torch.from_numpy(clips).float().to(dev)
    if has_pairs:
        transport_t = torch.from_numpy(transport).float().to(dev)
    skel_arr = torch.tensor([m['skel'] for m in meta], dtype=torch.long, device=dev)
    action_arr = torch.tensor([m['action'] for m in meta], dtype=torch.long, device=dev)

    rng = np.random.RandomState(seed + 1)
    losses = []
    t0 = time.time()
    G.train()
    for step in range(max_steps):
        opt.zero_grad()
        # Learn to move from noise towards a real clip of the training half
        idx = rng.choice(len(train_idx), batch_size)
        ids = [train_idx[i] for i in idx]
        x1 = clips_t[ids]                                # [B, T, J_max, 3]
        x0 = torch.randn_like(x1)
        q = torch.rand(x1.shape[0], device=dev)
        x_t = (1 - q.view(-1, 1, 1, 1)) * x0 + q.view(-1, 1, 1, 1) * x1
        v_target = x1 - x0
        v_pred = G(x_t, q, skel_arr[ids], action_arr[ids])
        l_flow = F.mse_loss(v_pred, v_target)

        # Where true pairs are available, also learn to land on the paired target
        if has_pairs and len(pair_idx) > 0 and w_paired > 0:
            pidx = rng.choice(len(pair_idx), batch_size)
            actual_pids = [pair_idx[i] for i in pidx]
            src_ids = [pairs[pi]['src_instance_id'] for pi in actual_pids]
            tgt_skels = torch.tensor([pairs[pi]['skel_b'] for pi in actual_pids],
                                     dtype=torch.long, device=dev)
            tgt_actions = torch.tensor([pairs[pi]['action'] for pi in actual_pids],
                                       dtype=torch.long, device=dev)
            x_a = clips_t[src_ids]
            x_b_target = transport_t[actual_pids]
            noise = torch.randn_like(x_b_target)
            t_zero = torch.zeros(noise.shape[0], device=dev)
            v_zero = G(noise, t_zero, tgt_skels, tgt_actions)
            x_b_hat = noise + v_zero
            l_paired = F.mse_loss(x_b_hat, x_b_target)
            loss = l_flow + w_paired * l_paired
        else:
            l_paired = torch.zeros(1, device=dev).squeeze()
            loss = l_flow

        loss.backward()
        torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
        opt.step()
        losses.append((float(l_flow.item()), float(l_paired.item())))
        if (step + 1) % log_interval == 0:
            recent = losses[-log_interval:]
            avg_flow = np.mean([l[0] for l in recent])
            avg_paired = np.mean([l[1] for l in recent])
            print(f"  [{step+1:5d}/{max_steps}] flow={avg_flow:.4f} paired={avg_paired:.4f} "
                  f"({time.time()-t0:.0f}s)")

    return G, meta, eval_idx, has_pairs


@torch.no_grad()
def eval_recovery(G, meta, eval_idx, clips, n_eval_samples=100, n_steps=20,
                  device='cuda', seed=0):
    """Ask the generator for target clips of held-back sources and measure the error.

    Each question picks a held-back source clip and a different skeleton to carry it to. The
    answer is compared against the one the rule in build_dataset.py gives.
    """
    print(f"\n=== Measuring recovery on {n_eval_samples} questions ===")
    G.eval()
    dev = torch.device(device)
    rng = np.random.RandomState(seed + 10)

    errors_by_action = {a: [] for a in range(len(ACTIONS))}
    n_tested = 0
    for trial in range(n_eval_samples * 2):
        src = eval_idx[rng.choice(len(eval_idx))]
        sa = meta[src]['skel']
        ai = meta[src]['action']
        sb_choices = [k for k in range(K_SKELETONS) if k != sa]
        sb = int(sb_choices[rng.choice(len(sb_choices))])

        x_a = clips[src][:, :JOINTS_PER_SKEL[sa], :]      # [T, J_a, 3], real joints only
        x_b_true = true_transport(x_a, JOINTS_PER_SKEL[sb], ACTIONS[ai])

        J_max = clips.shape[2]
        noise = torch.randn(1, T_FRAMES, J_max, 3, device=dev)
        z = noise
        sb_t = torch.tensor([sb], dtype=torch.long, device=dev)
        ai_t = torch.tensor([ai], dtype=torch.long, device=dev)
        for k in range(n_steps):
            t_k = torch.full((1,), k / n_steps, device=dev)
            v = G(z, t_k, sb_t, ai_t)
            z = z + v * (1.0 / n_steps)
        x_b_hat = z[0, :, :JOINTS_PER_SKEL[sb], :].cpu().numpy()  # [T, J_b, 3]

        err = float(np.mean((x_b_hat - x_b_true) ** 2))
        errors_by_action[ai].append(err)
        n_tested += 1
        if n_tested >= n_eval_samples:
            break

    flat = [e for v in errors_by_action.values() for e in v]
    return {
        'macro_mse_per_action': {ACTIONS[a]: float(np.mean(v)) if v else None
                                 for a, v in errors_by_action.items()},
        'overall_macro_mse': float(np.mean([np.mean(v) for v in errors_by_action.values()
                                            if v])),
        'overall_mean_mse':  float(np.mean(flat)),
        'overall_std_mse':   float(np.std(flat)),
        'n_tested': n_tested,
    }


def run_cell(cell_name: str, base_dir: Path, max_steps=2000, batch_size=32,
             device='cuda', seed=0):
    cell_dir = base_dir / cell_name
    if not cell_dir.exists():
        raise FileNotFoundError(f'{cell_dir} is missing. Run python -m synthetic.build_dataset first.')
    G, meta, eval_idx, has_pairs = train_one_cell(
        cell_dir, max_steps=max_steps, batch_size=batch_size, device=device, seed=seed,
    )
    clips = np.load(cell_dir / 'clips.npy')
    if len(eval_idx) == 0:
        # One clip per skeleton and action leaves nothing to hold back, so every clip asks a
        # question instead; what is tested is whether the answer carries to skeleton pairs
        # the training never showed.
        eval_idx = list(range(len(meta)))
        print(f"  nothing held back, so all {len(eval_idx)} clips are used as questions")
    metrics = eval_recovery(G, meta, eval_idx, clips, n_eval_samples=200,
                            device=device, seed=seed)
    metrics['cell'] = cell_name
    metrics['has_pairs'] = has_pairs
    metrics['n_train_clips'] = len(meta) - len(eval_idx)
    metrics['n_eval_clips'] = len(eval_idx)
    metrics['max_steps'] = max_steps
    out_path = cell_dir / 'metrics_recovery.json'
    with open(out_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  saved: {out_path}")
    print(f"  result: overall_macro_mse = {metrics['overall_macro_mse']:.4f}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cell', type=str, default=None,
                        choices=['sparse_unpaired', 'sparse_paired',
                                 'dense_unpaired', 'dense_paired'])
    parser.add_argument('--all', action='store_true', help='run all four in turn')
    parser.add_argument('--base_dir', type=str, default='save/synthetic_2x2',
                        help='where build_dataset.py wrote the four versions')
    parser.add_argument('--max_steps', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    base_dir = ROOT / args.base_dir
    if args.all:
        cells = ['sparse_unpaired', 'sparse_paired', 'dense_unpaired', 'dense_paired']
    else:
        if not args.cell:
            parser.error('--cell is required, or --all')
        cells = [args.cell]

    all_metrics = []
    for c in cells:
        m = run_cell(c, base_dir, max_steps=args.max_steps,
                     batch_size=args.batch_size, seed=args.seed)
        all_metrics.append(m)

    if args.all:
        print(f"\n=== Summary ===")
        for m in all_metrics:
            print(f"  {m['cell']:18s}  macro_mse={m['overall_macro_mse']:.4f}  "
                  f"(train={m['n_train_clips']}, questions={m['n_eval_clips']})")
        out_summary = base_dir / 'summary_2x2.json'
        with open(out_summary, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\n  saved summary: {out_summary}")


if __name__ == '__main__':
    main()
