"""Train DPG-SB-v3, a retargeting model that starts from a real clip.

The other methods here start generation from noise or from the source animal's codes. This
one starts from a clip the target animal has already performed, with noise added, and
learns to move that starting point towards the answer. Starting from something real is
meant to make the output easy to believe; the risk is that it simply reproduces what it
started from, which the evaluation is built to catch.

Each step draws pairs of clips that share an exact action and sit on two different animals,
one to play the source and one to be the answer. The starting point is a third clip: a
different one of the same action on the target animal. Three terms are trained together:

  a flow term, which asks for the direction from the starting point to the answer;
  an adversarial term, from a second network that judges whether the produced codes look
  like that animal's, switched on after a warm-up;
  a cycle term, which turns the produced codes back into a source and asks to recover the
  original source, so information about the source is not thrown away.

Codes are shifted and scaled per animal before training, and those numbers are written next
to the checkpoint as z_stats.pt. Generation does not read that file: it computes the same
statistics again from the latent cache it is given, which also covers the held-out animals.

The checkpoint holds the two networks' weights and nothing else. Every setting needed to
rebuild them is in args.json beside it, and the generation script reads both files. That is
how the model the paper reports was saved and scored, so it is kept that way.

The settings behind the model the paper reports are in methods/dpg_sb/configs/.

Usage:
  python -m methods.dpg_sb.train --config methods/dpg_sb/configs/dpg_sb.json
"""
from __future__ import annotations
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from core.truebones.param_utils import OBJECT_SUBSETS_DICT
from benchmark.action_taxonomy import (
    ACTION_CLUSTERS, parse_action_from_filename, action_to_cluster,
)
from methods.dpg_sb.model import BridgeGenerator, Discriminator, count_params
from methods.common.training import apply_config, lr_lambda, set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = PROJECT_ROOT / 'save/latents/cache_train.pt'
SAVE_ROOT = PROJECT_ROOT / 'save/dpg_sb'

CLUSTERS = sorted(ACTION_CLUSTERS.keys())


def build_index(cache, train_skels):
    """Group every window by the exact action its clip shows."""
    skels = sorted(s for s in cache.keys() if not s.startswith('_') and s in train_skels)
    skel_to_id = {s: i for i, s in enumerate(skels)}
    by_exact = defaultdict(list)
    by_skel_action = defaultdict(list)  # (skeleton, action) -> [row]
    for skel in skels:
        meta = cache[skel]['meta']
        for ri, (fname, _) in enumerate(meta):
            action = parse_action_from_filename(fname)
            cluster = action_to_cluster(action)
            if cluster is None: continue
            by_exact[action].append((skel, ri))
            by_skel_action[(skel, action)].append(ri)
    exact_actions = sorted(by_exact.keys())
    exact_to_idx = {a: i for i, a in enumerate(exact_actions)}
    return skels, skel_to_id, by_exact, by_skel_action, exact_actions, exact_to_idx


def sample_pairs(by_exact, exact_to_idx, batch_size, rng,
                 same_skel_p=0.0):
    """Sample (source skeleton, source row, target skeleton, target row, action index).

    Both rows share an exact action and sit on different skeletons. Actions are drawn
    uniformly among those present on at least two skeletons.
    """
    valid_actions = [a for a in by_exact.keys() if len(set(s for s, _ in by_exact[a])) >= 2]
    if not valid_actions: return []
    pairs = []
    for _ in range(batch_size):
        action = valid_actions[rng.randint(len(valid_actions))]
        candidates = by_exact[action]
        for _try in range(20):
            i, j = rng.choice(len(candidates), 2, replace=False)
            sa, sr = candidates[i]
            sb, br = candidates[j]
            if sa != sb:
                pairs.append((sa, sr, sb, br, exact_to_idx[action]))
                break
    return pairs


def get_retrieval_init(z_per_skel, by_skel_action, src_skel, src_ri, tgt_skel,
                       action: str, exclude_ri: int = None, rng=None):
    """Pick a clip of the same action on the target skeleton to start the bridge from.
    Falls back to any clip on the target skeleton when that action is not available there.
    """
    candidates = by_skel_action.get((tgt_skel, action), [])
    candidates = [c for c in candidates if c != exclude_ri]
    if not candidates:
        all_tgt_actions = [k for k in by_skel_action if k[0] == tgt_skel]
        if not all_tgt_actions:
            return None
        ka = all_tgt_actions[rng.randint(len(all_tgt_actions)) if rng else 0]
        candidates = by_skel_action[ka]
    if not candidates: return None
    if rng is not None:
        ri = candidates[rng.randint(len(candidates))]
    else:
        ri = candidates[0]
    return z_per_skel[tgt_skel][ri]  # [8, 256]


def train(args):
    save_dir = SAVE_ROOT / args.run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}, run: {args.run_name}")

    cache = torch.load(args.latent_cache, map_location='cpu', weights_only=False)
    train_skels = set(OBJECT_SUBSETS_DICT['train'])
    skels, skel_to_id, by_exact, by_skel_action, exact_actions, exact_to_idx = \
        build_index(cache, train_skels)
    print(f"Skeletons: {len(skels)}, exact actions: {len(exact_actions)}")
    print(f"Top actions: {[(a, len(by_exact[a])) for a in exact_actions[:8]]}")

    # Pin the latents to the GPU, normalized per skeleton for stable training
    z_stats = {}  # {skeleton: (mean, std)}
    z_per_skel = {}
    for s in skels:
        z_raw = cache[s]['z_continuous'].float()
        mu = z_raw.mean(dim=0, keepdim=True)
        sigma = z_raw.std(dim=0, keepdim=True).clamp_min(1e-3)
        z_norm = (z_raw - mu) / sigma
        z_per_skel[s] = z_norm.to(device)
        z_stats[s] = (mu.to(device), sigma.to(device))
    # The statistics are needed again at generation time
    torch.save({s: (mu.cpu(), sigma.cpu()) for s, (mu, sigma) in z_stats.items()},
                save_dir / 'z_stats.pt')
    print(f"Latents normalized per skeleton. First skeleton's std after normalizing: "
          f"{z_per_skel[skels[0]].std():.3f} (should be about 1)")

    # Row to action, used by the cycle term
    clip_to_action = {}  # (skeleton, row) -> action
    for skel in skels:
        meta = cache[skel]['meta']
        for ri, (fname, _) in enumerate(meta):
            action = parse_action_from_filename(fname)
            if action_to_cluster(action) is not None:
                clip_to_action[(skel, ri)] = action

    G = BridgeGenerator(
        codebook_dim=256, n_tokens=8,
        d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads,
        n_skels=len(skels), n_exact_actions=len(exact_actions),
        src_layers=args.src_layers, dropout=args.dropout,
    ).to(device)
    D = Discriminator(
        codebook_dim=256, n_tokens=8,
        d_model=args.d_model_disc, n_layers=args.n_layers_disc, n_heads=4,
        n_skels=len(skels), dropout=args.dropout,
    ).to(device)
    print(f"Generator: {count_params(G)/1e6:.1f}M, Discriminator: {count_params(D)/1e6:.1f}M")

    optG = torch.optim.AdamW(G.parameters(), lr=args.lr, weight_decay=1e-5,
                              betas=(0.9, 0.99))
    optD = torch.optim.AdamW(D.parameters(), lr=args.lr * 0.5, weight_decay=1e-5,
                              betas=(0.5, 0.99))
    schedG = torch.optim.lr_scheduler.LambdaLR(
        optG, lambda s: lr_lambda(s, args.warmup, args.max_steps))

    with open(save_dir / 'args.json', 'w') as f:
        json.dump({**vars(args),
                   'n_skels': len(skels), 'skels': skels,
                   'n_exact_actions': len(exact_actions),
                   'exact_actions': exact_actions,
                   'clusters': CLUSTERS}, f, indent=2)

    rng = np.random.RandomState(args.seed + 1)
    losses_window = {'flow': [], 'adv_g': [], 'adv_d': [], 'cycle': []}
    t0 = time.time()
    G.train(); D.train()

    for step in range(args.max_steps):
        pairs = sample_pairs(by_exact, exact_to_idx, args.batch_size, rng)
        if not pairs: continue

        sa_idx = torch.tensor([skel_to_id[p[0]] for p in pairs], device=device)
        sb_idx = torch.tensor([skel_to_id[p[2]] for p in pairs], device=device)
        aid = torch.tensor([p[4] for p in pairs], device=device)
        actions_str = [exact_actions[p[4]] for p in pairs]

        z_a_list = [z_per_skel[p[0]][p[1]] for p in pairs]
        z_b_list = [z_per_skel[p[2]][p[3]] for p in pairs]
        z_a = torch.stack(z_a_list).to(device)  # [B, 8, 256]
        z_b = torch.stack(z_b_list).to(device)  # [B, 8, 256]

        # The bridge starts from a retrieved clip: another clip of the same action on the
        # target skeleton, never the one being predicted. Noise stands in when the target
        # skeleton has no other clip of that action.
        z_init_list = []
        for p, action in zip(pairs, actions_str):
            init_z = get_retrieval_init(
                z_per_skel, by_skel_action,
                p[0], p[1], p[2], action, exclude_ri=p[3], rng=rng)
            if init_z is None:
                init_z = torch.randn(8, 256, device=device)
            z_init_list.append(init_z)
        z_init = torch.stack(z_init_list).to(device)  # [B, 8, 256]

        # === flow term: from the noisy start point to the real target ===
        t_diff = torch.rand(z_a.shape[0], device=device)
        noise = torch.randn_like(z_b) * args.noise_scale
        z_start = z_init + noise
        # Straight line from the start point to the target
        t_b = t_diff.view(-1, 1, 1)
        z_t = (1 - t_b) * z_start + t_b * z_b
        v_target = z_b - z_start

        src_tokens = G.encode_source(z_a)
        v_pred = G(z_t, t_diff, src_tokens, aid, sa_idx, sb_idx)
        l_flow = F.mse_loss(v_pred, v_target)

        # === adversarial term, after the warm-up ===
        if step >= args.adv_warmup:
            t_zero = torch.zeros(z_a.shape[0], device=device)
            v_one = G(z_start, t_zero, src_tokens, aid, sa_idx, sb_idx)
            z_b_pred = z_start + v_one
            d_fake = D(z_b_pred, sb_idx)
            l_adv_g = -d_fake.mean()
        else:
            z_b_pred = z_start.detach() + (z_b - z_start).detach()  # placeholder, no gradient
            l_adv_g = torch.zeros(1, device=device).squeeze()

        # === cycle term: mapping the predicted target back should recover the source ===
        if step >= args.adv_warmup:
            # already computed for the adversarial term
            z_b_pred_for_cycle = z_b_pred
        else:
            t_zero = torch.zeros(z_a.shape[0], device=device)
            v_one = G(z_start, t_zero, src_tokens, aid, sa_idx, sb_idx)
            z_b_pred_for_cycle = z_start + v_one
        # The prediction becomes the source, the real source becomes the target, and the
        # two skeleton ids swap.
        rev_src_tokens = G.encode_source(z_b_pred_for_cycle)
        z_rev_start = torch.randn_like(z_a) * args.noise_scale
        t_rev_zero = torch.zeros(z_a.shape[0], device=device)
        v_rev = G(z_rev_start, t_rev_zero, rev_src_tokens, aid, sb_idx, sa_idx)
        z_a_recon = z_rev_start + v_rev
        l_cycle = F.mse_loss(z_a_recon, z_a)

        loss_g = l_flow + args.w_adv * l_adv_g + args.w_cycle * l_cycle

        optG.zero_grad()
        loss_g.backward()
        torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
        optG.step()
        schedG.step()

        # === discriminator step, after the warm-up ===
        if step >= args.adv_warmup:
            d_real = D(z_b, sb_idx)
            d_fake = D(z_b_pred.detach(), sb_idx)
            l_d_real = F.relu(1.0 - d_real).mean()
            l_d_fake = F.relu(1.0 + d_fake).mean()
            loss_d = l_d_real + l_d_fake

            optD.zero_grad()
            loss_d.backward()
            torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=1.0)
            optD.step()
        else:
            loss_d = torch.zeros(1, device=device).squeeze()

        losses_window['flow'].append(l_flow.item())
        losses_window['adv_g'].append(l_adv_g.item())
        losses_window['adv_d'].append(loss_d.item())
        losses_window['cycle'].append(l_cycle.item())
        if len(losses_window['flow']) > 200:
            for k in losses_window:
                losses_window[k] = losses_window[k][-200:]

        if step % 50 == 0:
            elapsed = time.time() - t0
            avg = {k: sum(v)/len(v) for k, v in losses_window.items() if v}
            print(f"step {step:5d}/{args.max_steps} "
                  f"l_flow={l_flow.item():.4f} l_adv_g={l_adv_g.item():.3f} "
                  f"l_d={loss_d.item():.3f} avg_flow={avg['flow']:.4f} "
                  f"elapsed={elapsed:.0f}s")

        if step > 0 and step % args.eval_every == 0:
            ckpt_path = save_dir / f'ckpt_step{step:06d}.pt'
            torch.save({
                'step': step, 'G': G.state_dict(), 'D': D.state_dict(),
            }, ckpt_path)
            print(f"  Saved {ckpt_path.name}")

    torch.save({'step': args.max_steps, 'G': G.state_dict(), 'D': D.state_dict()},
                save_dir / 'final.pt')
    with open(save_dir / 'training_history.json', 'w') as f:
        json.dump({'final_step': args.max_steps,
                   'last_losses': {k: v[-1] if v else None for k, v in losses_window.items()}},
                   f, indent=2)
    print(f"\nDone. Final at {save_dir / 'final.pt'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run_name', type=str, default='dpg_sb')
    parser.add_argument('--latent_cache', type=str, default=str(DEFAULT_CACHE))
    parser.add_argument('--max_steps', type=int, default=15000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--warmup', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--d_model', type=int, default=384)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--n_heads', type=int, default=6)
    parser.add_argument('--src_layers', type=int, default=2)
    parser.add_argument('--d_model_disc', type=int, default=256)
    parser.add_argument('--n_layers_disc', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--noise_scale', type=float, default=0.5)
    parser.add_argument('--w_adv', type=float, default=0.1)
    parser.add_argument('--w_cycle', type=float, default=0.1,
                        help='weight of the term that asks the source to be recoverable')
    parser.add_argument('--adv_warmup', type=int, default=2000,
                        help='train on the flow term alone for this many steps, before the '
                             'second network is switched on')
    parser.add_argument('--eval_every', type=int, default=2500)
    parser.add_argument('--config', default=None, type=str,
                        help='file of settings; anything also typed on the command line wins')
    args = parser.parse_args()
    if args.config:
        apply_config(args, args.config)
    train(args)


if __name__ == '__main__':
    main()
