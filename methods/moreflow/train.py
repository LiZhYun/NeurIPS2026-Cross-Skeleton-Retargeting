"""Train MoReFlow, the flow-based retargeting model.

MoReFlow learns to carry one animal's motion codes over to another animal's. Rather than
predicting the answer in one go, it learns a direction of travel: at any point on the
straight line between the source's codes and the target's, it predicts which way to move.
Generation then follows those directions from one end to the other.

A model like this needs to know which source clip goes with which target clip, and the
dataset does not say. Each step therefore takes many windows from each animal, measures the
same small quantity on all of them, such as how fast the body travels or where the feet
are, and pairs them up so that the total mismatch in that measurement is as small as
possible. Those pairs are what the model is trained on, and the measurement used to make
them is also handed to the model, so at generation time the same kind of request can be
made.

After the first ten thousand steps a second term is faded in. It takes one step to the end
of the flow, snaps the result to the target's codebook, decodes it back into motion, and
measures it: the decoded motion's measurement is pulled towards the source's. This is what
makes the output follow the source clip and not just the target animal's habits.

One in ten training examples has its measurement hidden, so the model also learns to move
without being told what to produce. At generation time the two answers are combined, which
is how the request can be given more or less weight.

If the flow's own error is not falling by step 5000, the run stops rather than spending the
rest of the schedule.

The settings behind each model the paper reports are in methods/moreflow/configs/.

Usage:
  python -m methods.moreflow.train --config methods/moreflow/configs/moreflow_t.json
  python -m methods.moreflow.train --config methods/moreflow/configs/moreflow_i.json
  python -m methods.moreflow.train --scope all --run_name my_run --max_steps 200000
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from core.truebones.param_utils import DATASET_DIR
from methods.moreflow.flow_transformer import DiscreteFlowTransformer, count_parameters
from methods.moreflow.conditions import COND_TYPE_TO_INT, COUPLING_CONDITIONS
from methods.tokenizer.skel_graph import (
    SkelGraphEncoder, build_skel_features, pad_to_max_joints,
)
from methods.tokenizer.registry import TokenizerRegistry
from methods.common.motion_descriptors import (
    CONDITIONS, D_COND_PADDED, G_MAX,
    precompute_skel_descriptors, load_contact_groups,
)
from methods.common.motion_descriptors_torch import torch_phi, check_descriptors_or_abort
from methods.common.training import apply_config, lr_lambda, set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / DATASET_DIR
COND_PATH = DATA_ROOT / 'cond.npy'
CACHE_ROOT = PROJECT_ROOT / 'save/latents'
SAVE_ROOT = PROJECT_ROOT / 'save/moreflow'


# -------------------------------------------------------------------- helpers


def alpha_at(step, ramp_start, ramp_end, alpha_max):
    if step < ramp_start:
        return 0.0
    if step >= ramp_end:
        return alpha_max
    return alpha_max * (step - ramp_start) / (ramp_end - ramp_start)


# -------------------------------------------------------------------- training loop


def train(args):
    save_dir = SAVE_ROOT / args.run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / 'training_log.jsonl'

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}, run_name: {args.run_name}, scope: {args.scope}")

    # Load the latent cache
    cache_path = CACHE_ROOT / f'cache_{args.scope}.pt'
    if not cache_path.exists():
        raise FileNotFoundError(
            f"{cache_path} not found. Build it with "
            f"python -m methods.moreflow.build_cache --scope {args.scope}")
    print(f"Loading cache: {cache_path}")
    cache = torch.load(cache_path, map_location='cpu', weights_only=False)
    skels = [s for s in cache.get('_meta', {}).get('skel_list', []) if s != '_meta']
    if not skels:
        skels = [k for k in cache.keys() if k != '_meta']
    print(f"Cached skeletons: {len(skels)}")

    # Move the per-skeleton tensors to the GPU (about 1 GB in total)
    print("Pinning per-skeleton cache to GPU...")
    for skel in skels:
        for key, val in cache[skel].items():
            if isinstance(val, torch.Tensor):
                cache[skel][key] = val.to(device)

    # Frozen tokenizers, needed to decode inside the feature loss
    print("Loading the tokenizer registry...")
    registry = TokenizerRegistry(skels, device=device)
    # Skeleton ids follow the same list the registry was built from
    skel_to_id = {s: i for i, s in enumerate(skels)}
    n_skels = len(skels)

    # Skeleton-graph features
    print("Building skeleton-graph features...")
    cond_dict = np.load(COND_PATH, allow_pickle=True).item()
    contact_groups = load_contact_groups()
    skel_descs = precompute_skel_descriptors(cond_dict, contact_groups)
    skel_features = build_skel_features(cond_dict)
    max_J = max(skel_features[s]['n_joints'] for s in skels)
    print(f"max_J across the skeletons in scope: {max_J}")
    pj_padded = {}
    pj_mask = {}
    pj_agg = {}
    for s in skels:
        pj, m = pad_to_max_joints(skel_features[s]['per_joint'], max_J)
        pj_padded[s] = pj.to(device)
        pj_mask[s] = m.to(device)
        pj_agg[s] = skel_features[s]['agg'].to(device)

    # Per-skeleton descriptor tensors for torch_phi
    ee_joints_per_skel = {}
    bone_lengths_per_skel = {}
    n_ee_per_skel = {}
    for s in skels:
        d = skel_descs[s]
        ee_joints_per_skel[s] = torch.from_numpy(d['ee_joints']).long().to(device)
        bone_lengths_per_skel[s] = torch.from_numpy(d['bone_lengths']).float().to(device)
        n_ee_per_skel[s] = int(d['n_ee'])

    # The torch descriptors must agree with the NumPy reference before training starts
    if not args.skip_descriptor_check:
        print("\n=== Checking the condition descriptors: torch against NumPy ===")
        try:
            check_descriptors_or_abort(cond_dict, contact_groups, n_windows=20, n_skels=5,
                                       devices=('cpu',), dtypes=(torch.float64,))
        except RuntimeError as e:
            print(str(e))
            with open(save_dir / 'STATUS_DESCRIPTOR_CHECK_FAILED.txt', 'w') as f:
                f.write(str(e))
            sys.exit(1)
        print("Check PASS.\n")

    # Build the model
    print("Building the flow transformer and the graph encoder...")
    model = DiscreteFlowTransformer(
        codebook_dim=256, d_model=512, n_layers=6, n_heads=8,
        dim_ff=2048, dropout=0.1, max_seq_len=8,
        n_skels=n_skels,
    ).to(device)
    skel_enc = SkelGraphEncoder(d_in=6, d_hidden=64, d_agg=4, d_out=128).to(device)
    n_params = count_parameters(model) + count_parameters(skel_enc)
    print(f"Total trainable params: {n_params:,} ({n_params/1e6:.1f}M)")

    # Optimizer and schedule
    all_params = list(model.parameters()) + list(skel_enc.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=args.lr, betas=(0.9, 0.99),
                                   weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_lambda(step, args.warmup_steps, args.max_steps))

    # Save args
    with open(save_dir / 'args.json', 'w') as f:
        json.dump({**vars(args), 'n_skels': n_skels, 'skels': skels,
                   'max_J': max_J, 'n_params': n_params}, f, indent=2)

    # The training loop draws its pairs from `rng`; both this and the global NumPy state are
    # saved and restored, so a resumed run continues the sequence instead of replaying it.
    rng = np.random.RandomState(args.seed + 1)

    # Resume from the latest periodic checkpoint
    start_step = 0
    lfm_min_so_far_resumed = None
    ckpts = sorted(save_dir.glob('ckpt_step*.pt'))
    if ckpts and not args.no_resume:
        latest = ckpts[-1]
        print(f"Resuming from {latest.name}")
        sd = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(sd['model'])
        skel_enc.load_state_dict(sd['skel_enc'])
        optimizer.load_state_dict(sd['optimizer'])
        scheduler.load_state_dict(sd['scheduler'])
        start_step = sd['step']
        torch.set_rng_state(sd['torch_rng_state'].to('cpu', dtype=torch.uint8))
        if torch.cuda.is_available() and sd.get('cuda_rng_state') is not None:
            cuda_state = sd['cuda_rng_state']
            if isinstance(cuda_state, list):
                cuda_state = [s.to('cpu', dtype=torch.uint8) for s in cuda_state]
            torch.cuda.set_rng_state_all(cuda_state)
        np.random.set_state(sd['numpy_rng_state'])
        if 'train_rng_state' in sd:
            rng.set_state(sd['train_rng_state'])
        else:
            print(f"  WARN: checkpoint has no 'train_rng_state'; pair sampling restarts "
                  f"from seed={args.seed + 1}")
        lfm_min_so_far_resumed = sd.get('lfm_min_so_far', None)

    # Tracks whether the flow-matching loss is still improving
    lfm_window = deque(maxlen=200)
    lfm_min_so_far = lfm_min_so_far_resumed if lfm_min_so_far_resumed is not None else float('inf')

    model.train()
    skel_enc.train()
    # `rng` was created above and may have been restored from the checkpoint; do not
    # re-create it here or the restore is lost.
    t0 = time.time()

    for step in range(start_step, args.max_steps):
        # 1. Sample the source skeleton, the target skeleton and the condition family
        src_skel = skels[rng.randint(0, n_skels)]
        tgt_skel = skels[rng.randint(0, n_skels)]
        c_str = COUPLING_CONDITIONS[rng.randint(0, len(COUPLING_CONDITIONS))]

        N_src = cache[src_skel]['z_continuous'].shape[0]
        N_tgt = cache[tgt_skel]['z_continuous'].shape[0]
        k = min(N_src, N_tgt, args.k_coupling)
        if k < 64:
            continue  # too few windows to couple

        # 2. Sample k windows from each
        src_idx = torch.from_numpy(rng.choice(N_src, k, replace=False)).to(device)
        tgt_idx = torch.from_numpy(rng.choice(N_tgt, k, replace=False)).to(device)

        src_z = cache[src_skel]['z_continuous'][src_idx]      # [k, 8, 256]
        tgt_z = cache[tgt_skel]['z_continuous'][tgt_idx]      # [k, 8, 256]
        src_phi = cache[src_skel][f'phi_{c_str}'][src_idx]    # [k, 24]
        tgt_phi = cache[tgt_skel][f'phi_{c_str}'][tgt_idx]    # [k, 24]

        # 3. Cost matrix: squared descriptor distance
        M = ((src_phi[:, None, :] - tgt_phi[None, :, :]) ** 2).sum(dim=-1)  # [k, k]

        # 4. Linear assignment, on the CPU
        M_cpu = M.detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(M_cpu)

        # Shuffle the coupled pairs so the microbatches are not periodic
        order = rng.permutation(k)
        coupled_src_z = src_z[row_ind[order]]                  # [k, 8, 256]
        coupled_tgt_z = tgt_z[col_ind[order]]                  # [k, 8, 256]

        # 5. Accumulate gradient over microbatches
        optimizer.zero_grad()
        n_mb = math.ceil(k / args.microbatch)
        cur_alpha = alpha_at(step, args.feat_ramp_start, args.feat_ramp_end, args.alpha_max)
        l_fm_total = 0.0
        l_feat_total = 0.0
        n_seen = 0

        cond_type_int = COND_TYPE_TO_INT[c_str]
        src_id = skel_to_id[src_skel]
        tgt_id = skel_to_id[tgt_skel]
        # The graph encoder is re-run per microbatch: it is a small network, and each
        # backward frees its own graph.
        src_pj = pj_padded[src_skel].unsqueeze(0)              # [1, max_J, 6]
        src_m = pj_mask[src_skel].unsqueeze(0)
        src_a = pj_agg[src_skel].unsqueeze(0)
        tgt_pj = pj_padded[tgt_skel].unsqueeze(0)
        tgt_m = pj_mask[tgt_skel].unsqueeze(0)
        tgt_a = pj_agg[tgt_skel].unsqueeze(0)

        for mb_start in range(0, k, args.microbatch):
            mb_end = min(mb_start + args.microbatch, k)
            mb_size = mb_end - mb_start
            z_src_mb = coupled_src_z[mb_start:mb_end]
            z_tgt_mb = coupled_tgt_z[mb_start:mb_end]

            q_mb = torch.rand(mb_size, device=device)
            cond_mask_mb = (torch.rand(mb_size, device=device) < args.p_mask).float()
            z_q_mb = (1.0 - q_mb)[:, None, None] * z_src_mb + q_mb[:, None, None] * z_tgt_mb

            src_id_mb = torch.full((mb_size,), src_id, dtype=torch.long, device=device)
            tgt_id_mb = torch.full((mb_size,), tgt_id, dtype=torch.long, device=device)
            cond_type_mb = torch.full((mb_size,), cond_type_int, dtype=torch.long, device=device)
            cond_vec_mb = src_phi[row_ind[order][mb_start:mb_end]].float()  # the source's descriptor
            src_graph_emb = skel_enc(src_pj, src_m, src_a)                       # [1, 128]
            tgt_graph_emb = skel_enc(tgt_pj, tgt_m, tgt_a)                       # [1, 128]
            src_graph_mb = src_graph_emb.expand(mb_size, -1)
            tgt_graph_mb = tgt_graph_emb.expand(mb_size, -1)

            v_psi = model(z_q_mb, q_mb, src_id_mb, tgt_id_mb,
                          src_graph_mb, tgt_graph_mb,
                          cond_type_mb, cond_vec_mb, cond_mask_mb)  # [mb, 8, 256]

            # Flow-matching loss
            target_v = z_tgt_mb - z_src_mb
            l_fm = ((v_psi - target_v) ** 2).mean()

            l_feat = torch.zeros(1, device=device)
            if cur_alpha > 0:
                # 1. One Euler step from q to the end of the flow
                z_1_hat = z_q_mb + (1.0 - q_mb)[:, None, None] * v_psi
                # 2. Straight-through quantize to the target codebook
                z_1_q = registry.ste_quantize(tgt_skel, z_1_hat)
                # 3. Decode with the frozen target tokenizer
                x_tgt_hat_norm = registry.decode_tokens(tgt_skel, z_1_q)         # [mb, 32, J_tgt, 13]
                # 4. Back to physical units, differentiably
                x_tgt_hat_phys = registry.unnormalize(tgt_skel, x_tgt_hat_norm)  # [mb, 32, J_tgt, 13]
                # 5. Descriptor of the decoded motion, pulled towards the source's
                ee_tgt = ee_joints_per_skel[tgt_skel].unsqueeze(0).expand(mb_size, -1)
                bl_tgt = bone_lengths_per_skel[tgt_skel].unsqueeze(0).expand(mb_size, -1)
                nee_tgt = torch.full((mb_size,), n_ee_per_skel[tgt_skel], dtype=torch.long, device=device)
                phi_hat = torch_phi(x_tgt_hat_phys, c_str, ee_tgt, bl_tgt, nee_tgt)  # [mb, 24]
                phi_src_target = src_phi[row_ind[order][mb_start:mb_end]].float()
                l_feat = ((phi_src_target - phi_hat) ** 2).mean()

            loss = (l_fm + cur_alpha * l_feat) * (mb_size / k)
            loss.backward()

            l_fm_total += l_fm.item() * mb_size
            l_feat_total += l_feat.item() * mb_size
            n_seen += mb_size

        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        optimizer.step()
        scheduler.step()

        l_fm_mean = l_fm_total / n_seen
        l_feat_mean = l_feat_total / n_seen
        lfm_window.append(l_fm_mean)
        if l_fm_mean < lfm_min_so_far:
            lfm_min_so_far = l_fm_mean

        # Logging
        if (step + 1) % args.log_interval == 0:
            elapsed = time.time() - t0
            eta = elapsed / max(1, step + 1 - start_step) * (args.max_steps - step - 1)
            cur_lr = optimizer.param_groups[0]['lr']
            avg_lfm = np.mean(list(lfm_window))
            print(f"  [{step+1:6d}/{args.max_steps}] L_FM={l_fm_mean:.4f} (avg200={avg_lfm:.4f}) "
                  f"L_feat={l_feat_mean:.4f} alpha={cur_alpha:.3f} "
                  f"lr={cur_lr:.2e} k={k} pair={src_skel}->{tgt_skel} c={c_str} "
                  f"({elapsed:.0f}s, ETA {eta/60:.0f}min)")
            with open(log_path, 'a') as f:
                f.write(json.dumps({
                    'step': step + 1, 'L_FM': l_fm_mean, 'L_feat': l_feat_mean,
                    'alpha': cur_alpha, 'lr': cur_lr, 'k': int(k),
                    'src': src_skel, 'tgt': tgt_skel, 'c': c_str,
                }) + '\n')

        # Periodic checkpoint
        if (step + 1) % args.ckpt_interval == 0 and (step + 1) < args.max_steps:
            ck = save_dir / f'ckpt_step{step+1:07d}.pt'
            tmp = ck.with_suffix('.pt.tmp')
            torch.save({
                'step': step + 1,
                'model': model.state_dict(),
                'skel_enc': skel_enc.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'numpy_rng_state': np.random.get_state(),
                'train_rng_state': rng.get_state(),   # the pair-sampling generator
                'torch_rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                'lfm_min_so_far': lfm_min_so_far,
            }, tmp)
            os.replace(tmp, ck)
            for old in sorted(save_dir.glob('ckpt_step*.pt'))[:-2]:
                old.unlink()

        # Stop early if the flow-matching loss is not decreasing
        if step + 1 == args.loss_check_step:
            avg_lfm = np.mean(list(lfm_window))
            improvement = (avg_lfm - lfm_min_so_far) / max(avg_lfm, 1e-6)
            print(f"\n=== Loss check at step {step+1} ===")
            print(f"  current avg L_FM (last 200): {avg_lfm:.4f}")
            print(f"  best L_FM so far:            {lfm_min_so_far:.4f}")
            print(f"  relative improvement:        {improvement:.3f}")
            if improvement < 0.05:
                msg = (f"L_FM is not decreasing at step {step+1}. "
                       f"avg200={avg_lfm:.4f}, best={lfm_min_so_far:.4f}, "
                       f"improvement={improvement:.3f}. Stopping.")
                print(f"  STOPPING: {msg}")
                with open(save_dir / 'STATUS_LOSS_NOT_DECREASING.txt', 'w') as f:
                    f.write(msg)
                sys.exit(2)
            else:
                print(f"  L_FM is decreasing; continuing.")
                with open(save_dir / 'STATUS_LOSS_CHECK_PASSED.txt', 'w') as f:
                    f.write(f"avg200={avg_lfm:.4f}, best={lfm_min_so_far:.4f}, "
                            f"improvement={improvement:.3f}")

    # Final checkpoint
    ckpt_path = save_dir / 'ckpt_final.pt'
    tmp = ckpt_path.with_suffix('.pt.tmp')
    torch.save({
        'step': args.max_steps,
        'model': model.state_dict(),
        'skel_enc': skel_enc.state_dict(),
        'args': vars(args),
        'skels': skels,
        'skel_to_id': skel_to_id,
        'lfm_min_so_far': lfm_min_so_far,
    }, tmp)
    os.replace(tmp, ckpt_path)
    for stale in save_dir.glob('ckpt_step*.pt'):
        stale.unlink()
    print(f"\nSaved final ckpt: {ckpt_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--scope', choices=['train', 'all'], default='all',
                        help='train = the 60 training skeletons, leaving the other 10 unseen; '
                             'all = all 70')
    parser.add_argument('--run_name', type=str, default=None,
                        help='defaults to moreflow_t for all, moreflow_i for train')
    parser.add_argument('--max_steps', type=int, default=200000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--alpha_max', type=float, default=0.2)
    parser.add_argument('--feat_ramp_start', type=int, default=10000)
    parser.add_argument('--feat_ramp_end', type=int, default=30000)
    parser.add_argument('--p_mask', type=float, default=0.1)
    parser.add_argument('--k_coupling', type=int, default=512)
    parser.add_argument('--microbatch', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_interval', type=int, default=200)
    parser.add_argument('--ckpt_interval', type=int, default=5000)
    parser.add_argument('--loss_check_step', type=int, default=5000,
                        help='step at which the run stops unless its error is falling')
    parser.add_argument('--no_resume', action='store_true')
    parser.add_argument('--skip_descriptor_check', action='store_true',
                        help='skip the check that the two versions of the measurements agree')
    parser.add_argument('--config', default=None, type=str,
                        help='file of settings; anything also typed on the command line wins')
    args = parser.parse_args()
    if args.config:
        apply_config(args, args.config)

    if args.run_name is None:
        args.run_name = 'moreflow_t' if args.scope == 'all' else 'moreflow_i'

    train(args)


if __name__ == '__main__':
    main()
