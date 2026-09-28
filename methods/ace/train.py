"""Train ACE, the adversarial retargeting model.

ACE reads a short piece of the source animal's motion and writes the next short piece of
the target animal's, working throughout on the compact codes a frozen per-skeleton
tokenizer produces. Training has two networks. The generator makes the target codes. The
discriminator sees pairs of consecutive target windows, decoded back into motion, and says
whether they look like that animal really moving; the generator is trained partly to
convince it. A second term, the feature loss, compares a handful of measurements of the
source motion against the same measurements of what was generated, which is what ties the
output to the source rather than to the target animal in general.

Each step picks one source animal and one target animal. Half the time the pair is drawn so
that animals with few limbs appear as often as animals with many, and half the time
uniformly, so rare body plans are not drowned out. The discriminator is held fixed while
the generator learns, and the discriminator's own step adds a penalty on how sharply it
responds to real motion, which keeps the two networks in step.

Adversarial training can collapse: the generator finds one output that satisfies the
discriminator and stops varying. Every thousand steps, after a warm-up, a set of probe
pairs chosen once at the start measures how much the generated codes vary, how much of the
target's codebook they cover, and how much the decoded motion varies, each against the same
quantity measured on real motion. If the generated variety falls to a tenth of the real
variety and keeps falling, or if the generator's output grows without bound, the run stops
and writes a short file saying why.

The settings behind each model the paper reports are in methods/ace/configs/.

Usage:
  python -m methods.ace.train --config methods/ace/configs/ace_t.json
  python -m methods.ace.train --config methods/ace/configs/ace_i.json
  python -m methods.ace.train --scope all --run_name my_run --max_steps 50000
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

from core.truebones.param_utils import DATASET_DIR
from methods.ace.generator import ACEGenerator, ACEStartTokens, count_parameters as count_g
from methods.ace.discriminator import ACEDiscriminator, J_MAX, count_parameters as count_d
from methods.tokenizer.skel_graph import (
    SkelGraphEncoder, build_skel_features, pad_to_max_joints,
)
from methods.tokenizer.registry import TokenizerRegistry
from methods.ace.features import precompute_ace_descriptors
from methods.ace.features_torch import (
    ace_feature_per_frame_torch, compute_feature_loss_torch, check_features_or_abort,
)
from methods.common.motion_descriptors import load_contact_groups
from methods.common.training import apply_config, set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / DATASET_DIR
COND_PATH = DATA_ROOT / 'cond.npy'
CACHE_ROOT = PROJECT_ROOT / 'save/latents'
SAVE_ROOT = PROJECT_ROOT / 'save/ace'


def build_skel_cluster_partition(skel_descs, skels):
    """Sort the skeletons into four groups by how many feet or wing tips they have."""
    clusters = {'A': [], 'B': [], 'C': [], 'D': []}
    for s in skels:
        n = skel_descs[s]['n_ee']
        if n <= 2: clusters['A'].append(s)
        elif n <= 4: clusters['B'].append(s)
        elif n <= 6: clusters['C'].append(s)
        else: clusters['D'].append(s)
    return clusters


def stratified_sample_skel_pair(skels, clusters, rng, p_cluster_uniform=0.5):
    """Pick a pair of skeletons: half the time by group, so rare body plans keep coming
    up, and half the time uniformly over all pairs."""
    cluster_keys = sorted(clusters.keys())
    if rng.random() < p_cluster_uniform:
        # Cluster-uniform: pick (source group, target group) uniformly, then a skeleton in each
        src_c = rng.choice(cluster_keys)
        tgt_c = rng.choice(cluster_keys)
        if not clusters[src_c] or not clusters[tgt_c]:
            return rng.choice(skels), rng.choice(skels)
        return rng.choice(clusters[src_c]), rng.choice(clusters[tgt_c])
    else:
        return rng.choice(skels), rng.choice(skels)


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
    skels = [k for k in cache.keys() if k != '_meta']
    print(f"Cached skeletons: {len(skels)}")
    if 'prev_row_idx' not in cache[skels[0]]:
        raise RuntimeError(
            f"Cache has no previous-window index. Rebuild it with "
            f"python -m methods.moreflow.build_cache --scope {args.scope}")

    # Pin the per-skeleton cache to the GPU
    print("Pinning cache to GPU...")
    for skel in skels:
        for key, val in cache[skel].items():
            if isinstance(val, torch.Tensor):
                cache[skel][key] = val.to(device)

    # Frozen tokenizers (fp16 saves about 3 GB with 70 of them)
    tokenizer_dtype = torch.float16 if args.tokenizer_fp16 else torch.float32
    print(f"Loading tokenizer registry (model_dtype={tokenizer_dtype})...")
    registry = TokenizerRegistry(skels, device=device, model_dtype=tokenizer_dtype)
    skel_to_id = {s: i for i, s in enumerate(skels)}
    n_skels = len(skels)
    n_train_skels = n_skels  # every skeleton in scope is trained

    # Per-skeleton descriptors for the ACE feature
    print("Building ACE feature descriptors...")
    cond_dict = np.load(COND_PATH, allow_pickle=True).item()
    contact_groups = load_contact_groups()
    ace_descs = precompute_ace_descriptors(cond_dict, contact_groups)

    # Per-skeleton body length, end effectors and their count, for fast lookup
    body_length_per_skel = torch.zeros(n_skels, device=device)
    ee_joints_per_skel = -torch.ones(n_skels, 8, dtype=torch.long, device=device)  # G_MAX=8
    n_ee_per_skel = torch.zeros(n_skels, dtype=torch.long, device=device)
    for s in skels:
        d = ace_descs[s]
        body_length_per_skel[skel_to_id[s]] = d['body_length']
        ee_joints_per_skel[skel_to_id[s]] = torch.from_numpy(d['ee_joints']).to(device)
        n_ee_per_skel[skel_to_id[s]] = d['n_ee']

    # Skeleton-graph features
    print("Building skeleton-graph features...")
    skel_features = build_skel_features(cond_dict)
    max_J = max(skel_features[s]['n_joints'] for s in skels)
    print(f"max_J: {max_J}")
    pj_padded = {}
    pj_mask = {}
    pj_agg = {}
    for s in skels:
        pj, m = pad_to_max_joints(skel_features[s]['per_joint'], max_J)
        pj_padded[s] = pj.to(device)
        pj_mask[s] = m.to(device)
        pj_agg[s] = skel_features[s]['agg'].to(device)

    # Per-skeleton joint mask for the discriminator (padded to J_MAX)
    joint_mask_per_skel = torch.zeros(n_skels, J_MAX, device=device)
    n_joints_per_skel = torch.zeros(n_skels, dtype=torch.long, device=device)
    for s in skels:
        n_j = registry.get(s)['n_joints']
        joint_mask_per_skel[skel_to_id[s], :n_j] = 1.0
        n_joints_per_skel[skel_to_id[s]] = n_j

    # Per-skeleton mean of the real target latents, used by the START tokens
    print("Computing per-skeleton latent means for the START tokens...")
    z_means = torch.zeros(n_skels, 8, 256, device=device)
    for s in skels:
        z_means[skel_to_id[s]] = cache[s]['z_continuous'].mean(dim=0)

    # The torch feature must agree with the NumPy reference before training starts
    if not args.skip_feature_check:
        print("\n=== Checking the ACE feature: torch against NumPy ===")
        try:
            check_features_or_abort(cond_dict, contact_groups, n_windows=20, n_skels=5,
                                    devices=('cpu',), dtypes=(torch.float64,))
        except RuntimeError as e:
            print(str(e))
            with open(save_dir / 'STATUS_FEATURE_CHECK_FAILED.txt', 'w') as f:
                f.write(str(e))
            sys.exit(1)
        print("Check PASS.\n")

    # Build the models
    print("Building generator, discriminator, graph encoder and START tokens...")
    G = ACEGenerator(n_skels=n_skels).to(device)
    D = ACEDiscriminator(n_skels=n_skels).to(device)
    skel_enc = SkelGraphEncoder(d_in=6, d_hidden=64, d_agg=4, d_out=128).to(device)
    starts = ACEStartTokens(
        n_skels=n_skels, n_train_skels=n_train_skels,
        codebook_dim=256, n_tokens=8, z_means=z_means,
    ).to(device)
    n_g, n_d = count_g(G) + count_g(skel_enc) + count_g(starts), count_d(D)
    print(f"G+enc+starts params: {n_g:,} ({n_g/1e6:.1f}M)")
    print(f"D params:           {n_d:,} ({n_d/1e6:.1f}M)")

    # Optimizers
    g_params = list(G.parameters()) + list(skel_enc.parameters()) + list(starts.parameters())
    g_optim = torch.optim.Adam(g_params, lr=args.lr_g, betas=(0.5, 0.9))
    d_optim = torch.optim.Adam(D.parameters(), lr=args.lr_d, betas=(0.5, 0.9))

    # Group the skeletons by end-effector count
    clusters = build_skel_cluster_partition(ace_descs, skels)
    print(f"Skeleton groups by end-effector count: " + ", ".join(f"{k}={len(v)}" for k, v in clusters.items()))

    # 50 skeleton pairs and 32 source windows each, chosen once here and reused at every
    # check, so the three measurements below are comparable from one check to the next.
    print("Building the probe bank...")
    probe_pairs = []
    cluster_keys_nonempty = [k for k, v in clusters.items() if v]
    probe_rng = np.random.RandomState(args.seed + 17)
    if cluster_keys_nonempty:
        # 10 within-group pairs, drawn from the largest group
        largest = max(cluster_keys_nonempty, key=lambda k: len(clusters[k]))
        for _ in range(min(10, len(clusters[largest])**2)):
            probe_pairs.append((probe_rng.choice(clusters[largest]), probe_rng.choice(clusters[largest])))
        # 20 across groups
        for _ in range(20):
            ck1, ck2 = probe_rng.choice(cluster_keys_nonempty, 2, replace=True)
            probe_pairs.append((probe_rng.choice(clusters[ck1]), probe_rng.choice(clusters[ck2])))
        # 20 fully random
        for _ in range(20):
            probe_pairs.append((probe_rng.choice(skels), probe_rng.choice(skels)))
    probe_n_samples = 32
    # Choose each pair's source windows now
    probe_data = []
    for src_p, tgt_p in probe_pairs:
        n_src_p = cache[src_p]['z_continuous'].shape[0]
        n_tgt_p = cache[tgt_p]['z_continuous'].shape[0]
        if n_src_p < probe_n_samples or n_tgt_p < probe_n_samples:
            continue
        src_idx_p = torch.from_numpy(probe_rng.choice(n_src_p, probe_n_samples, replace=False)).to(device)
        tgt_idx_p = torch.from_numpy(probe_rng.choice(n_tgt_p, probe_n_samples, replace=False)).to(device)
        # Baselines from real target motion: latent variance and decoded-feature variance
        z_real_tgt_p = cache[tgt_p]['z_continuous'][tgt_idx_p]            # [32, 8, 256]
        var_z_real = float(z_real_tgt_p.var().item())
        with torch.no_grad():
            x_real_norm = registry.decode_tokens(tgt_p, z_real_tgt_p)
            x_real_phys = registry.unnormalize(tgt_p, x_real_norm)
            tgt_p_id = skel_to_id[tgt_p]
            psi_real = ace_feature_per_frame_torch(
                x_real_phys,
                body_length_per_skel[tgt_p_id].expand(probe_n_samples),
                ee_joints_per_skel[tgt_p_id].unsqueeze(0).expand(probe_n_samples, -1),
                n_ee_per_skel[tgt_p_id].expand(probe_n_samples),
            )                                                              # [32, T, 37]
            psi_real_mean = psi_real.mean(dim=1)                           # [32, 37]
            var_psi_real = float(psi_real_mean.var().item())
        probe_data.append({
            'src': src_p, 'tgt': tgt_p,
            'src_idx': src_idx_p,
            'var_z_real_baseline': var_z_real,
            'var_psi_real_baseline': var_psi_real,
        })
    print(f"  {len(probe_data)} probe pairs initialized")
    # Release the memory the probe decodes fragmented
    torch.cuda.empty_cache()

    # The three diagnostics, tracked across probe checks
    probe_history = {'latent_var_ratio': [], 'perplexity': [], 'feature_var_ratio': []}

    # Save args
    with open(save_dir / 'args.json', 'w') as f:
        json.dump({**vars(args), 'n_skels': n_skels, 'skels': skels,
                   'max_J': max_J, 'n_g_params': n_g, 'n_d_params': n_d}, f, indent=2)

    # Resume
    rng = np.random.RandomState(args.seed + 1)
    start_step = 0
    ckpts = sorted(save_dir.glob('ckpt_step*.pt'))
    if ckpts and not args.no_resume:
        latest = ckpts[-1]
        print(f"Resuming from {latest.name}")
        sd = torch.load(latest, map_location=device, weights_only=False)
        G.load_state_dict(sd['G'])
        D.load_state_dict(sd['D'])
        skel_enc.load_state_dict(sd['skel_enc'])
        starts.load_state_dict(sd['starts'])
        if args.reset_g_optim:
            print("  --reset_g_optim: the generator's Adam state starts fresh")
        else:
            g_optim.load_state_dict(sd['g_optim'])
        d_optim.load_state_dict(sd['d_optim'])
        start_step = sd['step']
        rng.set_state(sd['train_rng_state'])

    # Training loop
    G.train()
    D.train()
    skel_enc.train()
    starts.train()
    bce = nn.BCEWithLogitsLoss()
    t0 = time.time()
    gamma_r1 = args.gamma_r1

    # Running windows for the logged averages
    d_real_window = deque(maxlen=200)
    d_fake_window = deque(maxlen=200)
    l_adv_window = deque(maxlen=200)
    l_feat_window = deque(maxlen=200)

    # The discriminator's joint-axis attention costs O(J^2); for targets with many joints
    # (Dragon has 142) the plain attention kernel and the gradient penalty together can
    # run out of memory, so the batch shrinks with the target's joint count.
    def adaptive_batch_size(J_tgt, base_B=args.batch_size):
        if J_tgt > 120: return max(2, base_B // 8)
        if J_tgt > 90:  return max(4, base_B // 4)
        if J_tgt > 60:  return max(8, base_B // 2)
        return base_B

    for step in range(start_step, args.max_steps):
        # 1. Sample the skeleton pair
        src_skel, tgt_skel = stratified_sample_skel_pair(skels, clusters, rng,
                                                          p_cluster_uniform=args.p_cluster_uniform)
        src_id_int = skel_to_id[src_skel]
        tgt_id_int = skel_to_id[tgt_skel]

        eff_B = adaptive_batch_size(int(n_joints_per_skel[tgt_id_int].item()))

        # 2. Sample a microbatch from each cache
        N_src = cache[src_skel]['z_continuous'].shape[0]
        N_tgt = cache[tgt_skel]['z_continuous'].shape[0]
        if N_tgt < 4:
            continue
        # Sample target windows from the full pool, clip starts included, so the START
        # tokens receive gradient
        src_idx = torch.from_numpy(rng.choice(N_src, eff_B)).to(device)
        tgt_idx = torch.from_numpy(rng.choice(N_tgt, eff_B)).to(device)

        prev_row_idx_tgt = cache[tgt_skel]['prev_row_idx'][tgt_idx]       # [B] (-1 = clip start)
        is_start = (prev_row_idx_tgt < 0)
        # Pull the source latent, the current target latent and the previous target latent
        z_src = cache[src_skel]['z_continuous'][src_idx]                  # [B, 8, 256]
        z_cur_tgt = cache[tgt_skel]['z_continuous'][tgt_idx]              # [B, 8, 256]
        prev_idx_safe = prev_row_idx_tgt.clamp(min=0)
        z_prev_tgt_real = cache[tgt_skel]['z_continuous'][prev_idx_safe]  # [B, 8, 256]
        # START vectors; `starts` is in g_params, so it gets gradient
        tgt_id_for_start = torch.full((eff_B,), tgt_id_int, dtype=torch.long, device=device)
        start_vec = starts(tgt_id_for_start)                              # [B, 8, 256]
        # Use the cached previous latent where there is one, START at clip starts
        z_prev_tgt = torch.where(is_start.unsqueeze(-1).unsqueeze(-1),
                                 start_vec, z_prev_tgt_real)
        # The discriminator step needs a real previous window to decode, so it keeps only
        # the rows that are not clip starts.
        d_valid_mask = ~is_start
        n_d_valid = int(d_valid_mask.sum().item())

        # Skeleton and graph conditioning, shared within the iteration
        src_pj = pj_padded[src_skel].unsqueeze(0)
        src_m = pj_mask[src_skel].unsqueeze(0)
        src_a = pj_agg[src_skel].unsqueeze(0)
        tgt_pj = pj_padded[tgt_skel].unsqueeze(0)
        tgt_m = pj_mask[tgt_skel].unsqueeze(0)
        tgt_a = pj_agg[tgt_skel].unsqueeze(0)

        src_id_t = torch.full((eff_B,), src_id_int, dtype=torch.long, device=device)
        tgt_id_t = torch.full((eff_B,), tgt_id_int, dtype=torch.long, device=device)
        tgt_jmask_b = joint_mask_per_skel[tgt_id_int].unsqueeze(0).expand(eff_B, -1)

        # ==== discriminator step ====
        # Clip-start windows are excluded: the discriminator should not learn that a decoded
        # learnable START vector is real motion. If the microbatch has none left, skip.
        skip_d_step = (n_d_valid == 0)
        if not skip_d_step:
            with torch.no_grad():
                # Decode only the valid rows (a real previous latent exists for them)
                z_prev_real_filt = z_prev_tgt_real[d_valid_mask]                # [n_valid, 8, 256]
                z_cur_real_filt = z_cur_tgt[d_valid_mask]
                x_prev_decoded_norm = registry.decode_tokens(tgt_skel, z_prev_real_filt)
                x_prev_real = registry.unnormalize(tgt_skel, x_prev_decoded_norm)
                x_cur_real_norm = registry.decode_tokens(tgt_skel, z_cur_real_filt)
                x_cur_real = registry.unnormalize(tgt_skel, x_cur_real_norm)

            # The real input needs a gradient for the penalty below
            x_prev_real_grad = x_prev_real.detach().requires_grad_(True)
            x_cur_real_grad = x_cur_real.detach().requires_grad_(True)

            # Restrict the conditioning to the valid rows
            tgt_id_t_d = tgt_id_t[d_valid_mask]
            tgt_jmask_b_d = tgt_jmask_b[d_valid_mask]
            src_id_t_d = src_id_t[d_valid_mask]

            # The graph encoder is re-run per microbatch so each has its own autograd graph
            src_g_emb_d = skel_enc(src_pj, src_m, src_a).expand(n_d_valid, -1)
            tgt_g_emb_d = skel_enc(tgt_pj, tgt_m, tgt_a).expand(n_d_valid, -1)

            # The gradient penalty differentiates through a gradient, which the fast
            # attention kernels do not support, so this whole step uses the plain one.
            with torch.backends.cuda.sdp_kernel(enable_flash=False,
                                                 enable_mem_efficient=False,
                                                 enable_math=True):
                # The conditioning input is detached: this step must not update the graph encoder
                d_real_logit = D(x_prev_real_grad, x_cur_real_grad,
                                 tgt_id_t_d, tgt_g_emb_d.detach(), tgt_jmask_b_d)
                L_D_real = bce(d_real_logit, torch.ones_like(d_real_logit))

                # Gradient penalty on the real input, both windows
                gp_grads = torch.autograd.grad(
                    d_real_logit.sum(), [x_prev_real_grad, x_cur_real_grad],
                    create_graph=True, retain_graph=True)
                gp_norm_sq = (gp_grads[0]**2).flatten(1).sum(dim=-1) + (gp_grads[1]**2).flatten(1).sum(dim=-1)
                gp = 0.5 * gamma_r1 * gp_norm_sq.mean()

                # Generated: the generator runs on the valid rows. The latent stays
                # continuous throughout, at training and at inference; the tokenizer
                # decoder handles a latent that is not a codebook entry.
                z_src_d = z_src[d_valid_mask]
                z_prev_d = z_prev_tgt[d_valid_mask]   # all real, clip starts were excluded
                with torch.no_grad():
                    z_pred_fake = G(z_src_d, z_prev_d, src_id_t_d, tgt_id_t_d,
                                    src_g_emb_d, tgt_g_emb_d)
                    x_cur_fake_norm = registry.decode_tokens(tgt_skel, z_pred_fake)
                    x_cur_fake = registry.unnormalize(tgt_skel, x_cur_fake_norm)

                d_fake_logit = D(x_prev_real, x_cur_fake, tgt_id_t_d,
                                 tgt_g_emb_d.detach(), tgt_jmask_b_d)
                L_D_fake = bce(d_fake_logit, torch.zeros_like(d_fake_logit))
                L_D = L_D_real + L_D_fake + gp
                d_optim.zero_grad()
                L_D.backward()
                torch.nn.utils.clip_grad_norm_(D.parameters(), 1.0)
                d_optim.step()

            d_real_window.append(torch.sigmoid(d_real_logit).mean().item())
            d_fake_window.append(torch.sigmoid(d_fake_logit).mean().item())
        else:
            # Nothing valid this iteration; the running windows keep the last averages
            L_D_real = torch.tensor(0.0, device=device)
            L_D_fake = torch.tensor(0.0, device=device)
            gp = torch.tensor(0.0, device=device)

        # ==== generator step ====
        # Freeze the discriminator's parameters so they take no gradient from this backward.
        # D.eval() is deliberately not called: LayerNorm has no running statistics, and eval
        # mode combined with frozen parameters can break the autograd graph.
        for p in D.parameters(): p.requires_grad_(False)
        try:
            src_g_emb_g = skel_enc(src_pj, src_m, src_a).expand(eff_B, -1)
            tgt_g_emb_g = skel_enc(tgt_pj, tgt_m, tgt_a).expand(eff_B, -1)

            # The full batch is decoded here, including the decoded START for clip starts,
            # so the START tokens receive gradient.
            with torch.no_grad():
                x_prev_g_norm = registry.decode_tokens(tgt_skel, z_prev_tgt)
                x_prev_g = registry.unnormalize(tgt_skel, x_prev_g_norm)

            # The latent stays continuous: it is decoded directly, with no quantization.
            # --no_source_code zeros the source latent the generator sees, which is how the
            # paper removes the source during training.
            z_src_for_G = torch.zeros_like(z_src) if args.no_source_code else z_src
            z_pred = G(z_src_for_G, z_prev_tgt, src_id_t, tgt_id_t, src_g_emb_g, tgt_g_emb_g)
            x_cur_fake_norm_g = registry.decode_tokens(tgt_skel, z_pred)
            x_cur_fake_g = registry.unnormalize(tgt_skel, x_cur_fake_norm_g)

            # Adversarial loss; the conditioning handed to D is detached
            d_fake_g_logit = D(x_prev_g, x_cur_fake_g, tgt_id_t, tgt_g_emb_g.detach(), tgt_jmask_b)
            L_adv = bce(d_fake_g_logit, torch.ones_like(d_fake_g_logit))

            # Feature loss. Under --no_source_code the source latent is zeroed here too, so
            # no source signal reaches the generator by either route.
            z_src_for_feat = torch.zeros_like(z_src) if args.no_source_code else z_src
            x_src_norm = registry.decode_tokens(src_skel, z_src_for_feat)
            x_src = registry.unnormalize(src_skel, x_src_norm)

            body_src = body_length_per_skel[src_id_int].expand(eff_B)
            body_tgt = body_length_per_skel[tgt_id_int].expand(eff_B)
            ee_src = ee_joints_per_skel[src_id_int].unsqueeze(0).expand(eff_B, -1)
            ee_tgt = ee_joints_per_skel[tgt_id_int].unsqueeze(0).expand(eff_B, -1)
            nee_src = n_ee_per_skel[src_id_int].expand(eff_B)
            nee_tgt = n_ee_per_skel[tgt_id_int].expand(eff_B)

            L_feat = compute_feature_loss_torch(x_src, x_cur_fake_g, body_src, body_tgt,
                                                ee_src, ee_tgt, nee_src, nee_tgt)

            L_G = args.w_adv * L_adv + args.w_feat * L_feat

            g_optim.zero_grad()
            L_G.backward()
            torch.nn.utils.clip_grad_norm_(g_params, 1.0)
            g_optim.step()
        finally:
            # Always restore the discriminator's trainability, exceptions included
            for p in D.parameters(): p.requires_grad_(True)
            d_optim.zero_grad()

        l_adv_window.append(L_adv.item())
        l_feat_window.append(L_feat.item())

        # Logging
        if (step + 1) % args.log_interval == 0:
            elapsed = time.time() - t0
            eta = elapsed / max(1, step + 1 - start_step) * (args.max_steps - step - 1)
            d_real_avg = np.mean(d_real_window)
            d_fake_avg = np.mean(d_fake_window)
            l_adv_avg = np.mean(l_adv_window)
            l_feat_avg = np.mean(l_feat_window)
            print(f"  [{step+1:6d}/{args.max_steps}] L_D_real={L_D_real.item():.3f} "
                  f"L_D_fake={L_D_fake.item():.3f} gp={gp.item():.2e} "
                  f"L_adv={L_adv.item():.3f} L_feat={L_feat.item():.3f} "
                  f"D_real={d_real_avg:.3f} D_fake={d_fake_avg:.3f} "
                  f"pair={src_skel}->{tgt_skel} eff_B={eff_B} "
                  f"({elapsed:.0f}s, ETA {eta/60:.0f}min)")
            with open(log_path, 'a') as f:
                f.write(json.dumps({
                    'step': step + 1,
                    'L_D_real': L_D_real.item(), 'L_D_fake': L_D_fake.item(), 'gp': gp.item(),
                    'L_adv': L_adv.item(), 'L_feat': L_feat.item(),
                    'D_real_avg200': d_real_avg, 'D_fake_avg200': d_fake_avg,
                    'L_adv_avg200': l_adv_avg, 'L_feat_avg200': l_feat_avg,
                    'src': src_skel, 'tgt': tgt_skel,
                }) + '\n')

        # Periodic checkpoint
        if (step + 1) % args.ckpt_interval == 0 and (step + 1) < args.max_steps:
            ck = save_dir / f'ckpt_step{step+1:07d}.pt'
            tmp = ck.with_suffix('.pt.tmp')
            torch.save({
                'step': step + 1,
                'G': G.state_dict(),
                'D': D.state_dict(),
                'skel_enc': skel_enc.state_dict(),
                'starts': starts.state_dict(),
                'g_optim': g_optim.state_dict(),
                'd_optim': d_optim.state_dict(),
                'train_rng_state': rng.get_state(),
                'skels': skels,
                'skel_to_id': skel_to_id,
            }, tmp)
            os.replace(tmp, ck)
            for old in sorted(save_dir.glob('ckpt_step*.pt'))[:-2]:
                old.unlink()

        # ===== diagnostic probes, every 1000 steps after the warm-up =====
        if (step + 1) % 1000 == 0 and (step + 1) >= args.probe_start_step:
            # 1. Discriminator saturation
            d_real_avg = np.mean(d_real_window)
            d_fake_avg = np.mean(d_fake_window)
            d_sat = (d_real_avg > 0.95 and d_fake_avg < 0.05)
            if d_sat:
                with open(save_dir / 'STATUS_D_SATURATED.txt', 'w') as f:
                    f.write(f"step={step+1} D_real={d_real_avg:.3f} D_fake={d_fake_avg:.3f}")

            # 2. The three diversity diagnostics over the probe bank
            G.eval()
            starts.eval()
            with torch.no_grad():
                ratios_lat, perps, ratios_psi = [], [], []
                for probe in probe_data:
                    src_p, tgt_p = probe['src'], probe['tgt']
                    src_p_id = skel_to_id[src_p]
                    tgt_p_id = skel_to_id[tgt_p]
                    z_src_p = cache[src_p]['z_continuous'][probe['src_idx']]   # [32, 8, 256]
                    # START as the previous chunk, which is what inference does first
                    tgt_id_t_p = torch.full((probe_n_samples,), tgt_p_id, dtype=torch.long, device=device)
                    z_prev_p = starts(tgt_id_t_p)                              # [32, 8, 256]
                    src_emb_p = skel_enc(pj_padded[src_p].unsqueeze(0), pj_mask[src_p].unsqueeze(0),
                                          pj_agg[src_p].unsqueeze(0)).expand(probe_n_samples, -1)
                    tgt_emb_p = skel_enc(pj_padded[tgt_p].unsqueeze(0), pj_mask[tgt_p].unsqueeze(0),
                                          pj_agg[tgt_p].unsqueeze(0)).expand(probe_n_samples, -1)
                    src_id_t_p = torch.full((probe_n_samples,), src_p_id, dtype=torch.long, device=device)
                    z_pred_p = G(z_src_p, z_prev_p, src_id_t_p, tgt_id_t_p, src_emb_p, tgt_emb_p)

                    # Diagnostic 1: generated-latent variance against the real one
                    var_pred = float(z_pred_p.var().item())
                    ratios_lat.append(var_pred / max(probe['var_z_real_baseline'], 1e-9))

                    # Diagnostic 2: codebook perplexity
                    cb = registry.codebook(tgt_p, padded_to=0).float()    # cast for fp16 tokenizers
                    dists = torch.cdist(z_pred_p.reshape(-1, 256).unsqueeze(0),
                                         cb.unsqueeze(0)).squeeze(0)
                    indices = dists.argmin(dim=-1)                              # [32*8]
                    K_eff = registry.get(tgt_p)['K_eff']
                    counts = torch.bincount(indices, minlength=K_eff).float()
                    p_dist = counts / counts.sum()
                    entropy = -(p_dist * (p_dist + 1e-10).log()).sum()
                    perps.append(float((entropy / math.log(K_eff)).item()))    # normalized entropy

                    # Diagnostic 3: variance of the decoded motion's feature
                    z_pred_p_q = registry.ste_quantize(tgt_p, z_pred_p)
                    x_pred_norm = registry.decode_tokens(tgt_p, z_pred_p_q)
                    x_pred_phys = registry.unnormalize(tgt_p, x_pred_norm)
                    psi_pred = ace_feature_per_frame_torch(
                        x_pred_phys,
                        body_length_per_skel[tgt_p_id].expand(probe_n_samples),
                        ee_joints_per_skel[tgt_p_id].unsqueeze(0).expand(probe_n_samples, -1),
                        n_ee_per_skel[tgt_p_id].expand(probe_n_samples),
                    )
                    psi_pred_mean = psi_pred.mean(dim=1)
                    var_psi_pred = float(psi_pred_mean.var().item())
                    ratios_psi.append(var_psi_pred / max(probe['var_psi_real_baseline'], 1e-9))
            G.train()
            starts.train()

            mean_ratio_lat = float(np.mean(ratios_lat))
            mean_perp = float(np.mean(perps))
            mean_ratio_psi = float(np.mean(ratios_psi))
            probe_history['latent_var_ratio'].append(mean_ratio_lat)
            probe_history['perplexity'].append(mean_perp)
            probe_history['feature_var_ratio'].append(mean_ratio_psi)
            print(f"  PROBE [{step+1}]: latent_var_ratio={mean_ratio_lat:.3f} "
                  f"perplexity={mean_perp:.3f} feature_var_ratio={mean_ratio_psi:.3f} "
                  f"D={d_real_avg:.2f}/{d_fake_avg:.2f}")
            with open(log_path, 'a') as f:
                f.write(json.dumps({
                    'step': step + 1, 'event': 'probe',
                    'latent_var_ratio': mean_ratio_lat,
                    'perplexity': mean_perp,
                    'feature_var_ratio': mean_ratio_psi,
                    'd_saturated': bool(d_sat),
                }) + '\n')

            # Stop when a signal has stayed below threshold and kept falling for 3 checks
            def _trip(history, key, thresh):
                hist = probe_history[key][-3:]
                if len(hist) < 3:
                    return False
                return all(h < thresh for h in hist) and hist[-1] < hist[0]

            # A latent variance ratio above 1e6, or one that is not finite, means the
            # generator's output has become unbounded.
            if not np.isfinite(mean_ratio_lat) or mean_ratio_lat > 1e6:
                msg = f"Latent variance exploded at step {step+1}: ratio={mean_ratio_lat:.2e}"
                print(f"  GENERATOR EXPLOSION: {msg}")
                with open(save_dir / 'STATUS_G_EXPLODED.txt', 'w') as f:
                    f.write(msg)
                sys.exit(2)
            if _trip(probe_history, 'latent_var_ratio', 0.1):
                msg = f"Latent variance ratio collapsed at step {step+1}: {probe_history['latent_var_ratio'][-3:]}"
                print(f"  MODE COLLAPSE: {msg}")
                with open(save_dir / 'STATUS_MODE_COLLAPSE.txt', 'w') as f:
                    f.write(msg)
                sys.exit(2)
            # Perplexity is not used as a stopping signal. It measures codebook usage if the
            # generator's output were quantized, and nothing here quantizes it: the latent is
            # continuous at training and at inference. The decoded-feature variance below is
            # the diversity signal that does stop training.
            if _trip(probe_history, 'feature_var_ratio', 0.1):
                msg = f"Decoded-feature variance collapsed at step {step+1}: {probe_history['feature_var_ratio'][-3:]}"
                print(f"  MODE COLLAPSE: {msg}")
                with open(save_dir / 'STATUS_MODE_COLLAPSE.txt', 'w') as f:
                    f.write(msg)
                sys.exit(2)

    # Final checkpoint
    ckpt_path = save_dir / 'ckpt_final.pt'
    tmp = ckpt_path.with_suffix('.pt.tmp')
    torch.save({
        'step': args.max_steps,
        'G': G.state_dict(),
        'D': D.state_dict(),
        'skel_enc': skel_enc.state_dict(),
        'starts': starts.state_dict(),
        'args': vars(args),
        'skels': skels,
        'skel_to_id': skel_to_id,
    }, tmp)
    os.replace(tmp, ckpt_path)
    for stale in save_dir.glob('ckpt_step*.pt'):
        stale.unlink()
    print(f"\nSaved final ckpt: {ckpt_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--scope', choices=['train', 'all'], default='all',
                        help='train = the 60 training skeletons; all = all 70')
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--max_steps', type=int, default=50000)
    parser.add_argument('--lr_g', type=float, default=1e-4)
    parser.add_argument('--lr_d', type=float, default=1e-4)
    parser.add_argument('--w_adv', type=float, default=1.0)
    parser.add_argument('--w_feat', type=float, default=1.0)
    parser.add_argument('--gamma_r1', type=float, default=0.1,
                        help="weight of the discriminator's gradient penalty")
    parser.add_argument('--p_cluster_uniform', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_interval', type=int, default=200)
    parser.add_argument('--ckpt_interval', type=int, default=5000)
    parser.add_argument('--probe_start_step', type=int, default=10000,
                        help='start measuring the generated variety only from this step on')
    parser.add_argument('--no_resume', action='store_true')
    parser.add_argument('--reset_g_optim', action='store_true',
                        help="when resuming, start the generator's optimizer fresh; "
                             "use after changing w_feat or w_adv")
    parser.add_argument('--skip_feature_check', action='store_true',
                        help='skip the check that the two versions of the motion feature agree')
    parser.add_argument('--tokenizer_fp16', action='store_true', default=True,
                        help='keep the frozen tokenizers at half precision, which saves about 3 GB')
    parser.add_argument('--no_source_code', action='store_true',
                        help='remove the source motion while the model learns: the generator '
                             'sees zeros in its place, and so does the feature loss')
    parser.add_argument('--config', default=None, type=str,
                        help='file of settings; anything also typed on the command line wins')
    args = parser.parse_args()
    if args.config:
        apply_config(args, args.config)

    if args.run_name is None:
        args.run_name = 'ace_t' if args.scope == 'all' else 'ace_i'

    train(args)


if __name__ == '__main__':
    main()
