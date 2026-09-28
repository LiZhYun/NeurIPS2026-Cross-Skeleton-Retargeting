"""Train an AL-Flow model, which generates a motion from an action label.

These models answer a narrower question than a retargeter: given only the name of an action
and the animal to perform it, how good a motion can be produced? That matters because it
sets the bar a retargeter has to clear. Each clip in the dataset carries an action name
taken from its filename, and those names are grouped into ten coarse clusters; a model here
is given both the cluster and the exact name.

Training starts each example from noise and learns the direction that carries it to a real
clip's codes. Which extra information the model also sees is what --variant selects:

  labels              the two action labels and the target animal, and nothing else.
  labels_source       also the source clip. Examples are drawn in pairs that share an exact
                      action and sit on two different animals, so one can play the source
                      and the other the answer.
  labels_source_graph the same pairs, through a model that never looks an animal up by
                      name: both animals reach it only as a description of their skeleton,
                      so it can also be asked about animals it never trained on.

Each piece of information is withheld one time in ten, independently, so the model also
learns to work without it. That is what lets generation weigh a label up or down.

Clips whose action name matches none of the ten clusters are left out.

The settings behind each model the paper reports are in methods/alflow/configs/.

Usage:
  python -m methods.alflow.train --config methods/alflow/configs/al_flow.json
  python -m methods.alflow.train --config methods/alflow/configs/al_flow_src.json
  python -m methods.alflow.train --config methods/alflow/configs/al_flow_src_g.json
"""
from __future__ import annotations
import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.truebones.param_utils import OBJECT_SUBSETS_DICT, DATASET_DIR
from benchmark.action_taxonomy import (
    ACTION_CLUSTERS, parse_action_from_filename, action_to_cluster,
)
from methods.alflow.generator import (
    ALFlowGenerator, ALFlowSrcGenerator, ALFlowSrcGraphGenerator,
)
from methods.alflow.layers import count_parameters
from methods.common.training import apply_config, lr_lambda, set_seed
from methods.tokenizer.skel_graph import (
    SkelGraphEncoder, build_skel_features, pad_to_max_joints,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / DATASET_DIR
COND_PATH = DATA_ROOT / 'cond.npy'
DEFAULT_CACHE = PROJECT_ROOT / 'save/latents/cache_train.pt'
SAVE_ROOT = PROJECT_ROOT / 'save/alflow'

# Cluster vocabulary: index 0 is the null label, 1..N the real clusters
CLUSTERS = sorted(ACTION_CLUSTERS.keys())
CLUSTER_TO_IDX = {c: i + 1 for i, c in enumerate(CLUSTERS)}
N_CLUSTERS = len(CLUSTERS) + 1

# Exact-action vocabulary: index 0 is the null label, 1..N the real actions
EXACT_ACTIONS = sorted({a for acts in ACTION_CLUSTERS.values() for a in acts})
EXACT_ACTION_TO_IDX = {a: i + 1 for i, a in enumerate(EXACT_ACTIONS)}
N_EXACT_ACTIONS = len(EXACT_ACTIONS) + 1

VARIANTS = ('labels', 'labels_source', 'labels_source_graph')


def build_label_rows(cache, skel_to_id):
    """Row index for the labels variant: (skeleton, row, cluster index, exact index).

    Rows whose action maps to no cluster, or to no known exact action, are skipped.
    """
    rows = []
    skipped_cluster = 0
    skipped_exact = 0
    for skel_name, data in cache.items():
        if skel_name.startswith('_'):  # skip _meta and other reserved keys
            continue
        meta = data['meta']  # (clip filename, first frame)
        for ri, (fname, _) in enumerate(meta):
            action = parse_action_from_filename(fname)
            cluster = action_to_cluster(action)
            if cluster is None:
                skipped_cluster += 1
                continue
            if action not in EXACT_ACTION_TO_IDX:
                skipped_exact += 1
                continue
            cid = CLUSTER_TO_IDX[cluster]
            eid = EXACT_ACTION_TO_IDX[action]
            rows.append((skel_name, ri, cid, eid))
    print(f"Built {len(rows)} training rows "
          f"(skipped {skipped_cluster} with no cluster, "
          f"{skipped_exact} with no known exact action)")
    return rows


def build_pair_index(cache, train_skels):
    """Index for the source-conditioned variants:
    by_exact[action] -> [(skeleton, row, cluster index, exact index)]."""
    skels = sorted(s for s in cache.keys() if not s.startswith('_') and s in train_skels)
    skel_to_id = {s: i for i, s in enumerate(skels)}
    by_exact = defaultdict(list)
    skipped_cluster = skipped_exact = 0
    for skel in skels:
        meta = cache[skel]['meta']
        for ri, (fname, _) in enumerate(meta):
            action = parse_action_from_filename(fname)
            cluster = action_to_cluster(action)
            if cluster is None:
                skipped_cluster += 1
                continue
            if action not in EXACT_ACTION_TO_IDX:
                skipped_exact += 1
                continue
            cid = CLUSTER_TO_IDX[cluster]
            eid = EXACT_ACTION_TO_IDX[action]
            by_exact[action].append((skel, ri, cid, eid))
    print(f"Pair index: {len(skels)} skeletons, {len(by_exact)} exact actions; "
          f"skipped {skipped_cluster} with no cluster, {skipped_exact} with no known exact action")
    return skels, skel_to_id, by_exact


def sample_pairs(by_exact, batch_size, rng):
    """Sample (source skeleton, source row, target skeleton, target row, cluster, exact).

    Both rows share an exact action and sit on different skeletons. Actions are drawn
    uniformly among those present on at least two skeletons, so the batch is balanced over
    actions rather than over clips.
    """
    valid_actions = [a for a in by_exact.keys()
                     if len(set(s for s, _, _, _ in by_exact[a])) >= 2]
    if not valid_actions:
        return []
    pairs = []
    for _ in range(batch_size):
        action = valid_actions[rng.randint(len(valid_actions))]
        candidates = by_exact[action]
        for _try in range(20):
            i, j = rng.choice(len(candidates), 2, replace=False)
            sa, ra, ca, ea = candidates[i]
            sb, rb, cb, eb = candidates[j]
            if sa != sb:
                # The two share an exact action, so their labels are identical
                pairs.append((sa, ra, sb, rb, ca, ea))
                break
    return pairs


def train(args):
    save_dir = SAVE_ROOT / args.run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / 'training_log.jsonl'
    with_source = args.variant in ('labels_source', 'labels_source_graph')

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}, run_name: {args.run_name}, variant: {args.variant}")
    print(f"Vocabulary: {N_CLUSTERS} clusters (10 plus null), "
          f"{N_EXACT_ACTIONS} exact actions (124 plus null)")

    cache_path = Path(args.latent_cache)
    print(f"Loading cache: {cache_path}")
    cache = torch.load(cache_path, map_location='cpu', weights_only=False)

    if with_source:
        train_skels = set(OBJECT_SUBSETS_DICT['train'])
        skels, skel_to_id, by_exact = build_pair_index(cache, train_skels)
        if not by_exact:
            raise RuntimeError("No paired training rows after filtering")
        rows = None
    else:
        skels = sorted([s for s in cache.keys() if not s.startswith('_')])
        skel_to_id = {s: i for i, s in enumerate(skels)}
        print(f"Cache: {len(skels)} skeletons")
        rows = build_label_rows(cache, skel_to_id)
        if not rows:
            raise RuntimeError("No training rows after filtering — check the action taxonomy.")
        by_exact = None

    # Pin the latents to the GPU as fp32
    z_per_skel = {s: cache[s]['z_continuous'].float().to(device) for s in skels}

    # Skeleton-graph features
    print("Building skeleton-graph features...")
    cond_dict = np.load(COND_PATH, allow_pickle=True).item()
    skel_features = build_skel_features(cond_dict)
    max_J = max(skel_features[s]['n_joints'] for s in skels if s in skel_features)
    pj_padded, pj_mask, pj_agg = {}, {}, {}
    for s in skels:
        if s not in skel_features:
            continue
        p, m = pad_to_max_joints(skel_features[s]['per_joint'], max_J)
        pj_padded[s] = p.to(device)
        pj_mask[s] = m.to(device)
        pj_agg[s] = skel_features[s]['agg'].to(device)
    skel_enc = SkelGraphEncoder().to(device)

    # Generator
    if args.variant == 'labels':
        G = ALFlowGenerator(
            n_skels=len(skels),
            n_clusters=N_CLUSTERS,
            n_exact_actions=N_EXACT_ACTIONS,
            codebook_dim=256,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
        ).to(device)
    elif args.variant == 'labels_source':
        G = ALFlowSrcGenerator(
            n_skels=len(skels),
            n_clusters=N_CLUSTERS,
            n_exact_actions=N_EXACT_ACTIONS,
            codebook_dim=256,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
        ).to(device)
    else:
        G = ALFlowSrcGraphGenerator(
            n_clusters=N_CLUSTERS,
            n_exact_actions=N_EXACT_ACTIONS,
            codebook_dim=256,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
        ).to(device)
    n_g = count_parameters(G)
    n_se = count_parameters(skel_enc)
    print(f"Generator params: {n_g/1e6:.1f}M, graph encoder: {n_se/1e6:.1f}M")

    optim = torch.optim.AdamW(list(G.parameters()) + list(skel_enc.parameters()),
                              lr=args.lr, weight_decay=0.01, betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.LambdaLR(optim,
        lambda s: lr_lambda(s, args.warmup, args.max_steps))

    # Save args and the label vocabulary, for resuming and for generation
    with open(save_dir / 'args.json', 'w') as f:
        json.dump({**vars(args), 'n_skels': len(skels), 'skels': skels,
                   'n_clusters': N_CLUSTERS, 'clusters': CLUSTERS,
                   'n_exact_actions': N_EXACT_ACTIONS, 'exact_actions': EXACT_ACTIONS,
                   'max_J': max_J}, f, indent=2)

    # Resume
    rng = np.random.RandomState(args.seed + 1)
    start_step = 0
    ckpts = sorted(save_dir.glob('ckpt_step*.pt'))
    if ckpts and not args.no_resume:
        latest = ckpts[-1]
        print(f"Resuming from {latest.name}")
        sd = torch.load(latest, map_location=device, weights_only=False)
        G.load_state_dict(sd['G'])
        skel_enc.load_state_dict(sd['skel_enc'])
        optim.load_state_dict(sd['optim'])
        start_step = sd['step']
        rng.set_state(sd['rng'])

    print(f"Training {args.max_steps - start_step} steps from {start_step}")
    t0 = time.time()
    G.train(); skel_enc.train()
    losses_window = []
    for step in range(start_step, args.max_steps):
        optim.zero_grad()

        total_loss = 0.0
        n_micro = 0

        if not with_source:
            # Draw rows, then group them by skeleton so each microbatch is one skeleton
            idx = rng.choice(len(rows), args.batch_size)
            batch_rows = [rows[i] for i in idx]
            skel_groups = defaultdict(list)
            for (s, ri, cid, eid) in batch_rows:
                skel_groups[s].append((ri, cid, eid))

            for skel, items in skel_groups.items():
                ris = torch.tensor([it[0] for it in items], dtype=torch.long, device=device)
                cids = torch.tensor([it[1] for it in items], dtype=torch.long, device=device)
                eids = torch.tensor([it[2] for it in items], dtype=torch.long, device=device)
                B = len(items)

                z1 = z_per_skel[skel][ris]                      # [B, 8, 256] target
                z0 = torch.randn_like(z1)                       # noise
                q = torch.rand(B, device=device)                # flow time
                zt = (1 - q.view(-1, 1, 1)) * z0 + q.view(-1, 1, 1) * z1
                v_target = z1 - z0                              # straight-line velocity

                # The two label channels are dropped independently
                cluster_drop = (torch.rand(B, device=device) < args.p_cluster_drop)
                exact_drop = (torch.rand(B, device=device) < args.p_exact_drop)

                tid = torch.full((B,), skel_to_id[skel], dtype=torch.long, device=device)
                tg = skel_enc(pj_padded[skel].unsqueeze(0), pj_mask[skel].unsqueeze(0),
                              pj_agg[skel].unsqueeze(0)).expand(B, -1)

                v_pred = G(zt, q, tid, tg, cids, eids,
                           cluster_mask=cluster_drop, exact_mask=exact_drop)
                loss = F.mse_loss(v_pred, v_target)
                loss.backward()
                total_loss += loss.item()
                n_micro += 1
        else:
            pairs = sample_pairs(by_exact, args.batch_size, rng)
            if not pairs:
                continue
            # Group by (source, target) skeleton so the graph encoder runs once per group
            groups_by_src = defaultdict(list)
            if args.variant == 'labels_source':
                groups_by_tgt = defaultdict(list)
                for (sa, ra, sb, rb, ca, ea) in pairs:
                    groups_by_tgt[sb].append((sa, ra, rb, ca, ea))
                for sb, items in groups_by_tgt.items():
                    for (sa, ra, rb, ca, ea) in items:
                        groups_by_src[(sa, sb)].append((ra, rb, ca, ea))
            else:
                for (sa, ra, sb, rb, ca, ea) in pairs:
                    groups_by_src[(sa, sb)].append((ra, rb, ca, ea))

            for (sa, sb), items in groups_by_src.items():
                ra_t = torch.tensor([it[0] for it in items], dtype=torch.long, device=device)
                rb_t = torch.tensor([it[1] for it in items], dtype=torch.long, device=device)
                ca_t = torch.tensor([it[2] for it in items], dtype=torch.long, device=device)
                ea_t = torch.tensor([it[3] for it in items], dtype=torch.long, device=device)
                B = len(items)

                z_a = z_per_skel[sa][ra_t]                      # [B, 8, 256] source latent
                z_b = z_per_skel[sb][rb_t]                      # [B, 8, 256] target latent
                z_noise = torch.randn_like(z_b)
                q = torch.rand(B, device=device)
                z_b_t = (1 - q.view(-1, 1, 1)) * z_noise + q.view(-1, 1, 1) * z_b
                v_target = z_b - z_noise

                # The three channels are dropped independently
                cluster_drop = (torch.rand(B, device=device) < args.p_cluster_drop)
                exact_drop = (torch.rand(B, device=device) < args.p_exact_drop)
                src_drop = (torch.rand(B, device=device) < args.p_src_drop)

                tg = skel_enc(pj_padded[sb].unsqueeze(0), pj_mask[sb].unsqueeze(0),
                              pj_agg[sb].unsqueeze(0)).expand(B, -1)
                sg = skel_enc(pj_padded[sa].unsqueeze(0), pj_mask[sa].unsqueeze(0),
                              pj_agg[sa].unsqueeze(0)).expand(B, -1)

                if args.variant == 'labels_source':
                    tid = torch.full((B,), skel_to_id[sb], dtype=torch.long, device=device)
                    sid = torch.full((B,), skel_to_id[sa], dtype=torch.long, device=device)
                    v_pred = G(z_b_t, q, tid, tg, z_a, sid, sg, ca_t, ea_t,
                               cluster_mask=cluster_drop, exact_mask=exact_drop,
                               src_mask=src_drop)
                else:
                    v_pred = G(z_b_t, q, tg, z_a, sg, ca_t, ea_t,
                               cluster_mask=cluster_drop, exact_mask=exact_drop,
                               src_mask=src_drop)
                loss = F.mse_loss(v_pred, v_target)
                loss.backward()
                total_loss += loss.item()
                n_micro += 1

        if n_micro == 0:
            continue
        torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(skel_enc.parameters(), 1.0)
        optim.step()
        sched.step()
        avg_loss = total_loss / n_micro
        losses_window.append(avg_loss)
        if len(losses_window) > 200:
            losses_window = losses_window[-200:]

        if (step + 1) % args.log_interval == 0:
            elapsed = time.time() - t0
            eta = elapsed / max(1, step + 1 - start_step) * (args.max_steps - step - 1)
            ravg = np.mean(losses_window)
            cur_lr = sched.get_last_lr()[0]
            print(f"  [{step+1:6d}/{args.max_steps}] loss={avg_loss:.4f} avg200={ravg:.4f} "
                  f"lr={cur_lr:.2e} ({elapsed:.0f}s, ETA {eta/60:.0f}min)")
            with open(log_path, 'a') as f:
                f.write(json.dumps({'step': step + 1, 'loss': avg_loss,
                                    'avg200': float(ravg), 'lr': cur_lr}) + '\n')

        if (step + 1) % args.ckpt_interval == 0 and (step + 1) < args.max_steps:
            ck = save_dir / f'ckpt_step{step+1:07d}.pt'
            tmp = ck.with_suffix('.pt.tmp')
            torch.save({
                'step': step + 1, 'G': G.state_dict(), 'skel_enc': skel_enc.state_dict(),
                'optim': optim.state_dict(), 'rng': rng.get_state(),
                'skels': skels, 'skel_to_id': skel_to_id,
                'n_clusters': N_CLUSTERS, 'clusters': CLUSTERS,
                'n_exact_actions': N_EXACT_ACTIONS, 'exact_actions': EXACT_ACTIONS,
            }, tmp)
            os.replace(tmp, ck)
            for old in sorted(save_dir.glob('ckpt_step*.pt'))[:-2]:
                old.unlink()

    # Final checkpoint
    final = save_dir / 'ckpt_final.pt'
    tmp = final.with_suffix('.pt.tmp')
    torch.save({
        'step': args.max_steps, 'G': G.state_dict(), 'skel_enc': skel_enc.state_dict(),
        'args': vars(args), 'skels': skels, 'skel_to_id': skel_to_id,
        'n_clusters': N_CLUSTERS, 'clusters': CLUSTERS,
        'n_exact_actions': N_EXACT_ACTIONS, 'exact_actions': EXACT_ACTIONS,
    }, tmp)
    os.replace(tmp, final)
    for stale in save_dir.glob('ckpt_step*.pt'):
        stale.unlink()
    print(f"\nSaved final ckpt: {final}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--variant', choices=VARIANTS, default='labels',
                        help='labels = AL-Flow; labels_source = AL-Flow-Src; '
                             'labels_source_graph = AL-Flow-Src-G')
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--latent_cache', type=str, default=str(DEFAULT_CACHE))
    parser.add_argument('--max_steps', type=int, default=50000)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--warmup', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--p_cluster_drop', type=float, default=0.10,
                        help='how often the coarse action label is withheld during training')
    parser.add_argument('--p_exact_drop', type=float, default=0.10,
                        help='how often the exact action name is withheld during training')
    parser.add_argument('--p_src_drop', type=float, default=0.10,
                        help='how often the source clip is withheld during training; '
                             'only the variants that see one')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_interval', type=int, default=200)
    parser.add_argument('--ckpt_interval', type=int, default=2500)
    parser.add_argument('--no_resume', action='store_true')
    parser.add_argument('--config', default=None, type=str,
                        help='file of settings; anything also typed on the command line wins')
    args = parser.parse_args()
    if args.config:
        apply_config(args, args.config)

    if args.run_name is None:
        args.run_name = {'labels': 'al_flow',
                         'labels_source': 'al_flow_src',
                         'labels_source_graph': 'al_flow_src_g'}[args.variant]

    train(args)


if __name__ == '__main__':
    main()
