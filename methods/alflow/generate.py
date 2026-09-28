"""Produce an AL-Flow model's answer to every query in a benchmark set.

A query names one clip performed by one animal and asks for a motion on another animal. The
answer is written as query_XXXX.npy, one file per query, which is what the scoring scripts
read.

The model is told the action to perform, in the same two forms the retrieval method ANCHOR
uses, so that the comparison between them is about the method and not about what each was
told. The coarse cluster is guessed from the source clip's motion by the shared classifier
in benchmark/action_classifier.py, never read off the clip's filename. The
exact action name comes from the benchmark set's own record of the source clip.

From there the model starts at noise and moves towards a motion for the target animal, by
default in a single step. The result is decoded with the target animal's tokenizer and cut
to the typical length of the clips the query will be compared against.

--variant says which of the three trained models is being run, and what else it is given:

  labels              the action and the target animal only
  labels_source       also the source clip, encoded with its own tokenizer, and which
                      animal performed it
  labels_source_graph the same, through the model that never looks an animal up by name, so
                      queries about animals it never trained on are answered rather than
                      skipped

    python -m methods.alflow.generate --variant labels --set 49 \\
        --ckpt save/alflow/al_flow/ckpt_final.pt --out_dir outputs/al_flow/set49

With --fold 42 or --fold 43 in place of --set, it answers one of the two query folds of the
action-level test in benchmark/queries instead.
"""
from __future__ import annotations
import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from core.truebones.param_utils import OBJECT_SUBSETS_DICT, DATASET_DIR
from benchmark.action_classifier import (
    train_classifier, feature_vector, CLUSTER_TO_IDX as CLASSIFIER_CLUSTER_TO_IDX,
)
from benchmark.clip_features import CLIP_FEATURES_PATH, open_clip_features
from methods.alflow.generator import (
    ALFlowGenerator, ALFlowSrcGenerator, ALFlowSrcGraphGenerator,
)
# The label vocabulary comes from the training script, so the two always agree
from methods.alflow.train import VARIANTS, CLUSTER_TO_IDX, EXACT_ACTION_TO_IDX
from methods.common.queries import reference_length
from methods.tokenizer.skel_graph import (
    SkelGraphEncoder, build_skel_features, pad_to_max_joints,
)
from methods.tokenizer.registry import TokenizerRegistry

ROOT = Path(__file__).resolve().parents[2]
SETS = ROOT / 'benchmark' / 'sets'
QUERIES_ROOT = ROOT / 'benchmark' / 'queries'
DATA_ROOT = ROOT / DATASET_DIR

CACHE_WINDOW = 32  # must match methods/moreflow/build_cache.py


def load_checkpoint(variant, ckpt_path, device):
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = sd.get('args', {})
    skels = sd['skels']
    skel_to_id = sd['skel_to_id']
    n_clusters = sd['n_clusters']
    n_exact_actions = sd['n_exact_actions']

    common = dict(
        n_clusters=n_clusters,
        n_exact_actions=n_exact_actions,
        codebook_dim=256,
        d_model=args.get('d_model', 512),
        n_layers=args.get('n_layers', 6),
        n_heads=args.get('n_heads', 8),
    )
    if variant == 'labels':
        G = ALFlowGenerator(n_skels=len(skels), **common).to(device)
    elif variant == 'labels_source':
        G = ALFlowSrcGenerator(n_skels=len(skels), **common).to(device)
    else:
        G = ALFlowSrcGraphGenerator(**common).to(device)
    G.load_state_dict(sd['G'])
    G.eval()

    skel_enc = SkelGraphEncoder().to(device)
    skel_enc.load_state_dict(sd['skel_enc'])
    skel_enc.eval()

    return G, skel_enc, skels, skel_to_id, args


def sample_labels(G, z_init, tgt_id, tgt_graph, cluster_id, exact_id,
                  n_steps=1, cfg_scale=1.0):
    """Euler sampling for the labels variant.

    n_steps=1 takes a single step from the noise; n_steps=k integrates the flow in k steps.
    cfg_scale mixes the conditional and unconditional velocities; 1.0 means no guidance.
    """
    z = z_init
    if n_steps == 1:
        t = torch.zeros(z.shape[0], device=z.device)
        v_cond = G(z, t, tgt_id, tgt_graph, cluster_id, exact_id)
        if cfg_scale != 1.0:
            cmask = torch.ones_like(cluster_id, dtype=torch.bool)
            emask = torch.ones_like(exact_id, dtype=torch.bool)
            v_uncond = G(z, t, tgt_id, tgt_graph, cluster_id, exact_id,
                         cluster_mask=cmask, exact_mask=emask)
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            v = v_cond
        return z + v
    dt = 1.0 / n_steps
    for k in range(n_steps):
        t = torch.full((z.shape[0],), k * dt, device=z.device)
        v_cond = G(z, t, tgt_id, tgt_graph, cluster_id, exact_id)
        if cfg_scale != 1.0:
            cmask = torch.ones_like(cluster_id, dtype=torch.bool)
            emask = torch.ones_like(exact_id, dtype=torch.bool)
            v_uncond = G(z, t, tgt_id, tgt_graph, cluster_id, exact_id,
                         cluster_mask=cmask, exact_mask=emask)
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            v = v_cond
        z = z + v * dt
    return z


def sample_labels_source(G, z_init, tgt_id, tgt_graph, src_z, src_id, src_graph,
                         cluster_id, exact_id, n_steps=1, cfg_scale=1.0):
    """Euler sampling for the source-conditioned variant, with optional guidance."""
    z = z_init
    if n_steps == 1:
        t = torch.zeros(z.shape[0], device=z.device)
        v_cond = G(z, t, tgt_id, tgt_graph, src_z, src_id, src_graph,
                   cluster_id, exact_id)
        if cfg_scale != 1.0:
            cmask = torch.ones_like(cluster_id, dtype=torch.bool)
            emask = torch.ones_like(exact_id, dtype=torch.bool)
            smask = torch.ones_like(cluster_id, dtype=torch.bool)
            v_uncond = G(z, t, tgt_id, tgt_graph, src_z, src_id, src_graph,
                         cluster_id, exact_id,
                         cluster_mask=cmask, exact_mask=emask, src_mask=smask)
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            v = v_cond
        return z + v
    dt = 1.0 / n_steps
    for k in range(n_steps):
        t = torch.full((z.shape[0],), k * dt, device=z.device)
        v_cond = G(z, t, tgt_id, tgt_graph, src_z, src_id, src_graph,
                   cluster_id, exact_id)
        if cfg_scale != 1.0:
            cmask = torch.ones_like(cluster_id, dtype=torch.bool)
            emask = torch.ones_like(exact_id, dtype=torch.bool)
            smask = torch.ones_like(cluster_id, dtype=torch.bool)
            v_uncond = G(z, t, tgt_id, tgt_graph, src_z, src_id, src_graph,
                         cluster_id, exact_id,
                         cluster_mask=cmask, exact_mask=emask, src_mask=smask)
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            v = v_cond
        z = z + v * dt
    return z


def sample_labels_source_graph(G, z_init, tgt_graph, src_z, src_graph,
                               cluster_id, exact_id, n_steps=1):
    """Euler sampling for the graph-only variant."""
    z = z_init
    if n_steps == 1:
        t = torch.zeros(z.shape[0], device=z.device)
        v = G(z, t, tgt_graph, src_z, src_graph, cluster_id, exact_id)
        return z + v
    dt = 1.0 / n_steps
    for k in range(n_steps):
        t = torch.full((z.shape[0],), k * dt, device=z.device)
        v = G(z, t, tgt_graph, src_z, src_graph, cluster_id, exact_id)
        z = z + v * dt
    return z


def encode_source(registry, skel_a, motion_phys):
    """Encode a clip with its own skeleton's tokenizer, into 8 tokens.

    This follows the cache build exactly: crop or hold the last frame to reach
    CACHE_WINDOW frames, normalize, replace any non-finite value, then encode. At a
    downsample factor of 4 a 32-frame window gives the 8 tokens the model trained on.
    """
    if skel_a not in registry.tokenizers:
        return None
    motion_t = torch.as_tensor(motion_phys, dtype=torch.float32, device=registry.device)
    if motion_t.dim() == 3:
        motion_t = motion_t.unsqueeze(0)  # [1, T, J, 13]
    T = motion_t.shape[1]
    if T < CACHE_WINDOW:
        pad_T = CACHE_WINDOW - T
        last = motion_t[:, -1:, :, :]
        pad = last.expand(-1, pad_T, -1, -1)
        motion_t = torch.cat([motion_t, pad], dim=1)
    elif T > CACHE_WINDOW:
        motion_t = motion_t[:, :CACHE_WINDOW, :, :]
    with torch.no_grad():
        motion_norm = registry.normalize(skel_a, motion_t)
        motion_norm = torch.nan_to_num(motion_norm)
        z, _ = registry.encode_window(skel_a, motion_norm)  # [1, T_tokens, codebook_dim]
    return z[0]  # [T_tokens, codebook_dim]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--variant', choices=VARIANTS, default='labels')
    parser.add_argument('--ckpt', type=str, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--set', choices=['49', '1891'], default='1891',
                       help='benchmark set, read from benchmark/sets/truebones_<set>.json')
    group.add_argument('--fold', type=int, choices=[42, 43],
                       help='answer one of the two query folds in benchmark/queries instead')
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--max_queries', type=int, default=10000)
    parser.add_argument('--motion_dir', type=str,
                        default='dataset/truebones/zoo/truebones_processed/motions')
    parser.add_argument('--clip_features', type=str, default=str(CLIP_FEATURES_PATH),
                        help='the per-clip measurements the action classifier is fitted on')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_euler_steps', type=int, default=1,
                        help='how many steps to take along the flow; 1 goes straight there')
    parser.add_argument('--cfg_scale', type=float, default=1.0,
                        help='how strongly to weigh the action labels; 1.0 leaves them as they '
                             'are. The graph-only variant ignores this, as the paper ran '
                             'it that way')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}, variant: {args.variant}")

    np.random.seed(args.seed); torch.manual_seed(args.seed)

    print(f"Loading ckpt: {args.ckpt}")
    G, skel_enc, skels, skel_to_id, ckpt_args = load_checkpoint(args.variant, args.ckpt, device)
    print(f"  checkpoint skeletons: {len(skels)}")

    if args.fold is not None:
        manifest_path = QUERIES_ROOT / f'fold_{args.fold}' / 'manifest.json'
    else:
        manifest_path = SETS / f'truebones_{args.set}.json'
    with open(manifest_path) as f:
        manifest = json.load(f)
    queries = manifest['queries'][:args.max_queries]
    print(f"{manifest_path.relative_to(ROOT)}: {len(queries)} queries")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")

    cond_dict = np.load(ROOT / DATASET_DIR / 'cond.npy', allow_pickle=True).item()

    # Fit the action cluster classifier, exactly as ANCHOR does
    print("Fitting the action cluster classifier on the training skeletons...")
    clip_features = open_clip_features(args.clip_features)
    train_skels_set = set(OBJECT_SUBSETS_DICT['train'])
    clf = train_classifier(clip_features, train_skels_set)
    fname_to_clip_idx = {m['fname']: i for i, m in enumerate(clip_features['meta'])}
    idx_to_cluster = {v: k for k, v in CLASSIFIER_CLUSTER_TO_IDX.items()}
    print(f"  classifier ready; {len(fname_to_clip_idx)} clips measured")

    # Skeleton-graph features
    if args.variant == 'labels_source_graph':
        # No ids are looked up, so every skeleton in the set gets graph features
        print("Building skeleton-graph features for every skeleton in the set...")
        skel_features = build_skel_features(cond_dict)
        all_skels_in_queries = sorted(set([q['skel_a'] for q in queries] +
                                          [q['skel_b'] for q in queries]))
        available = [s for s in all_skels_in_queries if s in skel_features]
        print(f"  available: {len(available)}/{len(all_skels_in_queries)}")
        max_J = max(skel_features[s]['n_joints'] for s in available)
        graph_skels = available
    else:
        print("Building skeleton-graph features...")
        skel_features = build_skel_features(cond_dict)
        max_J = max(skel_features[s]['n_joints'] for s in skels if s in skel_features)
        graph_skels = [s for s in skels if s in skel_features]

    pj_padded, pj_mask, pj_agg = {}, {}, {}
    for s in graph_skels:
        p, m = pad_to_max_joints(skel_features[s]['per_joint'], max_J)
        pj_padded[s] = p.to(device)
        pj_mask[s] = m.to(device)
        pj_agg[s] = skel_features[s]['agg'].to(device)

    # Tokenizers: the target ones are needed to decode, the source ones to encode
    if args.variant == 'labels':
        needed_skels = sorted({q['skel_b'] for q in queries})
        print(f"Loading the tokenizers of {len(needed_skels)} target skeletons...")
    elif args.variant == 'labels_source':
        needed_skels = sorted(set([q['skel_a'] for q in queries] + [q['skel_b'] for q in queries]))
        print(f"Loading the tokenizers of {len(needed_skels)} skeletons (source and target)...")
    else:
        needed_skels = graph_skels
        print(f"Loading the tokenizers of {len(needed_skels)} skeletons...")
    registry = TokenizerRegistry(needed_skels, device=str(device))

    per_query = []
    n_token = 8  # tokens per window, as the tokenizers produce
    codebook_dim = 256
    t_total_0 = time.time()
    motion_dir = ROOT / args.motion_dir

    for i, q in enumerate(queries):
        qid = q['query_id']
        skel_b = q['skel_b']
        skel_a = q.get('skel_a', '')
        src_action = q.get('src_action', '')
        cluster_gt = q.get('cluster', '')
        split = q.get('split', '')

        rec = {'query_id': qid, 'cluster': cluster_gt, 'split': split,
               'skel_a': skel_a, 'skel_b': skel_b,
               'src_action': src_action, 'status': 'pending'}

        try:
            if args.variant == 'labels_source_graph':
                # Nothing is looked up by id, so only the graph features must exist
                if skel_b not in pj_padded:
                    rec['status'] = 'skipped_no_graph_for_target'
                    per_query.append(rec)
                    continue
                if skel_a not in pj_padded:
                    rec['status'] = 'skipped_no_graph_for_source'
                    per_query.append(rec)
                    continue
            else:
                if skel_b not in skel_to_id:
                    rec['status'] = 'skipped_target_not_trained'
                    per_query.append(rec)
                    continue
                if skel_b not in registry.tokenizers:
                    rec['status'] = 'skipped_no_tokenizer_for_target'
                    per_query.append(rec)
                    continue
                if args.variant == 'labels_source':
                    if skel_a not in skel_to_id:
                        rec['status'] = 'skipped_source_not_trained'
                        per_query.append(rec)
                        continue
                    if skel_a not in registry.tokenizers:
                        rec['status'] = 'skipped_no_tokenizer_for_source'
                        per_query.append(rec)
                        continue

            # Predict the source clip's cluster from its motion
            src_fname = q['src_fname']
            if src_fname in fname_to_clip_idx:
                ci = fname_to_clip_idx[src_fname]
                feat = feature_vector(clip_features['com_path'][ci],
                                      clip_features['heading_vel'][ci],
                                      clip_features['contact_sched'][ci],
                                      clip_features['cadence'][ci],
                                      clip_features['limb_usage'][ci])
            else:
                # This clip has no measurements, so no action is given
                feat = None
            if feat is not None:
                pred_cluster_idx = int(clf.predict(feat.reshape(1, -1))[0])
                pred_cluster_str = idx_to_cluster.get(pred_cluster_idx, '')
            else:
                pred_cluster_str = ''
            cluster_id_int = CLUSTER_TO_IDX.get(pred_cluster_str, 0)

            # The exact action is the benchmark set's own label for the source clip
            exact_id_int = EXACT_ACTION_TO_IDX.get(src_action, 0)

            B = 1
            if args.variant == 'labels':
                z_init = torch.randn(B, n_token, codebook_dim, device=device)
                tgt_id = torch.full((B,), skel_to_id[skel_b], dtype=torch.long, device=device)
                tgt_graph = skel_enc(pj_padded[skel_b].unsqueeze(0),
                                     pj_mask[skel_b].unsqueeze(0),
                                     pj_agg[skel_b].unsqueeze(0))  # [1, d_graph]
                cid = torch.full((B,), cluster_id_int, dtype=torch.long, device=device)
                eid = torch.full((B,), exact_id_int, dtype=torch.long, device=device)

                with torch.no_grad():
                    z_b = sample_labels(G, z_init, tgt_id, tgt_graph, cid, eid,
                                        n_steps=args.n_euler_steps, cfg_scale=args.cfg_scale)
            else:
                # Both source-conditioned variants encode the source clip first
                src_path = motion_dir / src_fname
                if not src_path.exists():
                    rec['status'] = f'skipped_source_motion_missing: {src_fname}'
                    per_query.append(rec)
                    continue
                src_motion = np.load(src_path).astype(np.float32)
                src_z = encode_source(registry, skel_a, src_motion)
                if src_z is None or src_z.shape[0] != n_token:
                    rec['status'] = 'skipped_source_encoding_failed'
                    per_query.append(rec)
                    continue

                if args.variant == 'labels_source':
                    z_init = torch.randn(B, n_token, codebook_dim, device=device)
                    tgt_id = torch.full((B,), skel_to_id[skel_b], dtype=torch.long, device=device)
                    sid = torch.full((B,), skel_to_id[skel_a], dtype=torch.long, device=device)
                    tgt_graph = skel_enc(pj_padded[skel_b].unsqueeze(0),
                                         pj_mask[skel_b].unsqueeze(0),
                                         pj_agg[skel_b].unsqueeze(0))
                    src_graph = skel_enc(pj_padded[skel_a].unsqueeze(0),
                                         pj_mask[skel_a].unsqueeze(0),
                                         pj_agg[skel_a].unsqueeze(0))
                    cid = torch.full((B,), cluster_id_int, dtype=torch.long, device=device)
                    eid = torch.full((B,), exact_id_int, dtype=torch.long, device=device)
                    src_z_b = src_z.unsqueeze(0)  # [1, 8, 256]

                    with torch.no_grad():
                        z_b = sample_labels_source(G, z_init, tgt_id, tgt_graph,
                                                   src_z_b, sid, src_graph, cid, eid,
                                                   n_steps=args.n_euler_steps,
                                                   cfg_scale=args.cfg_scale)
                else:
                    tgt_graph = skel_enc(pj_padded[skel_b].unsqueeze(0),
                                         pj_mask[skel_b].unsqueeze(0),
                                         pj_agg[skel_b].unsqueeze(0))
                    src_graph = skel_enc(pj_padded[skel_a].unsqueeze(0),
                                         pj_mask[skel_a].unsqueeze(0),
                                         pj_agg[skel_a].unsqueeze(0))
                    cid = torch.full((B,), cluster_id_int, dtype=torch.long, device=device)
                    eid = torch.full((B,), exact_id_int, dtype=torch.long, device=device)
                    src_z_b = src_z.unsqueeze(0)
                    z_init = torch.randn(B, n_token, codebook_dim, device=device)

                    with torch.no_grad():
                        z_b = sample_labels_source_graph(G, z_init, tgt_graph, src_z_b, src_graph,
                                                         cid, eid, n_steps=args.n_euler_steps)

            # Decode with the target skeleton's tokenizer
            with torch.no_grad():
                motion_norm = registry.decode_tokens(skel_b, z_b)
                motion_phys = registry.unnormalize(skel_b, motion_norm)  # [B, T, J, 13]

            sample = motion_phys[0].cpu().numpy().astype(np.float32)  # [T, J, 13]

            # Crop to the median length of the target clips the query lists
            T_tgt = reference_length(q, sample.shape[0])
            T_out = min(T_tgt, sample.shape[0])
            sample = sample[:T_out]

            np.save(out_dir / f'query_{qid:04d}.npy', sample)
            rec['status'] = 'ok'
            rec['T_out'] = int(sample.shape[0])
            rec['predicted_cluster'] = pred_cluster_str
            rec['exact_id'] = int(exact_id_int)

        except Exception as e:
            rec['status'] = f'error: {type(e).__name__}: {e}'
            rec['traceback'] = traceback.format_exc()

        per_query.append(rec)

        if (i + 1) % 25 == 0:
            elapsed = time.time() - t_total_0
            rate = (i + 1) / max(1, elapsed)
            eta = (len(queries) - i - 1) / max(1e-6, rate)
            ok = sum(1 for r in per_query if r['status'] == 'ok')
            print(f"  [{i+1:4d}/{len(queries)}] ok={ok} elapsed={elapsed:.0f}s "
                  f"rate={rate:.1f}/s ETA={eta:.0f}s")

    summary = {
        'method': {'labels': 'AL-Flow',
                   'labels_source': 'AL-Flow-Src',
                   'labels_source_graph': 'AL-Flow-Src-G'}[args.variant],
        'ckpt': str(args.ckpt),
        'set': args.set if args.fold is None else None,
        'fold': args.fold,
        'manifest': str(manifest_path),
        'n_queries': len(queries),
        'n_ok': sum(1 for r in per_query if r['status'] == 'ok'),
        'n_skipped': sum(1 for r in per_query if r['status'].startswith('skipped')),
        'n_error': sum(1 for r in per_query if r['status'].startswith('error')),
        'n_euler_steps': args.n_euler_steps,
        'cfg_scale': args.cfg_scale,
        'wall_clock_s': time.time() - t_total_0,
        'per_query': per_query,
    }
    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_dir}/metrics.json")
    print(f"  ok={summary['n_ok']} / skipped={summary['n_skipped']} / error={summary['n_error']}")
    print(f"  wall_clock={summary['wall_clock_s']:.0f}s")


if __name__ == '__main__':
    main()
