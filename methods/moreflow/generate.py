"""Produce MoReFlow's answer to every query in a benchmark set.

A query names one clip performed by one animal and asks for the same motion on another
animal. The answer is written as query_XXXX.npy, one file per query, which is what the
scoring scripts read.

The source clip is encoded with its own tokenizer and cut into short chunks that overlap by
half. For each chunk the model is asked, repeatedly, which way to move, and the chunk is
carried from the source animal's codes to the target animal's in twenty-five steps of a
second-order rule. Four requests are made at once, each asking the motion to keep a
different property of the source, and their answers are averaged at every step; each request
is also weighed against what the model would do unprompted, which sharpens its effect.
Overlapping chunks are averaged, the result is snapped to the target's codebook and decoded
back into motion.

Nothing here is random: the flow starts at the encoded source, not at noise, so the same
checkpoint and the same query always give the same answer.

    python -m methods.moreflow.generate --set 49 \\
        --ckpt save/moreflow/moreflow_t/ckpt_final.pt --out_dir outputs/moreflow_t/set49

With --fold 42 or --fold 43 in place of --set, it answers one of the two query folds of the
action-level test in benchmark/queries instead.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from core.truebones.param_utils import OBJECT_SUBSETS_DICT, DATASET_DIR
from methods.moreflow.flow_transformer import DiscreteFlowTransformer
from methods.moreflow.conditions import (
    COND_TYPE_TO_INT, DEFAULT_INFERENCE_CONDS, DEFAULT_CFG_GAMMA, DEFAULT_HEUN_STEPS,
)
from methods.tokenizer.skel_graph import (
    SkelGraphEncoder, build_skel_features, pad_to_max_joints,
)
from methods.tokenizer.registry import TokenizerRegistry
from methods.common.motion_descriptors import (
    CONDITIONS, D_COND_PADDED, G_MAX,
    phi as phi_np, precompute_skel_descriptors, load_contact_groups,
)

ROOT = Path(__file__).resolve().parents[2]
SETS = ROOT / 'benchmark' / 'sets'
QUERIES_ROOT = ROOT / 'benchmark' / 'queries'
DATA_ROOT = ROOT / DATASET_DIR
COND_PATH = DATA_ROOT / 'cond.npy'
MOTION_DIR = DATA_ROOT / 'motions'
WINDOW = 32
STRIDE = 4
TOKENS_PER_WINDOW = 8


def heun_integrate_ensemble(model, skel_enc, z_init, src_id, tgt_id, src_graph, tgt_graph,
                            cond_type_ints, cond_vecs, n_steps, cfg_gamma, device):
    """Integrate the velocity field from q=0 to q=1 with Heun's second-order method.

    The velocity fields of the conditions in the ensemble are averaged at every step, not
    once per integration; for a nonlinear flow the two are not the same thing.

    z_init:         [B, T_token, codebook_dim]
    cond_type_ints: list of int, one per condition in the ensemble
    cond_vecs:      list of Tensor[1, D], one per condition in the ensemble
    Returns z at q=1: [B, T_token, codebook_dim]
    """
    B, T_tok, d = z_init.shape
    z = z_init.clone()
    dq = 1.0 / n_steps

    src_id_t = torch.full((B,), src_id, dtype=torch.long, device=device)
    tgt_id_t = torch.full((B,), tgt_id, dtype=torch.long, device=device)

    def one_cond_velocity(z_cur, q_cur, cond_type_int, cond_vec):
        """Guided velocity for ONE condition at q_cur. Returns [B, T_tok, d]."""
        q_t = torch.full((B,), q_cur, device=device)
        cond_type_t = torch.full((B,), cond_type_int, dtype=torch.long, device=device)
        cond_vec_t = cond_vec.expand(B, -1).to(device)

        if cond_type_int == COND_TYPE_TO_INT['null']:
            # The null condition is the unconditional branch: always called with the mask
            # on, which zeroes the condition exactly as the dropout did during training.
            cond_mask = torch.ones(B, device=device)
            return model(z_cur, q_t, src_id_t, tgt_id_t, src_graph, tgt_graph,
                          cond_type_t, cond_vec_t, cond_mask)

        # Conditioned: mix the conditional and unconditional velocities
        cond_mask_off = torch.zeros(B, device=device)
        cond_mask_on = torch.ones(B, device=device)
        v_cond = model(z_cur, q_t, src_id_t, tgt_id_t, src_graph, tgt_graph,
                       cond_type_t, cond_vec_t, cond_mask_off)
        if cfg_gamma == 1.0:
            return v_cond
        v_uncond = model(z_cur, q_t, src_id_t, tgt_id_t, src_graph, tgt_graph,
                         cond_type_t, cond_vec_t, cond_mask_on)
        return (1.0 - cfg_gamma) * v_uncond + cfg_gamma * v_cond

    def ensemble_velocity(z_cur, q_cur):
        vs = [one_cond_velocity(z_cur, q_cur, ct, cv)
              for ct, cv in zip(cond_type_ints, cond_vecs)]
        return torch.stack(vs).mean(dim=0)

    for i in range(n_steps):
        q_cur = i * dq
        v1 = ensemble_velocity(z, q_cur)
        z_pred = z + dq * v1
        q_next = min(1.0 - 1e-6, q_cur + dq)
        v2 = ensemble_velocity(z_pred, q_next)
        z = z + 0.5 * dq * (v1 + v2)
    return z


@torch.no_grad()
def retarget_one_query(query, model, skel_enc, registry, skel_features, max_J,
                       skel_to_id, skel_descs, args, device):
    """Answer one query. Returns the motion as [frames, joints, 13]."""
    src_skel = query['skel_a']
    tgt_skel = query['skel_b']
    src_fname = query['src_fname']

    # A skeleton the model never trained on (the held-out ones, for a model trained without
    # them) has no id of its own. The embedding has no slot for it, so id 0 stands in and the
    # skeleton features, built from offsets, parents and depths, describe the body. That is
    # what the held-out animals test, and the result is reported as it comes out.
    UNK_ID = len(skel_to_id)  # one past the last trained id
    src_id = skel_to_id.get(src_skel, UNK_ID)
    tgt_id = skel_to_id.get(tgt_skel, UNK_ID)
    if src_id == UNK_ID or tgt_id == UNK_ID:
        if src_id == UNK_ID:
            src_id = 0
        if tgt_id == UNK_ID:
            tgt_id = 0

    # Skeleton graph embeddings
    src_pj = pad_to_max_joints(skel_features[src_skel]['per_joint'], max_J)[0].unsqueeze(0).to(device)
    src_m = pad_to_max_joints(skel_features[src_skel]['per_joint'], max_J)[1].unsqueeze(0).to(device)
    src_a = skel_features[src_skel]['agg'].unsqueeze(0).to(device)
    tgt_pj = pad_to_max_joints(skel_features[tgt_skel]['per_joint'], max_J)[0].unsqueeze(0).to(device)
    tgt_m = pad_to_max_joints(skel_features[tgt_skel]['per_joint'], max_J)[1].unsqueeze(0).to(device)
    tgt_a = skel_features[tgt_skel]['agg'].unsqueeze(0).to(device)
    src_graph = skel_enc(src_pj, src_m, src_a)
    tgt_graph = skel_enc(tgt_pj, tgt_m, tgt_a)

    # Load and encode the source motion
    src_path = MOTION_DIR / src_fname
    motion_phys = np.load(src_path).astype(np.float32)              # [T, J, 13]
    T = motion_phys.shape[0]
    # Crop T to a multiple of STRIDE so the token sequence is well defined
    T_crop = (T // STRIDE) * STRIDE
    motion_phys = motion_phys[:T_crop]
    motion_t = torch.from_numpy(motion_phys).to(device)

    # The encoder takes any length that is a multiple of the downsample factor
    motion_norm = registry.normalize(src_skel, motion_t.unsqueeze(0))      # [1, T_crop, J_src, 13]
    z_src_full, _ = registry.encode_window(src_skel, motion_norm.squeeze(0))  # [1, T_crop/4, d]
    z_src_full = z_src_full.squeeze(0)                                     # [T_token_full, d]
    T_token_full = z_src_full.shape[0]

    # Cut into 8-token chunks that overlap by half. A final chunk aligned to the end is
    # appended when the stride schedule does not already reach it, so no tail is dropped.
    chunk_token_stride = TOKENS_PER_WINDOW // 2
    if T_token_full <= TOKENS_PER_WINDOW:
        chunk_starts = [0]
    else:
        chunk_starts = list(range(0, T_token_full - TOKENS_PER_WINDOW + 1, chunk_token_stride))
        last_start = T_token_full - TOKENS_PER_WINDOW
        if chunk_starts[-1] != last_start:
            chunk_starts.append(last_start)
    n_chunks = len(chunk_starts)

    # Accumulator for the overlap average
    codebook_dim = registry.get(tgt_skel)['model'].codebook_dim
    z_tgt_acc = torch.zeros(T_token_full, codebook_dim, device=device)
    z_tgt_count = torch.zeros(T_token_full, device=device)

    # Each chunk's condition values are computed on that chunk of the source motion; the
    # per-step averaging over conditions happens inside heun_integrate_ensemble.
    for chunk_start in chunk_starts:
        chunk_end = min(chunk_start + TOKENS_PER_WINDOW, T_token_full)
        chunk_z = z_src_full[chunk_start:chunk_end]
        if chunk_end - chunk_start < TOKENS_PER_WINDOW:
            pad = TOKENS_PER_WINDOW - chunk_z.shape[0]
            chunk_z = torch.cat([chunk_z, torch.zeros(pad, chunk_z.shape[1], device=device)])
        chunk_z = chunk_z.unsqueeze(0)                                           # [1, 8, d]

        chunk_frame_start = chunk_start * STRIDE
        chunk_frame_end = chunk_frame_start + WINDOW
        if chunk_frame_end > T_crop:
            chunk_frame_end = T_crop
            chunk_frame_start = T_crop - WINDOW
        chunk_motion_phys = motion_phys[chunk_frame_start:chunk_frame_end]       # [WINDOW, J_src, 13]

        cond_type_ints = []
        cond_vecs = []
        for c_str in args.inference_conds:
            cond_type_ints.append(COND_TYPE_TO_INT[c_str])
            if c_str == 'null':
                cond_vecs.append(torch.zeros(1, D_COND_PADDED, device=device))
            else:
                cond_vec_np = phi_np(chunk_motion_phys, c_str, skel_descs[src_skel])
                cond_vecs.append(torch.from_numpy(cond_vec_np).float().unsqueeze(0).to(device))

        z_tgt_chunk = heun_integrate_ensemble(
            model, skel_enc, chunk_z, src_id, tgt_id, src_graph, tgt_graph,
            cond_type_ints, cond_vecs,
            args.heun_steps, args.cfg_gamma, device,
        ).squeeze(0)                                                              # [8, d]

        chunk_len_actual = chunk_end - chunk_start
        z_tgt_acc[chunk_start:chunk_end] += z_tgt_chunk[:chunk_len_actual]
        z_tgt_count[chunk_start:chunk_end] += 1.0

    # Average the overlaps. A count of zero would mean the chunking missed a position,
    # which the tail chunk above prevents; warn if it ever happens.
    if (z_tgt_count == 0).any():
        n_missed = int((z_tgt_count == 0).sum())
        print(f"  WARN: {n_missed}/{T_token_full} token positions missed by the overlap "
              f"average — check the chunk starts.")
    z_tgt_avg = z_tgt_acc / z_tgt_count.clamp(min=1.0).unsqueeze(-1)

    # Quantize to the target codebook, over its valid entries only
    cb = registry.codebook(tgt_skel, padded_to=0)                              # [K_eff, d]
    dists = torch.cdist(z_tgt_avg.unsqueeze(0), cb.unsqueeze(0))                # [1, T_token, K_eff]
    indices = dists.argmin(dim=-1).squeeze(0)                                   # [T_token_full]

    # Decode with the target tokenizer
    indices = indices.unsqueeze(0)                                              # [1, T_token_full]
    decoded = registry.decode_tokens(tgt_skel, indices)                         # [1, T, J_tgt, 13]
    decoded_phys = registry.unnormalize(tgt_skel, decoded).squeeze(0)           # [T, J_tgt, 13]

    return decoded_phys.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--ckpt', type=str, required=True,
                        help='the trained model to run')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--set', choices=['49', '1891'], default='1891',
                       help='benchmark set, read from benchmark/sets/truebones_<set>.json')
    group.add_argument('--fold', type=int, choices=[42, 43],
                       help='answer one of the two query folds in benchmark/queries instead')
    parser.add_argument('--out_dir', type=str, required=True,
                        help='where the query_XXXX.npy files are written')
    parser.add_argument('--inference_conds', type=str, nargs='+',
                        default=DEFAULT_INFERENCE_CONDS,
                        help='which properties of the source motion to ask the model to keep')
    parser.add_argument('--cfg_gamma', type=float, default=DEFAULT_CFG_GAMMA)
    parser.add_argument('--heun_steps', type=int, default=DEFAULT_HEUN_STEPS)
    parser.add_argument('--max_queries', type=int, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Loading ckpt: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    skels = ckpt['skels']
    skel_to_id = ckpt['skel_to_id']
    print(f"Trained on {len(skels)} skeletons")

    model = DiscreteFlowTransformer(n_skels=len(skels)).to(device)
    skel_enc = SkelGraphEncoder().to(device)
    model.load_state_dict(ckpt['model'])
    skel_enc.load_state_dict(ckpt['skel_enc'])
    model.eval()
    skel_enc.eval()

    # A query may target any of the 70 skeletons, so every tokenizer is loaded.
    print("Loading the tokenizers for all 70 skeletons...")
    registry = TokenizerRegistry(list(OBJECT_SUBSETS_DICT['all']), device=device)

    cond_dict = np.load(COND_PATH, allow_pickle=True).item()
    contact_groups = load_contact_groups()
    skel_descs = precompute_skel_descriptors(cond_dict, contact_groups)
    skel_features = build_skel_features(cond_dict)
    max_J = max(skel_features[s]['n_joints'] for s in skels)

    if args.fold is not None:
        manifest_path = QUERIES_ROOT / f'fold_{args.fold}' / 'manifest.json'
    else:
        manifest_path = SETS / f'truebones_{args.set}.json'
    print(f"Loading {manifest_path}")
    with open(manifest_path) as f:
        manifest = json.load(f)
    queries = manifest['queries']
    if args.max_queries:
        queries = queries[:args.max_queries]
    print(f"Conditions: {args.inference_conds}, guidance={args.cfg_gamma}, "
          f"Heun steps={args.heun_steps}")
    print(f"Processing {len(queries)} queries...")

    n_done = 0
    n_skipped = 0
    t0 = time.time()
    for i, q in enumerate(queries):
        out_path = out_dir / f"query_{q['query_id']:04d}.npy"
        if out_path.exists():
            n_done += 1
            continue
        try:
            result = retarget_one_query(q, model, skel_enc, registry, skel_features,
                                         max_J, skel_to_id, skel_descs, args, device)
            if result is None:
                n_skipped += 1
                continue
            np.save(out_path, result.astype(np.float32))
            n_done += 1
        except Exception as e:
            print(f"  query {q['query_id']} ({q['skel_a']}->{q['skel_b']}): FAILED — {e}")
            import traceback
            traceback.print_exc()
            n_skipped += 1

        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(queries) - i - 1)
            print(f"  [{i+1}/{len(queries)}] done={n_done}, skipped={n_skipped}, "
                  f"({elapsed:.0f}s, ETA {eta/60:.0f}min)")

    print(f"\nFinished: {n_done}/{len(queries)} queries written, {n_skipped} skipped.")
    print(f"Output dir: {out_dir}")


if __name__ == '__main__':
    main()
