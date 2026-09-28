"""Produce ACE's answer to every query in a benchmark set.

A query names one clip performed by one animal and asks for the same motion on another
animal. The answer is written as query_XXXX.npy, one file per query, which is what the
scoring scripts read.

The source clip is encoded with its own tokenizer, cut into short chunks that overlap by
half, and the model predicts the target chunk by chunk. Each chunk is given the model's own
previous prediction as context; the first chunk has none, so a starting vector belonging to
the target animal stands in. Overlapping predictions are averaged, and the result is decoded
with the target animal's tokenizer.

Nothing here is random: the same checkpoint and the same query always give the same answer.

    python -m methods.ace.generate --set 49 \\
        --ckpt save/ace/ace_t/ckpt_final.pt --out_dir outputs/ace_t/set49

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
from methods.ace.generator import ACEGenerator, ACEStartTokens
from methods.tokenizer.skel_graph import SkelGraphEncoder, build_skel_features, pad_to_max_joints
from methods.tokenizer.registry import TokenizerRegistry

ROOT = Path(__file__).resolve().parents[2]
SETS = ROOT / 'benchmark' / 'sets'
QUERIES_ROOT = ROOT / 'benchmark' / 'queries'
DATA_ROOT = ROOT / DATASET_DIR
COND_PATH = DATA_ROOT / 'cond.npy'
MOTION_DIR = DATA_ROOT / 'motions'
DEFAULT_LATENT_CACHE = ROOT / 'save/latents/cache_all.pt'
WINDOW = 32
STRIDE = 4
TOKENS_PER_WINDOW = 8


@torch.no_grad()
def retarget_one_query(query, G, skel_enc, starts, registry, skel_features, max_J,
                       skel_to_id, device, zero_source=False, fixed_length=None):
    """Answer one query. Returns the motion as [frames, joints, 13].

    zero_source: hand the model zeros where the source clip would go, which is the control
    in methods/ace/controls.py.
    fixed_length: make the answer exactly this many frames instead of following the source
    clip; the source is cropped when it is longer and its last frame held when shorter.
    """
    src_skel = query['skel_a']
    tgt_skel = query['skel_b']
    src_fname = query['src_fname']

    UNK_ID = len(skel_to_id)
    src_id = skel_to_id.get(src_skel, UNK_ID)
    tgt_id = skel_to_id.get(tgt_skel, UNK_ID)
    if src_id == UNK_ID or tgt_id == UNK_ID:
        # A skeleton the model never trained on has no id, so fall back to id 0 and let the
        # graph features carry the morphology.
        if src_id == UNK_ID: src_id = 0
        if tgt_id == UNK_ID: tgt_id = 0

    # Skeleton graph features
    src_pj_pad, src_pj_mask = pad_to_max_joints(skel_features[src_skel]['per_joint'], max_J)
    tgt_pj_pad, tgt_pj_mask = pad_to_max_joints(skel_features[tgt_skel]['per_joint'], max_J)
    src_pj_pad = src_pj_pad.unsqueeze(0).to(device)
    src_pj_mask = src_pj_mask.unsqueeze(0).to(device)
    src_a = skel_features[src_skel]['agg'].unsqueeze(0).to(device)
    tgt_pj_pad = tgt_pj_pad.unsqueeze(0).to(device)
    tgt_pj_mask = tgt_pj_mask.unsqueeze(0).to(device)
    tgt_a = skel_features[tgt_skel]['agg'].unsqueeze(0).to(device)
    src_graph = skel_enc(src_pj_pad, src_pj_mask, src_a)
    tgt_graph = skel_enc(tgt_pj_pad, tgt_pj_mask, tgt_a)

    # Load and encode the source motion
    motion_phys = np.load(MOTION_DIR / src_fname).astype(np.float32)
    T = motion_phys.shape[0]
    if fixed_length is not None:
        assert fixed_length % STRIDE == 0, f"fixed_length={fixed_length} must be a multiple of {STRIDE}"
        if T >= fixed_length:
            motion_phys = motion_phys[:fixed_length]
        else:
            pad = fixed_length - T
            tail = np.repeat(motion_phys[-1:], pad, axis=0)
            motion_phys = np.concatenate([motion_phys, tail], axis=0)
        T_crop = fixed_length
    else:
        T_crop = (T // STRIDE) * STRIDE
        motion_phys = motion_phys[:T_crop]
    motion_t = torch.from_numpy(motion_phys).to(device)

    motion_norm = registry.normalize(src_skel, motion_t.unsqueeze(0))
    z_src_full, _ = registry.encode_window(src_skel, motion_norm.squeeze(0))
    z_src_full = z_src_full.squeeze(0)                                    # [T_token_full, codebook_dim]
    T_token_full = z_src_full.shape[0]

    # Cut into 8-token chunks that overlap by half
    chunk_token_stride = TOKENS_PER_WINDOW // 2
    if T_token_full <= TOKENS_PER_WINDOW:
        chunk_starts = [0]
    else:
        chunk_starts = list(range(0, T_token_full - TOKENS_PER_WINDOW + 1, chunk_token_stride))
        last_start = T_token_full - TOKENS_PER_WINDOW
        if chunk_starts[-1] != last_start:
            chunk_starts.append(last_start)

    codebook_dim = registry.get(tgt_skel)['model'].codebook_dim
    z_tgt_acc = torch.zeros(T_token_full, codebook_dim, device=device)
    z_tgt_count = torch.zeros(T_token_full, device=device)

    src_id_t = torch.tensor([src_id], dtype=torch.long, device=device)
    tgt_id_t = torch.tensor([tgt_id], dtype=torch.long, device=device)

    # The first chunk's previous target is the target's START vector: a per-target mean plus
    # a learned offset, which does not depend on the source.
    prev_z_pred = starts(tgt_id_t)                                        # [1, 8, codebook_dim]

    for n, chunk_start in enumerate(chunk_starts):
        chunk_end = min(chunk_start + TOKENS_PER_WINDOW, T_token_full)
        chunk_z = z_src_full[chunk_start:chunk_end]
        if chunk_end - chunk_start < TOKENS_PER_WINDOW:
            pad = TOKENS_PER_WINDOW - chunk_z.shape[0]
            chunk_z = torch.cat([chunk_z, torch.zeros(pad, chunk_z.shape[1], device=device)])
        chunk_z = chunk_z.unsqueeze(0)                                     # [1, 8, codebook_dim]

        if zero_source:
            chunk_z = torch.zeros_like(chunk_z)

        z_pred = G(chunk_z, prev_z_pred, src_id_t, tgt_id_t, src_graph, tgt_graph)
        # The next chunk's previous target is this chunk's prediction. Detached, so that
        # calling this from a context with gradients does not chain graphs across chunks.
        prev_z_pred = z_pred.detach()                                      # [1, 8, codebook_dim]

        chunk_len_actual = chunk_end - chunk_start
        z_tgt_acc[chunk_start:chunk_end] += z_pred.squeeze(0)[:chunk_len_actual]
        z_tgt_count[chunk_start:chunk_end] += 1.0

    if (z_tgt_count == 0).any():
        n_missed = int((z_tgt_count == 0).sum())
        print(f"  WARN: {n_missed}/{T_token_full} tokens missed by the overlap average")
    z_tgt_avg = z_tgt_acc / z_tgt_count.clamp(min=1.0).unsqueeze(-1)

    # The latent stays continuous and is decoded directly, with no quantization, which is
    # how ACE treats it at training time too.
    decoded = registry.decode_tokens(tgt_skel, z_tgt_avg.unsqueeze(0))
    decoded_phys = registry.unnormalize(tgt_skel, decoded).squeeze(0)
    return decoded_phys.cpu().numpy()


def load_ace(ckpt_path, device, latent_cache=DEFAULT_LATENT_CACHE):
    """Load a trained ACE checkpoint plus the tokenizers every query may need."""
    print(f"Loading ckpt: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    skels = ckpt['skels']
    skel_to_id = ckpt['skel_to_id']
    print(f"Trained on {len(skels)} skeletons")

    G = ACEGenerator(n_skels=len(skels)).to(device)
    G.load_state_dict(ckpt['G'])
    G.eval()
    skel_enc = SkelGraphEncoder().to(device)
    skel_enc.load_state_dict(ckpt['skel_enc'])
    skel_enc.eval()

    # A query may target any of the 70 skeletons, so every tokenizer is loaded.
    print("Loading the tokenizers for all 70 skeletons...")
    registry = TokenizerRegistry(list(OBJECT_SUBSETS_DICT['all']), device=device)

    # The START tokens are built with the per-skeleton latent means, then the checkpoint's
    # own values (means included) replace them.
    cache = torch.load(latent_cache, map_location='cpu', weights_only=False)
    z_means = torch.zeros(len(skels), 8, 256)
    for s in skels:
        z_means[skel_to_id[s]] = cache[s]['z_continuous'].mean(dim=0)
    starts = ACEStartTokens(n_skels=len(skels), n_train_skels=len(skels),
                            codebook_dim=256, n_tokens=8, z_means=z_means).to(device)
    starts.load_state_dict(ckpt['starts'])
    starts.eval()

    cond_dict = np.load(COND_PATH, allow_pickle=True).item()
    skel_features = build_skel_features(cond_dict)
    max_J = max(skel_features[s]['n_joints'] for s in skels)
    return G, skel_enc, starts, registry, skel_features, max_J, skel_to_id


def run(queries, out_dir, G, skel_enc, starts, registry, skel_features, max_J, skel_to_id,
        device, zero_source=False, fixed_length=None):
    """Answer every query and write one query_XXXX.npy per query into out_dir."""
    t0 = time.time()
    n_done = n_skipped = 0
    for i, q in enumerate(queries):
        out_path = out_dir / f"query_{q['query_id']:04d}.npy"
        if out_path.exists():
            n_done += 1
            continue
        try:
            result = retarget_one_query(q, G, skel_enc, starts, registry, skel_features,
                                        max_J, skel_to_id, device,
                                        zero_source=zero_source, fixed_length=fixed_length)
            np.save(out_path, result.astype(np.float32))
            n_done += 1
        except Exception as e:
            print(f"  query {q['query_id']} ({q['skel_a']}->{q['skel_b']}): FAILED — {e}")
            import traceback; traceback.print_exc()
            n_skipped += 1
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(queries) - i - 1)
            print(f"  [{i+1}/{len(queries)}] done={n_done} skipped={n_skipped} "
                  f"({elapsed:.0f}s, ETA {eta/60:.0f}min)")

    print(f"\nFinished: {n_done}/{len(queries)} written, {n_skipped} skipped.")
    return n_done, n_skipped


def load_set(set_name, max_queries=None, fold=None):
    """The queries of a benchmark set, or of an action-level query fold when `fold` is given."""
    if fold is not None:
        path = QUERIES_ROOT / f'fold_{fold}' / 'manifest.json'
    else:
        path = SETS / f'truebones_{set_name}.json'
    with open(path) as f:
        manifest = json.load(f)
    queries = manifest['queries']
    if max_queries:
        queries = queries[:max_queries]
    return queries


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--ckpt', type=str, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--set', choices=['49', '1891'], default='1891',
                       help='benchmark set, read from benchmark/sets/truebones_<set>.json')
    group.add_argument('--fold', type=int, choices=[42, 43],
                       help='answer one of the two query folds in benchmark/queries instead')
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--max_queries', type=int, default=None)
    parser.add_argument('--latent_cache', type=str, default=str(DEFAULT_LATENT_CACHE),
                        help='the prepared tokenized clips, read for each skeleton\'s average')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    G, skel_enc, starts, registry, skel_features, max_J, skel_to_id = load_ace(
        args.ckpt, device, latent_cache=args.latent_cache)

    queries = load_set(args.set, args.max_queries, fold=args.fold)
    source = f"fold {args.fold}" if args.fold is not None else f"set {args.set}"
    print(f"Processing {len(queries)} queries of {source}...")
    run(queries, out_dir, G, skel_enc, starts, registry, skel_features, max_J, skel_to_id,
        device)


if __name__ == '__main__':
    main()
