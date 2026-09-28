"""Prepare the tokenized clips that four of the methods train on.

Training a model on top of the per-skeleton tokenizers means encoding the same clips over
and over. This does it once. For every skeleton it walks that skeleton's training clips,
takes a 32-frame window every 4 frames, encodes each window with that skeleton's frozen
tokenizer, and stores the result along with a few measurements of the window and a note of
where it came from. ACE, MoReFlow, AL-Flow and DPG-SB all read the file it writes.

What is stored per skeleton:

  z_continuous    the encoded window, 8 tokens wide
  z_indices       which codebook entry each token is
  phi_<name>      each of the five measurements taken on the window
  meta            the clip each window came from, and its first frame
  prev_row_idx    the row holding the window 16 frames earlier in the same clip, or -1
                  where the window starts a clip
  is_clip_start   true exactly where prev_row_idx is -1

Each tokenizer set aside a tenth of its skeleton's clips to check its own progress against.
Those clips are left out here too, so nothing a tokenizer was chosen on reaches the models
trained on top of it.

The previous-window note is what lets ACE learn to continue a motion: the chunk before a
window starting at frame t begins at frame t minus 16, which at this spacing is four rows
earlier within the clip. It is stored as a row number rather than a second copy.

Output: save/latents/cache_<scope>.pt, one file per scope.

Usage:
  python -m methods.moreflow.build_cache --scope train   # the 60 training skeletons
  python -m methods.moreflow.build_cache --scope all     # all 70
"""
from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from core.truebones.param_utils import OBJECT_SUBSETS_DICT, DATASET_DIR
from methods.tokenizer.registry import TokenizerRegistry, CKPT_ROOT
from methods.common.motion_descriptors import (
    CONDITIONS, D_COND_PADDED,
    phi as phi_np, precompute_skel_descriptors, load_contact_groups,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / DATASET_DIR
MOTION_DIR = DATA_ROOT / 'motions'
COND_PATH = DATA_ROOT / 'cond.npy'
SAVE_ROOT = PROJECT_ROOT / 'save/latents'
WINDOW = 32
STRIDE = 4


def list_skel_motions(skel_name):
    return sorted([f for f in MOTION_DIR.iterdir()
                   if (f.name.startswith(skel_name + '___') or
                       f.name.startswith(skel_name + '_'))
                   and f.suffix == '.npy'])


def get_train_clip_split(skel_name, n_all, val_frac, seed=42):
    """Work out which clips the tokenizer trained on, so the rest can be left out.

    Mirrors train_one_skel in methods/tokenizer/train.py. Returns the train clip indices.
    """
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_all)
    if n_all >= 3 and val_frac > 0:
        n_val = max(1, int(round(n_all * val_frac)))
        n_val = min(n_val, n_all - 2)
        val_idx = set(perm[:n_val].tolist())
        return [i for i in range(n_all) if i not in val_idx]
    else:
        return list(range(n_all))


def extract_one_skel(skel_name, registry, skel_descs, cond_dict, args):
    """Extract every training window for one skeleton.

    Returns dict: {z_continuous, z_indices, phi_<family>, meta}
    """
    # The tokenizer's args.json records the validation fraction and seed it actually used
    args_json_path = CKPT_ROOT / skel_name / 'args.json'
    with open(args_json_path) as f:
        a = json.load(f)
    val_frac = float(a.get('val_frac', 0.10))
    seed = int(a.get('seed', 42))

    all_clips = list_skel_motions(skel_name)
    if not all_clips:
        return None
    valid_clips = []
    for c in all_clips:
        m = np.load(c)
        if m.shape[0] >= WINDOW:
            valid_clips.append(c)
    if not valid_clips:
        return None
    n_all = len(valid_clips)

    # Training clips only, matching the tokenizer's split
    train_indices = get_train_clip_split(skel_name, n_all, val_frac, seed=seed)
    train_clips = [valid_clips[i] for i in train_indices]

    # Normalize with the tokenizer's own mean and std
    mean = registry.get(skel_name)['mean']
    std = registry.get(skel_name)['std']
    desc = skel_descs[skel_name]

    all_z = []
    all_idx = []
    all_phi = {c: [] for c in CONDITIONS}
    all_meta = []

    for clip_path in train_clips:
        motion = np.load(clip_path).astype(np.float32)
        T = motion.shape[0]
        n_windows = (T - WINDOW) // STRIDE + 1
        for i in range(n_windows):
            t_start = i * STRIDE
            window = motion[t_start:t_start + WINDOW]                          # [WINDOW, J, 13]
            window_t = torch.from_numpy(window).to(registry.device).float()
            window_norm = (window_t - mean) / std
            window_norm = torch.nan_to_num(window_norm)
            with torch.no_grad():
                z, idx = registry.encode_window(skel_name, window_norm)
            all_z.append(z.squeeze(0).cpu())                                    # [8, codebook_dim]
            all_idx.append(idx.squeeze(0).cpu())                                # [8]
            # Condition descriptors, computed on the window in physical units
            for c in CONDITIONS:
                all_phi[c].append(torch.from_numpy(phi_np(window, c, desc)))    # [24]
            all_meta.append((clip_path.name, int(t_start)))

    if not all_z:
        return None
    out = {
        'z_continuous': torch.stack(all_z),                                     # [N, 8, codebook_dim]
        'z_indices': torch.stack(all_idx),                                      # [N, 8]
        'meta': all_meta,
    }
    for c in CONDITIONS:
        out[f'phi_{c}'] = torch.stack(all_phi[c])                                # [N, 24]
    return out


def add_previous_window_index(skel_data):
    """Note, for each window, which row holds the window just before it.

    meta lists (clip filename, first frame) per row, in extraction order. Within one clip,
    the row 16 frames earlier is the previous chunk; where there is none, the window is a
    clip start. Returns (number of clip starts, number of rows).
    """
    meta = skel_data['meta']                                              # (clip filename, t_start)
    N = len(meta)
    prev_row_idx = -torch.ones(N, dtype=torch.long)
    is_clip_start = torch.ones(N, dtype=torch.bool)                       # default True

    clip_to_rows = {}
    for i, (fname, t_start) in enumerate(meta):
        clip_to_rows.setdefault(fname, []).append((i, t_start))

    for fname, rows in clip_to_rows.items():
        rows_sorted = sorted(rows, key=lambda x: x[1])
        t_to_idx = {t: i for i, t in rows_sorted}
        for i, t_start in rows_sorted:
            prev_t = t_start - 16
            if prev_t < 0 or prev_t not in t_to_idx:
                continue
            prev_row_idx[i] = t_to_idx[prev_t]
            is_clip_start[i] = False

    skel_data['prev_row_idx'] = prev_row_idx
    skel_data['is_clip_start'] = is_clip_start
    return int(is_clip_start.sum().item()), N


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--scope', type=str, default='train',
                        choices=['train', 'held_out', 'all'],
                        help='train = the 60 training skeletons; held_out = the 10 held out; all = 70')
    parser.add_argument('--out', type=str, default=None,
                        help='write to this filename instead of cache_<scope>.pt')
    parser.add_argument('--skip_existing', action='store_true')
    args = parser.parse_args()

    SAVE_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = SAVE_ROOT / (args.out or f'cache_{args.scope}.pt')
    if args.skip_existing and out_path.exists():
        print(f"Output {out_path} exists; skipping (--skip_existing).")
        return

    if args.scope == 'train':
        skels = list(OBJECT_SUBSETS_DICT['train'])
    elif args.scope == 'held_out':
        skels = list(OBJECT_SUBSETS_DICT['held_out'])
    elif args.scope == 'all':
        skels = list(OBJECT_SUBSETS_DICT['train']) + list(OBJECT_SUBSETS_DICT['held_out'])
    else:
        raise ValueError(args.scope)
    print(f"Scope: {args.scope} ({len(skels)} skeletons)")

    print("Loading cond, contact groups and the tokenizer registry...")
    cond_dict = np.load(COND_PATH, allow_pickle=True).item()
    contact_groups = load_contact_groups()
    skel_descs = precompute_skel_descriptors(cond_dict, contact_groups)
    registry = TokenizerRegistry(skels, device='cuda' if torch.cuda.is_available() else 'cpu')

    cache = {}
    t0 = time.time()
    for i, skel in enumerate(skels):
        if skel not in registry.tokenizers:
            print(f"[{i+1}/{len(skels)}] {skel}: SKIP (no tokenizer)")
            continue
        if skel_descs[skel]['n_ee'] == 0:
            print(f"[{i+1}/{len(skels)}] {skel}: WARN no end effectors "
                  f"(its condition descriptors will be partly zero)")
        try:
            data = extract_one_skel(skel, registry, skel_descs, cond_dict, args)
            if data is None:
                print(f"[{i+1}/{len(skels)}] {skel}: SKIP (no training windows)")
                continue
            n_start, n_rows = add_previous_window_index(data)
            cache[skel] = data
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(skels) - i - 1)
            print(f"[{i+1}/{len(skels)}] {skel}: {n_rows} windows, "
                  f"{n_start} clip starts ({100*n_start/n_rows:.1f}%) "
                  f"(elapsed {elapsed:.0f}s, ETA {eta:.0f}s)")
        except Exception as e:
            print(f"[{i+1}/{len(skels)}] {skel}: FAILED ({type(e).__name__}: {e})")
            import traceback
            traceback.print_exc()

    cache['_meta'] = {
        'scope': args.scope,
        'window': WINDOW,
        'stride': STRIDE,
        'n_skels': len(cache),
        'skel_list': list(cache.keys()),
    }
    print(f"\nSaving cache to {out_path}")
    tmp_path = out_path.with_suffix('.pt.tmp')
    torch.save(cache, tmp_path)
    os.replace(tmp_path, out_path)
    total_windows = sum(v['z_continuous'].shape[0] for k, v in cache.items() if k != '_meta')
    print(f"Total windows across {cache['_meta']['n_skels']} skeletons: {total_windows}")
    print(f"Cache file size: {out_path.stat().st_size / (1024**3):.2f} GB")


if __name__ == '__main__':
    main()
