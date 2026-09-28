"""Produce DPG-SB-v3's answer to every query in a benchmark set.

A query names one clip performed by one animal and asks for the same motion on another
animal. The answer is written as query_XXXX.npy, one file per query, which is what the
scoring scripts read.

This model does not start from nothing. For each query it looks for a clip the target animal
has already performed with the query's action, adds noise to it, and moves it towards what
the source motion asks for. Every clip the query will later be scored against is excluded
from that search, so the answer cannot be a copy of something the scoring is about to
compare it with. If no other clip of that action exists for the target animal, any other
clip of that animal is used, and failing that, noise.

The source clip's codes are taken from the prepared cache when it holds them, and encoded on
the spot when it does not. The model's shape is read from args.json beside the checkpoint.

The noise comes from PyTorch's own generator, which the run reported in the paper left
unseeded, so two runs differ a little. Pass --seed to make a run repeatable; leaving it out
keeps the behaviour the paper had.

    python -m methods.dpg_sb.generate --set 49 \\
        --ckpt save/dpg_sb/dpg_sb/final.pt --out_dir outputs/dpg_sb/set49

With --fold 42 or --fold 43 in place of --set, it answers one of the two query folds of the
action-level test in benchmark/queries instead.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

from core.truebones.param_utils import OBJECT_SUBSETS_DICT, DATASET_DIR
from benchmark.action_taxonomy import (
    parse_action_from_filename, action_to_cluster, ACTION_CLUSTERS,
)
from methods.common.queries import REFERENCE_KEYS
from methods.dpg_sb.model import BridgeGenerator
from methods.tokenizer.registry import TokenizerRegistry

ROOT = Path(__file__).resolve().parents[2]
SETS = ROOT / 'benchmark' / 'sets'
QUERIES_ROOT = ROOT / 'benchmark' / 'queries'
DATA_ROOT = ROOT / DATASET_DIR
MOTION_DIR = DATA_ROOT / 'motions'
DEFAULT_LATENT_CACHE = ROOT / 'save/latents/cache_all.pt'

CLUSTERS = sorted(ACTION_CLUSTERS.keys())


def load_model(ckpt_path: str, device):
    """Build the generator from args.json next to the checkpoint, then load its weights."""
    ckpt_dir = Path(ckpt_path).parent
    args_dict = json.load(open(ckpt_dir / 'args.json'))
    G = BridgeGenerator(
        codebook_dim=256, n_tokens=8,
        d_model=args_dict['d_model'],
        n_layers=args_dict['n_layers'],
        n_heads=args_dict['n_heads'],
        n_skels=args_dict['n_skels'],
        n_exact_actions=args_dict['n_exact_actions'],
        src_layers=args_dict['src_layers'],
        dropout=args_dict['dropout'],
    ).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    G.load_state_dict(state['G'])
    G.eval()
    return G, args_dict


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
    parser.add_argument('--n_steps', type=int, default=20,
                        help='how many steps to take from the starting clip to the answer')
    parser.add_argument('--noise_scale', type=float, default=0.3)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max_queries', type=int, default=10000)
    parser.add_argument('--latent_cache', type=str, default=str(DEFAULT_LATENT_CACHE),
                        help='the prepared tokenized clips, covering all 70 skeletons')
    parser.add_argument('--seed', type=int, default=None,
                        help='make the run repeatable; leaving it out keeps the behaviour '
                             'the reported run had')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    print(f"Loading model: {args.ckpt}")
    G, args_dict = load_model(args.ckpt, device)
    skels_train = args_dict['skels']
    skel_to_id = {s: i for i, s in enumerate(skels_train)}
    exact_actions = args_dict['exact_actions']
    exact_to_idx = {a: i for i, a in enumerate(exact_actions)}
    print(f"Trained skeletons: {len(skels_train)}, exact actions: {len(exact_actions)}")

    print(f"Loading the latent cache...")
    cache = torch.load(args.latent_cache, map_location='cpu', weights_only=False)
    all_skels = sorted(s for s in cache.keys() if not s.startswith('_'))
    print(f"  {len(all_skels)} skeletons in the cache")

    # Per-skeleton normalization, recomputed from this cache so the held-out skeletons have
    # statistics too
    z_stats = {}
    z_per_skel = {}
    for s in all_skels:
        z_raw = cache[s]['z_continuous'].float()
        mu = z_raw.mean(dim=0, keepdim=True)
        sigma = z_raw.std(dim=0, keepdim=True).clamp_min(1e-3)
        z_per_skel[s] = ((z_raw - mu) / sigma).to(device)
        z_stats[s] = (mu.to(device), sigma.to(device))

    # (skeleton, clip) -> row, and (skeleton, action) -> [(clip, row)]
    fname_to_ri = {}
    skel_action_clips = defaultdict(list)
    for s in all_skels:
        meta = cache[s]['meta']
        for ri, (fname, _) in enumerate(meta):
            fname_to_ri[(s, fname)] = ri
            action = parse_action_from_filename(fname)
            if action_to_cluster(action) is not None:
                skel_action_clips[(s, action)].append((fname, ri))

    # Tokenizers, including the held-out skeletons, since queries may target them
    print(f"Loading the tokenizers...")
    decode_skels = sorted(set(skels_train) | set(OBJECT_SUBSETS_DICT['held_out']))
    reg = TokenizerRegistry(decode_skels, device=str(device))
    n_skels = reg.n_skels if isinstance(reg.n_skels, int) else reg.n_skels()
    print(f"  loaded {n_skels} tokenizers")

    rng = np.random.RandomState(42)

    if args.fold is not None:
        manifest_path = QUERIES_ROOT / f'fold_{args.fold}' / 'manifest.json'
    else:
        manifest_path = SETS / f'truebones_{args.set}.json'
    with open(manifest_path) as f:
        manifest = json.load(f)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_done = n_failed = 0
    per_query = []
    t0 = time.time()
    queries = manifest['queries'][:args.max_queries]
    for i, q in enumerate(queries):
        qid = q['query_id']
        src_skel = q['skel_a']; tgt_skel = q['skel_b']
        src_fname = q['src_fname']
        src_action = q['src_action']

        try:
            # 1. The source latent, normalized; encoded from disk if the cache lacks it
            if (src_skel, src_fname) in fname_to_ri:
                src_ri = fname_to_ri[(src_skel, src_fname)]
                z_a = z_per_skel[src_skel][src_ri].unsqueeze(0).to(device)  # [1, 8, 256]
            else:
                src_motion = np.load(MOTION_DIR / src_fname).astype(np.float32)
                # Crop or zero-pad to the 32-frame window the tokenizer expects
                if src_motion.shape[0] >= 32:
                    win = src_motion[:32]
                else:
                    pad = np.zeros((32 - src_motion.shape[0], src_motion.shape[1], 13),
                                   dtype=np.float32)
                    win = np.concatenate([src_motion, pad], axis=0)
                win_t = torch.from_numpy(win).to(device).unsqueeze(0)  # [1, T, J, 13]
                win_norm = reg.normalize(src_skel, win_t)
                z_raw, _ = reg.encode_window(src_skel, win_norm)         # [1, 8, 256]
                mu_a, sigma_a = z_stats[src_skel]
                z_a = ((z_raw - mu_a) / sigma_a).to(device)

            # 2. Starting point: a clip on the target skeleton with the query's action.
            # Every clip this query's scoring will look at is excluded, so nothing that
            # would be compared against can be copied straight through.
            forbidden = set()
            for key in REFERENCE_KEYS:
                for x in q.get(key, []):
                    forbidden.add(x['fname'])

            init_candidates = skel_action_clips.get((tgt_skel, src_action), [])
            init_candidates = [(f, r) for f, r in init_candidates if f not in forbidden]
            used_noise_init = False
            if not init_candidates:
                # Fall back to any other clip on the target skeleton
                all_tgt = []
                for k in skel_action_clips:
                    if k[0] == tgt_skel:
                        for f, r in skel_action_clips[k]:
                            if f not in forbidden:
                                all_tgt.append((f, r))
                if all_tgt:
                    init_candidates = all_tgt
                else:
                    used_noise_init = True

            if used_noise_init:
                init_fname = '__noise__'
                z_init = torch.randn(1, 8, 256, device=device)
            else:
                init_fname, init_ri = init_candidates[rng.randint(len(init_candidates))]
                z_init = z_per_skel[tgt_skel][init_ri].unsqueeze(0).to(device)  # [1, 8, 256]

            # 3. Add noise
            noise = torch.randn_like(z_init) * args.noise_scale
            z_start = z_init + noise

            # 4. Conditioning ids; 0 stands in for a skeleton the model never trained on
            sa_id = torch.tensor([skel_to_id.get(src_skel, 0)], device=device, dtype=torch.long)
            sb_id = torch.tensor([skel_to_id.get(tgt_skel, 0)], device=device, dtype=torch.long)
            aid = torch.tensor([exact_to_idx.get(src_action, 0)], device=device, dtype=torch.long)

            # 5. Integrate the bridge
            with torch.no_grad():
                src_tokens = G.encode_source(z_a)
                z_t = z_start.clone()
                dt = 1.0 / args.n_steps
                for k in range(args.n_steps):
                    t_now = (k + 0.5) / args.n_steps
                    t_b = torch.tensor([t_now], device=device)
                    v = G(z_t, t_b, src_tokens, aid, sa_id, sb_id)
                    z_t = z_t + dt * v

            # 6. Undo the target's normalization
            mu_b, sigma_b = z_stats[tgt_skel]
            z_b_pred = z_t * sigma_b + mu_b  # [1, 8, 256]

            # 7. Decode with the target skeleton's tokenizer
            with torch.no_grad():
                motion_norm = reg.decode_tokens(tgt_skel, z_b_pred)  # [1, T, J, 13]
                motion = reg.unnormalize(tgt_skel, motion_norm)      # physical units

            motion_np = motion[0].cpu().numpy().astype(np.float32)
            np.save(out_dir / f'query_{qid:04d}.npy', motion_np)
            per_query.append({
                'query_id': qid, 'status': 'ok',
                'src_skel': src_skel, 'tgt_skel': tgt_skel,
                'src_action': src_action,
                'init_fname': init_fname,
                'output_T': int(motion_np.shape[0]),
                'output_J': int(motion_np.shape[1]),
            })
            n_done += 1
        except Exception as e:
            import traceback
            tb = traceback.format_exc(limit=2)
            print(f"  query {qid} FAILED: {e}\n{tb}")
            per_query.append({'query_id': qid, 'status': 'failed', 'error': str(e)})
            n_failed += 1

        if (i + 1) % 25 == 0 or i == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(queries) - i - 1)
            print(f"  [{i+1}/{len(queries)}] "
                  f"elapsed {elapsed:.0f}s, ETA {eta:.0f}s, ok={n_done}")

    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump({'method': 'DPG-SB-v3', 'queries': str(manifest_path.relative_to(ROOT)),
                   'ckpt': args.ckpt,
                   'n_steps': args.n_steps, 'noise_scale': args.noise_scale,
                   'n_done': n_done, 'n_failed': n_failed,
                   'per_query': per_query}, f, indent=2)
    print(f"\nFinished: {n_done} ok, {n_failed} failed.")


if __name__ == '__main__':
    main()
