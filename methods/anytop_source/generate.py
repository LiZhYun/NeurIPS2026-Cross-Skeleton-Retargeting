"""Run the source-conditioned AnyTop model over a Truebones benchmark set.

Every query names one source clip on skeleton A and one target skeleton B. The model encodes the
source clip into a latent z, then samples a motion for B conditioned on z, and the sample is
written as query_XXXX.npy so that benchmark/score.py can pick it up.

    python -m methods.anytop_source.generate --set 49 \\
        --ckpt <folder>/model000175000.pt --out_dir outputs/anytop/set49

With --fold 42 or --fold 43 in place of --set, it answers one of the two query folds of the
action-level test in benchmark/queries instead.
"""
from __future__ import annotations
import argparse
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from methods.common.queries import reference_length

ROOT = Path(__file__).resolve().parents[2]
SETS = ROOT / 'benchmark' / 'sets'
QUERIES_ROOT = ROOT / 'benchmark' / 'queries'


def load_args_from_checkpoint(model_path):
    args_path = os.path.join(os.path.dirname(model_path), 'args.json')
    with open(args_path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True,
                        help='trained checkpoint; args.json must sit next to it')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--set', choices=['49', '1891'], default='1891',
                       help='benchmark set, read from benchmark/sets/truebones_<set>.json')
    group.add_argument('--fold', type=int, choices=[42, 43],
                       help='answer one of the two query folds in benchmark/queries instead')
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--max_queries', type=int, default=10000)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--motion_dir', type=str, default='dataset/truebones/zoo/truebones_processed/motions')
    parser.add_argument('--splits', type=str, nargs='+', default=None,
                        help='answer only the queries of these splits (default: every query)')
    args = parser.parse_args()

    if args.fold is not None:
        manifest_path = QUERIES_ROOT / f'fold_{args.fold}' / 'manifest.json'
    else:
        manifest_path = SETS / f'truebones_{args.set}.json'
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Queries: {manifest_path}")
    print(f"Output dir: {out_dir}")

    # Imports (deferred to keep startup fast)
    from core.anytop.utils.fixseed import fixseed
    from core.anytop.utils import dist_util
    from core.anytop.utils.model_util import load_model
    from core.truebones.get_opt import get_opt
    from core.anytop.model.conditioners import T5Conditioner
    from methods.anytop_source.model_util import create_conditioned_model_and_diffusion
    from methods.anytop_source.condition import build_target_condition

    fixseed(args.seed)
    dist_util.setup_dist(args.device)

    ckpt_args = load_args_from_checkpoint(args.ckpt)

    class Namespace:
        def __init__(self, d):
            self.__dict__.update(d)
    ckpt = Namespace(ckpt_args)

    opt = get_opt(args.device)
    cond_dict = np.load(opt.cond_file, allow_pickle=True).item()

    print("Creating model and diffusion...")
    model, diffusion = create_conditioned_model_and_diffusion(ckpt)
    state_dict = torch.load(args.ckpt, map_location='cpu')
    load_model(model, state_dict)
    model.to(dist_util.dev())
    model.eval()

    print("Loading T5...")
    t5_conditioner = T5Conditioner(
        name=ckpt.t5_name, finetune=False, word_dropout=0.0,
        normalize_text=False, device=str(dist_util.dev()))

    with open(manifest_path) as f:
        manifest = json.load(f)
    queries = [q for q in manifest['queries']
               if args.splits is None or q['split'] in args.splits][:args.max_queries]
    print(f"{len(manifest['queries'])} queries; running {len(queries)} "
          f"(splits: {args.splits or 'all'})")

    # Pre-flight: verify all skeletons in cond_dict
    skel_set = {q['skel_a'] for q in queries} | {q['skel_b'] for q in queries}
    missing = skel_set - set(cond_dict.keys())
    assert not missing, f"cond_dict missing skeletons: {missing}"

    per_query = []
    t_total_0 = time.time()

    for i, q in enumerate(queries):
        qid = q['query_id']
        skel_a = q['skel_a']
        skel_b = q['skel_b']
        src_fname = q['src_fname']
        cluster = q['cluster']
        split = q['split']

        rec = {'query_id': qid, 'cluster': cluster, 'split': split,
               'skel_a': skel_a, 'skel_b': skel_b, 'status': 'pending'}

        try:
            # Load source motion
            src_path = os.path.join(args.motion_dir, src_fname)
            raw = np.load(src_path)  # [T_src, J_src, 13]
            T_src, J_src, _ = raw.shape
            assert J_src == len(cond_dict[skel_a]['parents']), \
                f"src joints {J_src} != cond {len(cond_dict[skel_a]['parents'])} for {skel_a}"

            # Output length: median length of the target clips the query lists
            T_tgt = reference_length(q, 120)

            # Use n_frames matching target (or src length if longer)
            n_frames = max(T_src, T_tgt)
            # Round up to multiple of temporal_window
            tw = ckpt.temporal_window
            n_frames = ((n_frames + tw - 1) // tw) * tw

            # Normalize source
            mean_a = cond_dict[skel_a]['mean']
            std_a = cond_dict[skel_a]['std'] + 1e-6
            norm = (raw - mean_a[None, :]) / std_a[None, :]
            norm = np.nan_to_num(norm)

            # Pad/crop to n_frames
            if T_src >= n_frames:
                norm = norm[:n_frames]
            else:
                pad = np.zeros((n_frames - T_src, J_src, 13))
                norm = np.concatenate([norm, pad], axis=0)

            # Pad to max_joints
            max_joints = opt.max_joints
            offsets_a = cond_dict[skel_a]['offsets']
            source_motion = np.zeros((n_frames, max_joints, 13))
            source_motion[:, :J_src, :] = norm
            source_offsets = np.zeros((max_joints, 3))
            source_offsets[:J_src, :] = offsets_a

            source_motion_t = torch.tensor(source_motion).permute(1, 2, 0).float().unsqueeze(0).to(dist_util.dev())
            source_offsets_t = torch.tensor(source_offsets).float().unsqueeze(0).to(dist_util.dev())
            source_mask = torch.zeros(1, max_joints, dtype=torch.bool).to(dist_util.dev())
            source_mask[0, :J_src] = True

            # Encode source
            with torch.no_grad():
                z, _, _ = model.encoder(source_motion_t, source_offsets_t, source_mask)

            # Build target condition
            _, model_kwargs = build_target_condition(
                skel_b, cond_dict, n_frames, ckpt.temporal_window,
                t5_conditioner, opt.max_joints, opt.feature_len)
            model_kwargs['y']['z'] = z

            # Sample
            with torch.no_grad():
                sample = diffusion.p_sample_loop(
                    model,
                    (1, opt.max_joints, opt.feature_len, n_frames),
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                    skip_timesteps=0, init_image=None,
                    progress=False, dump_steps=None, noise=None, const_noise=False,
                )

            # Denormalize and crop to target's actual joint count + length
            tgt = cond_dict[skel_b]
            n_joints_b = len(tgt['parents'])
            mean_b = tgt['mean']
            std_b = tgt['std']
            sample = sample[0].permute(2, 0, 1).cpu().numpy()  # [T, J_max, 13]
            sample = sample[:T_tgt, :n_joints_b, :]
            sample = sample * std_b[None, :] + mean_b[None, :]  # match dataset.denormalize()

            np.save(out_dir / f'query_{qid:04d}.npy', sample.astype(np.float32))
            rec['status'] = 'ok'
            rec['T_out'] = int(sample.shape[0])
            rec['J_out'] = int(sample.shape[1])
            rec['T_src'] = int(T_src)
            rec['T_tgt'] = int(T_tgt)
            rec['n_frames_used'] = int(n_frames)
            rec['src_padded_frames'] = max(0, n_frames - T_src)

            if (i + 1) % 10 == 0 or i == 0:
                elapsed = time.time() - t_total_0
                eta = elapsed / (i + 1) * (len(queries) - i - 1)
                print(f"  [{i+1}/{len(queries)}] {cluster}/{split}: {skel_a}→{skel_b} "
                      f"T={rec['T_out']} J={rec['J_out']} (elapsed {elapsed:.0f}s, ETA {eta:.0f}s)")

        except Exception as e:
            rec['status'] = 'failed'
            rec['error'] = str(e) + '\n' + traceback.format_exc(limit=3)
            print(f"  FAILED query {qid}: {e}")

        per_query.append(rec)

    total_time = time.time() - t_total_0
    print(f"\nTotal: {total_time:.0f}s, "
          f"{sum(1 for r in per_query if r['status'] == 'ok')}/{len(per_query)} OK")

    summary = {
        'method': 'AnyTop',
        'ckpt': args.ckpt,
        'queries': str(manifest_path.relative_to(ROOT)),
        'n_queries': len(per_query),
        'n_ok': sum(1 for r in per_query if r['status'] == 'ok'),
        'total_time_sec': total_time,
        'per_query': per_query,
    }
    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved: {out_dir}/metrics.json")


if __name__ == '__main__':
    main()
