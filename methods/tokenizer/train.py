"""Train the per-skeleton VQ-VAE motion tokenizers shared by ACE, MoReFlow, AL-Flow and DPG-SB.

One tokenizer is trained per skeleton, on that skeleton's own motion clips, from 32-frame
windows. The settings follow MoReFlow's quadruped configuration, which fits Truebones better
than their humanoid one: hidden=512, codebook_dim=256, K=256, n_resblocks=3, ReLU,
smooth-L1 reconstruction, AdamW(lr=2e-4, wd=0), cosine schedule after a 1k-step warmup,
100k steps, and a 10% clip-level validation split that selects ckpt_best.pt.

The paper trained tokenizers for all 70 skeletons; methods/tokenizer/config.json records that run.

Usage:
  python -m methods.tokenizer.train --skel Horse                    # one skeleton
  python -m methods.tokenizer.train --all_train_skeletons           # the 60 training skeletons
  python -m methods.tokenizer.train --config methods/tokenizer/config.json   # the paper's run
"""
from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from core.truebones.param_utils import OBJECT_SUBSETS_DICT, DATASET_DIR
from methods.tokenizer.vqvae import MoReFlowVQVAE, count_parameters
from methods.common.training import apply_config, lr_lambda, set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / DATASET_DIR
MOTION_DIR = DATA_ROOT / 'motions'
COND_PATH = DATA_ROOT / 'cond.npy'
SAVE_ROOT = PROJECT_ROOT / 'save/tokenizers'
WINDOW = 32  # about 1 s at 30 fps, as in MoReFlow §3.1; with n_downsample=2 → 8 tokens


def list_skel_motions(skel_name):
    """All motions for one skeleton."""
    return sorted([f for f in os.listdir(MOTION_DIR)
                   if (f.startswith(skel_name + '___') or f.startswith(skel_name + '_'))
                   and f.endswith('.npy')])


def load_skel_clips(skel_name, cond):
    """Load all clips for skel, normalize via cond, return list of [T, J, 13] tensors."""
    clips = []
    mean = cond[skel_name]['mean']  # [J, 13]
    std = cond[skel_name]['std'] + 1e-6
    for fname in list_skel_motions(skel_name):
        m = np.load(MOTION_DIR / fname).astype(np.float32)
        if m.shape[0] < WINDOW:
            continue
        norm = (m - mean[None, :]) / std[None, :]
        norm = np.nan_to_num(norm)
        clips.append(norm)
    return clips


def sample_window(clips, rng, clip_weights):
    """Window-uniform sample: P(clip) ∝ (T_clip - WINDOW + 1).

    Equivalent to drawing uniformly over all valid (clip, start) pairs across
    the dataset, which avoids the short-clip bias of uniform-over-clips.
    """
    clip_idx = rng.choice(len(clips), p=clip_weights)
    clip = clips[clip_idx]
    n_starts = clip.shape[0] - WINDOW + 1  # ≥ 1 by construction
    start = 0 if n_starts == 1 else rng.randint(0, n_starts)
    return clip[start:start + WINDOW]


def adaptive_codebook_size(n_total_windows, requested_K):
    """Cap K so we have ≥ ~10 unique training windows per codebook entry on average.

    Prevents a 512-entry codebook from being trained on 76 windows (Roach).
    Also clamps to nearest power-of-two ≥ 64 for VQ stability.
    """
    target = max(64, n_total_windows // 10)
    K = min(requested_K, 1 << max(0, (target - 1).bit_length()))
    return max(64, K)


def train_one_skel(skel_name, args, cond):
    save_dir = SAVE_ROOT / skel_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Seed BEFORE model construction so conv-init RNG is deterministic
    set_seed(args.seed)
    rng = np.random.RandomState(args.seed)

    all_clips = load_skel_clips(skel_name, cond)
    if not all_clips:
        print(f"  [{skel_name}] NO clips with ≥{WINDOW} frames — skipping")
        with open(save_dir / 'SKIPPED.txt', 'w') as f:
            f.write(f"No clips ≥{WINDOW} frames")
        return

    # Clip-level train/val split (10% by default; ≥1 val clip when feasible)
    n_all = len(all_clips)
    perm = rng.permutation(n_all)
    if n_all >= 3 and args.val_frac > 0:
        n_val = max(1, int(round(n_all * args.val_frac)))
        # Cap val so train always has at least 2 clips
        n_val = min(n_val, n_all - 2)
        val_idx = set(perm[:n_val].tolist())
        clips = [all_clips[i] for i in range(n_all) if i not in val_idx]
        val_clips = [all_clips[i] for i in sorted(val_idx)]
    else:
        clips = all_clips
        val_clips = []
        if args.val_frac > 0:
            print(f"  [{skel_name}] only {n_all} clips — disabling validation split")

    # Window-uniform sampling weights (train clips)
    n_starts = np.array([c.shape[0] - WINDOW + 1 for c in clips], dtype=np.float64)
    n_total_windows = int(n_starts.sum())
    clip_weights = n_starts / n_total_windows

    if val_clips:
        val_starts = np.array([c.shape[0] - WINDOW + 1 for c in val_clips], dtype=np.float64)
        n_val_windows = int(val_starts.sum())
        val_weights = val_starts / val_starts.sum()
    else:
        n_val_windows = 0
        val_weights = None

    n_joints = clips[0].shape[1]
    feat_dim = n_joints * 13

    # Adaptive K computed against train windows only (no val leakage)
    K_eff = adaptive_codebook_size(n_total_windows, args.codebook_size)

    print(f"\n[{skel_name}] {n_all} clips → {len(clips)} train ({n_total_windows} windows) / "
          f"{len(val_clips)} val ({n_val_windows} windows), J={n_joints}, "
          f"feat_dim={feat_dim}, K={K_eff} (req {args.codebook_size})")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MoReFlowVQVAE(
        input_dim=feat_dim,
        hidden=args.hidden,
        codebook_size=K_eff,
        codebook_dim=args.codebook_dim,
        n_resblocks=args.n_resblocks,
        n_downsample=args.n_downsample,
        activation=args.activation,
        dead_code_threshold=args.dead_code_threshold,
    ).to(device)
    # A 32-frame window must give 8 tokens at n_downsample=2
    assert WINDOW % model.downsample_factor == 0, (
        f"WINDOW={WINDOW} not divisible by downsample_factor={model.downsample_factor}")
    n_tokens = WINDOW // model.downsample_factor
    assert n_tokens == 8, (
        f"Expected 8 tokens per window (MoReFlow §3.1), got {n_tokens}. "
        f"Set --n_downsample 2 for WINDOW=32.")
    n_params = count_parameters(model)
    print(f"  Params: {n_params:,}, tokens/window: {n_tokens}")

    # MoReFlow Appendix A.1: weight_decay=0.0, smooth L1 reconstruction loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                   betas=(0.9, 0.99))

    # MoReFlow Appendix A.1: linear warmup → cosine decay to 0 over remaining steps.
    warmup = max(0, args.warmup_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step_idx: lr_lambda(step_idx, warmup, args.max_steps))

    log_path = save_dir / 'training_log.jsonl'

    # Codebook use: the distinct codes seen across the last 100 minibatches
    from collections import deque
    code_history = deque(maxlen=100)

    # Save args + effective K + downsample_factor + arch metadata
    with open(save_dir / 'args.json', 'w') as f:
        json.dump({**vars(args), 'skel': skel_name,
                   'n_clips_all': n_all, 'n_clips_train': len(clips),
                   'n_clips_val': len(val_clips),
                   'n_total_windows': n_total_windows,
                   'n_val_windows': n_val_windows,
                   'n_joints': n_joints, 'feat_dim': feat_dim,
                   'codebook_size_effective': K_eff,
                   'downsample_factor': model.downsample_factor,
                   'n_tokens_per_window': n_tokens,
                   'window': WINDOW,
                   'n_params': n_params}, f, indent=2)

    @torch.no_grad()
    def evaluate(at_step):
        """Validation pass with a fixed RNG seeded by (skel_seed, at_step) so val sampling
        is reproducible AND independent of the train RNG stream."""
        if not val_clips:
            return None
        val_rng = np.random.RandomState(args.seed * 1_000_003 + at_step)
        model.eval()
        losses = []
        for _ in range(args.val_batches):
            vb = np.stack([sample_window(val_clips, val_rng, val_weights)
                           for _ in range(args.batch_size)])
            vb = vb.reshape(args.batch_size, WINDOW, -1)
            vb_t = torch.from_numpy(vb).float().to(device)
            recon, _, _ = model(vb_t)
            losses.append(F.smooth_l1_loss(recon, vb_t).item())
        model.train()
        return float(np.mean(losses))

    model.train()
    t0 = time.time()
    start_step = 0

    # main() skips a skeleton that already has ckpt_final.pt unless --no_resume is given, in
    # which case training starts over here and overwrites it.

    best_val = float('inf')

    # Anchor best_val from existing ckpt_best.pt FIRST (independent of periodic ckpt cadence).
    # Without this, a resumed run can overwrite a better existing best ckpt with a worse one
    # if the previous best was set after the last periodic checkpoint and before the crash.
    best_path = save_dir / 'ckpt_best.pt'
    if best_path.exists():
        try:
            prev_best = torch.load(best_path, map_location='cpu')
            if 'val_recon' in prev_best and isinstance(prev_best['val_recon'], (int, float)):
                best_val = float(prev_best['val_recon'])
                print(f"  Anchored best_val={best_val:.4f} from existing {best_path.name}")
            del prev_best
        except Exception as e:
            print(f"  WARN: could not read existing {best_path.name} ({e}); ignoring")

    # Resume from latest periodic checkpoint if present (defensive: if the latest is corrupt
    # from a torn write, fall back to the second-most-recent).
    ckpts = sorted(save_dir.glob('ckpt_step*.pt'))
    if ckpts and not args.no_resume:
        sd = None
        for cand in reversed(ckpts):
            try:
                sd = torch.load(cand, map_location=device)
                latest = cand
                break
            except Exception as e:
                print(f"  WARN: {cand.name} unreadable ({type(e).__name__}: {e}); trying older")
                cand.unlink(missing_ok=True)
        if sd is None:
            print(f"  No usable periodic ckpt — starting fresh")
        else:
            print(f"  Resuming from {latest.name}")
            # The saved codebook must have the size this run would choose for the same clips
            saved_K = sd['model_state_dict']['vq._codebook.embed'].shape[1]
            if saved_K != K_eff:
                raise RuntimeError(
                    f"K_eff drift on resume for {skel_name}: ckpt has K={saved_K}, "
                    f"current adaptive K={K_eff}. Did the clip set change? "
                    f"Pass --no_resume to retrain from scratch.")
            model.load_state_dict(sd['model_state_dict'])
            optimizer.load_state_dict(sd['optimizer_state_dict'])
            if 'scheduler_state_dict' in sd:
                scheduler.load_state_dict(sd['scheduler_state_dict'])
            start_step = sd['step']
            # Merge: keep the better of the periodic ckpt's best_val and the ckpt_best anchor
            periodic_best = sd.get('best_val', float('inf'))
            best_val = min(best_val, periodic_best)
            rng.set_state(sd['numpy_rng_state'])
            # RNG states must live on CPU. torch.load(map_location=device) moves all tensors
            # to `device`, so we explicitly bring them back to CPU before set_rng_state.
            torch.set_rng_state(sd['torch_rng_state'].to('cpu', dtype=torch.uint8))
            if torch.cuda.is_available() and sd.get('cuda_rng_state') is not None:
                cuda_state = sd['cuda_rng_state']
                if isinstance(cuda_state, list):
                    cuda_state = [s.to('cpu', dtype=torch.uint8) for s in cuda_state]
                torch.cuda.set_rng_state_all(cuda_state)

    for step in range(start_step, args.max_steps):
        # Build minibatch
        batch_x = np.stack([sample_window(clips, rng, clip_weights)
                            for _ in range(args.batch_size)])
        batch_x = batch_x.reshape(args.batch_size, WINDOW, -1)  # [B, T, J*C]
        batch_t = torch.from_numpy(batch_x).float().to(device)

        recon, vq_loss, indices = model(batch_t)
        # Smooth L1, as in MoReFlow Appendix A.1 (more robust than MSE on motion)
        recon_loss = F.smooth_l1_loss(recon, batch_t)
        loss = recon_loss + vq_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Track codes used in this minibatch (cumulative window via deque)
        code_history.append(set(indices.cpu().numpy().flatten().tolist()))

        if (step + 1) % args.log_interval == 0:
            elapsed = time.time() - t0
            eta = elapsed / (step + 1 - start_step) * (args.max_steps - step - 1)
            cumulative_codes = set().union(*code_history)
            n_cum = len(cumulative_codes)
            n_batch = len(code_history[-1])
            cur_lr = optimizer.param_groups[0]['lr']
            print(f"  [{step+1:5d}/{args.max_steps}] loss={loss.item():.4f} "
                  f"recon={recon_loss.item():.4f} vq={vq_loss.item():.4f} "
                  f"lr={cur_lr:.2e} codes_batch={n_batch} cum100={n_cum}/{K_eff} "
                  f"({elapsed:.0f}s, ETA {eta:.0f}s)")
            with open(log_path, 'a') as f:
                f.write(json.dumps({
                    'step': step + 1, 'loss': float(loss),
                    'recon_loss': float(recon_loss), 'vq_loss': float(vq_loss),
                    'lr': cur_lr,
                    'codes_batch': int(n_batch), 'codes_cumulative100': int(n_cum),
                }) + '\n')

        # Validation loss chooses ckpt_best.pt, as in MoReFlow
        if val_clips and (step + 1) % args.val_interval == 0:
            val_loss = evaluate(step + 1)
            print(f"    val_recon={val_loss:.4f} (best so far {min(val_loss, best_val):.4f})")
            with open(log_path, 'a') as f:
                f.write(json.dumps({'step': step + 1, 'val_recon': val_loss}) + '\n')
            if val_loss < best_val:
                best_val = val_loss
                tmp = best_path.with_suffix('.pt.tmp')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'args': vars(args),
                    'skel': skel_name,
                    'n_joints': n_joints,
                    'feat_dim': feat_dim,
                    'codebook_size_effective': K_eff,
                    'downsample_factor': model.downsample_factor,
                    'n_tokens_per_window': n_tokens,
                    'window': WINDOW,
                    'mean': cond[skel_name]['mean'],
                    'std': cond[skel_name]['std'],
                    'step': step + 1,
                    'val_recon': val_loss,
                }, tmp)
                os.replace(tmp, best_path)

        # Periodic checkpoint (preserves optimizer + scheduler + RNG state, atomic)
        if (step + 1) % args.ckpt_interval == 0 and (step + 1) < args.max_steps:
            ck = save_dir / f'ckpt_step{step+1:06d}.pt'
            tmp = ck.with_suffix('.pt.tmp')
            torch.save({
                'step': step + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val': best_val,
                'numpy_rng_state': rng.get_state(),
                'torch_rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            }, tmp)
            os.replace(tmp, ck)
            for old in sorted(save_dir.glob('ckpt_step*.pt'))[:-2]:
                old.unlink()

    # Save final ckpt
    ckpt_path = save_dir / 'ckpt_final.pt'
    tmp_path = ckpt_path.with_suffix('.pt.tmp')
    torch.save({
        'model_state_dict': model.state_dict(),
        'args': vars(args),
        'skel': skel_name,
        'n_joints': n_joints,
        'feat_dim': feat_dim,
        'codebook_size_effective': K_eff,
        'downsample_factor': model.downsample_factor,
        'n_tokens_per_window': n_tokens,
        'window': WINDOW,
        'mean': cond[skel_name]['mean'],
        'std': cond[skel_name]['std'],
    }, tmp_path)
    os.replace(tmp_path, ckpt_path)  # atomic
    # Periodic ckpts no longer needed once final exists
    for stale in save_dir.glob('ckpt_step*.pt'):
        stale.unlink()
    # Clear any prior FAILED marker now that this skel succeeded
    fmark = SAVE_ROOT / f'{skel_name}_FAILED.txt'
    if fmark.exists():
        fmark.unlink()
    print(f"  Saved {ckpt_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skel', type=str, default=None)
    parser.add_argument('--all_train_skeletons', action='store_true',
                        help='Train one tokenizer for each of the 60 training skeletons')
    parser.add_argument('--include_held_out_skeletons', action='store_true',
                        help='Also train tokenizers for the 10 held-out animals, as the paper did. '
                             'A tokenizer only sees its own animal\'s clips.')
    parser.add_argument('--max_steps', type=int, default=100000,
                        help='MoReFlow Appendix A.1: 100k steps per skeleton.')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Per MoReFlow Appendix A.1 (default 0.0)')
    parser.add_argument('--ckpt_interval', type=int, default=1000,
                        help='Save resumable checkpoint every N steps')
    parser.add_argument('--no_resume', action='store_true',
                        help='Ignore existing periodic checkpoints (start fresh)')
    parser.add_argument('--hidden', type=int, default=512,
                        help="MoReFlow's quadruped setting: 512 (humanoid: 256).")
    parser.add_argument('--codebook_size', type=int, default=256,
                        help="MoReFlow's quadruped setting: 256 (humanoid: 512). "
                             'Effective K is data-adaptive (see adaptive_codebook_size).')
    parser.add_argument('--codebook_dim', type=int, default=256,
                        help="MoReFlow's quadruped setting: 256 (humanoid: 128).")
    parser.add_argument('--n_resblocks', type=int, default=3,
                        help='MoReFlow: 3.')
    parser.add_argument('--activation', type=str, default='relu',
                        choices=['relu', 'silu', 'gelu'], help='MoReFlow: relu.')
    parser.add_argument('--dead_code_threshold', type=int, default=5,
                        help='EMA-cluster-size threshold below which a code is replaced with a '
                             'random batch feature. The library default is 2; 5 comes closer to '
                             'the periodic re-initialisation MoReFlow describes.')
    parser.add_argument('--warmup_steps', type=int, default=1000,
                        help='Linear warmup from 0 to lr over N steps, then cosine to 0 (MoReFlow Appendix A.1).')
    parser.add_argument('--val_frac', type=float, default=0.10,
                        help='Fraction of clips held out for validation (clip-level). 0 disables.')
    parser.add_argument('--val_interval', type=int, default=200,
                        help='Compute val recon every N training steps.')
    parser.add_argument('--val_batches', type=int, default=8,
                        help='Number of val minibatches to average per evaluation.')
    parser.add_argument('--n_downsample', type=int, default=2,
                        help='2 → 32-frame window / 4 = 8 tokens (MoReFlow §3.1)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_interval', type=int, default=200)
    parser.add_argument('--config', default=None, type=str,
                        help='JSON file of settings; anything also typed on the command line wins.')
    args = parser.parse_args()
    if args.config:
        apply_config(args, args.config)

    print("Loading cond...")
    cond = np.load(COND_PATH, allow_pickle=True).item()

    # Build skel list
    if args.all_train_skeletons:
        skels = list(OBJECT_SUBSETS_DICT['train'])
        if args.include_held_out_skeletons:
            skels += list(OBJECT_SUBSETS_DICT['held_out'])
            print(f"Training {len(skels)} tokenizers (60 training animals + 10 held out). "
                  f"A tokenizer only sees its own animal's clips.")
        else:
            print(f"Training the {len(skels)} training animals only")
    elif args.skel:
        skels = [args.skel]
    else:
        parser.error("Must specify --skel or --all_train_skeletons")

    SAVE_ROOT.mkdir(parents=True, exist_ok=True)

    # A skeleton that already has ckpt_final.pt is skipped unless --no_resume asks for it to be
    # trained again, so rerunning --all_train_skeletons after an interrupted run trains only
    # what is missing.
    for i, skel in enumerate(skels):
        save_dir = SAVE_ROOT / skel
        ckpt_final = save_dir / 'ckpt_final.pt'
        if ckpt_final.exists() and not args.no_resume:
            print(f"\n[{i+1}/{len(skels)}] {skel}: SKIPPED (existing ckpt_final.pt)")
            continue
        print(f"\n[{i+1}/{len(skels)}] Training VQ-VAE for {skel}")
        try:
            train_one_skel(skel, args, cond)
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()
            with open(SAVE_ROOT / f'{skel}_FAILED.txt', 'w') as f:
                f.write(f"{type(e).__name__}: {e}")


if __name__ == '__main__':
    main()
