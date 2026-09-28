"""Train the three human-to-G1 models the paper reports, and write their motions.

All three read the same clips and have the same size, the same number of training steps
and the same optimiser. They differ only in what they are asked to match:

unpaired_objective   Each skeleton gets its own autoencoder, a network that squeezes a
                     window of motion into a short code and rebuilds it from that code.
                     The two share one space of codes (a latent space), but neither ever
                     sees the other skeleton; the human-to-robot map is whatever that
                     shared space happens to give. This is the family the paper's first
                     result is about: nothing in the training tells the model which human
                     clip a robot motion should belong to.
averaging_objective  The model is asked to match a robot clip drawn at random from the
                     same action, redrawn at every step, rather than the true counterpart
                     of its input. The best answer to that request is the average motion
                     of the action, which is the same answer for every input.
true_pair_model      The same model asked to match the true counterpart of its input,
                     frame for frame. This shows what the same model can do when the
                     correspondence is given rather than guessed.

Each model is saved in `<data_dir>/checkpoints/` with the two scales used to normalise the
human and robot sides, so `generate.py` can write its motions again without retraining.

Runs in the main environment. On our GPU the three models took 173 seconds in all (about a
minute and a half for the unpaired objective and half a minute for each of the other two);
on a CPU they take far longer:
    python -m robots.human_to_g1.train --model all
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from robots.human_to_g1.data import DEFAULT_DATA, load_corpus
from robots.models import (FRAMES, AutoencoderPair, DirectMap, kl_divergence,
                           root_center, split_windows, windows_per_clip)

HUMAN_JOINTS, ROBOT_POINTS = 29, 30
MODELS = ["unpaired_objective", "averaging_objective", "true_pair_model"]


def train_direct_map(clips, steps, batch, lr, device, seed, use_true_pairs, name):
    """Train the one-pass human-to-robot model against one of the two kinds of target."""
    human_windows, human_scale = windows_per_clip(clips, "human")
    robot_windows, robot_scale = windows_per_clip(clips, "g1")
    inputs, of_clip, window_index, of_action = [], [], [], []
    for clip_id, windows in human_windows.items():
        for i, window in enumerate(windows):
            inputs.append(window)
            of_clip.append(clip_id)
            window_index.append(i)
            of_action.append(clips[clip_id]["action"])
    by_action = {}
    for clip_id, windows in robot_windows.items():
        by_action.setdefault(clips[clip_id]["action"], []).append((clip_id, windows))

    model = DirectMap(HUMAN_JOINTS, ROBOT_POINTS).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    rng = np.random.RandomState(seed)
    model.train()
    started = time.time()
    for step in range(steps):
        picks = rng.randint(0, len(inputs), batch)
        source = np.stack([inputs[i] for i in picks]) / human_scale
        target = np.empty((batch, FRAMES, ROBOT_POINTS, 3), np.float32)
        for b, i in enumerate(picks):
            if use_true_pairs:
                windows = robot_windows[of_clip[i]]
                target[b] = windows[min(window_index[i], len(windows) - 1)]
            else:
                pool = [x for x in by_action[of_action[i]] if x[0] != of_clip[i]] \
                    or by_action[of_action[i]]
                _, windows = pool[rng.randint(len(pool))]
                target[b] = windows[rng.randint(len(windows))]
        source = torch.from_numpy(source).float().to(device)
        target = torch.from_numpy(target / robot_scale).float().to(device)
        loss = F.mse_loss(model(source), target)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if (step + 1) % max(1, steps // 5) == 0:
            print(f"  {name} [{step + 1}/{steps}] error={loss.item():.4f} "
                  f"({time.time() - started:.0f}s)")
    return model, human_scale, robot_scale


def train_autoencoder_pair(clips, steps, batch, lr, device, seed, beta=1.0):
    """Train one autoencoder per skeleton; neither is ever shown the other skeleton."""
    human_windows, human_scale = windows_per_clip(clips, "human")
    robot_windows, robot_scale = windows_per_clip(clips, "g1")
    human_pool = [w for windows in human_windows.values() for w in windows]
    robot_pool = [w for windows in robot_windows.values() for w in windows]

    model = AutoencoderPair(HUMAN_JOINTS, ROBOT_POINTS).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    rng = np.random.RandomState(seed)
    model.train()
    started = time.time()
    for step in range(steps):
        human = _draw(human_pool, batch, rng, device, human_scale)
        robot = _draw(robot_pool, batch, rng, device, robot_scale)
        human_out, human_mean, human_log_var = model.human(human)
        robot_out, robot_mean, robot_log_var = model.robot(robot)
        loss = (F.mse_loss(human_out, human) + F.mse_loss(robot_out, robot)
                + beta * (kl_divergence(human_mean, human_log_var)
                          + kl_divergence(robot_mean, robot_log_var))
                / (HUMAN_JOINTS * 3))
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if (step + 1) % max(1, steps // 5) == 0:
            print(f"  unpaired_objective [{step + 1}/{steps}] error={loss.item():.4f} "
                  f"({time.time() - started:.0f}s)")
    return model, human_scale, robot_scale


def _draw(pool, batch, rng, device, scale):
    """A batch of windows drawn at random from `pool`, divided by `scale`."""
    picks = rng.randint(0, len(pool), batch)
    return torch.from_numpy(np.stack([pool[i] for i in picks]) / scale).float().to(device)


@torch.no_grad()
def write_motions(model, name, clips, human_scale, robot_scale, device, out_dir):
    """One robot motion per human clip, written as one file each.

    The clip is cut into back-to-back 64-frame windows and the motion covers every whole
    window; frames left over at the end are not produced. A clip shorter than one window
    is padded, and its motion is cut back to the clip's own length.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    for clip_id, clip in clips.items():
        windows = split_windows(root_center(clip["human"].astype(np.float32)))
        source = torch.from_numpy(windows / human_scale).float().to(device)
        produced = model.cross(source) if name == "unpaired_objective" else model(source)
        motion = (produced.cpu().numpy() * robot_scale).reshape(-1, ROBOT_POINTS, 3)
        np.save(out_dir / f"{clip_id}.npy", motion[:clip["g1"].shape[0]].astype(np.float32))


def save_checkpoint(path, name, model, human_scale, robot_scale):
    """Save a trained model with its name and the two normalising scales."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": name, "state_dict": model.state_dict(),
                "human_scale": float(human_scale), "robot_scale": float(robot_scale)}, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all", choices=MODELS + ["all"])
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data_dir", default=str(DEFAULT_DATA))
    ap.add_argument("--out_dir", default=None,
                    help="where the produced motions go (default <data_dir>/generated)")
    ap.add_argument("--checkpoint_dir", default=None,
                    help="where trained models go (default <data_dir>/checkpoints)")
    ap.add_argument("--max_groups", type=int, default=0,
                    help="use only the first few actions")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir) if args.out_dir else data_dir / "generated"
    checkpoints = (Path(args.checkpoint_dir) if args.checkpoint_dir
                   else data_dir / "checkpoints")
    clips, groups = load_corpus(data_dir, max_groups=args.max_groups)
    print(f"{len(clips)} clips over {len(groups)} actions on {device}, "
          f"{args.steps} steps of {args.batch}")

    for name in (MODELS if args.model == "all" else [args.model]):
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if name == "unpaired_objective":
            model, human_scale, robot_scale = train_autoencoder_pair(
                clips, args.steps, args.batch, args.lr, device, args.seed)
        else:
            model, human_scale, robot_scale = train_direct_map(
                clips, args.steps, args.batch, args.lr, device, args.seed,
                use_true_pairs=(name == "true_pair_model"), name=name)
        save_checkpoint(checkpoints / f"{name}.pt", name, model, human_scale, robot_scale)
        write_motions(model, name, clips, human_scale, robot_scale, device, out_dir / name)
        print(f"  {name}: saved {checkpoints / f'{name}.pt'}, "
              f"motions in {out_dir / name}")


if __name__ == "__main__":
    main()
