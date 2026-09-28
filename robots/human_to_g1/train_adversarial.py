"""Train the adversarial human-to-G1 model.

The three models in `train.py` all belong to the family the paper's argument is about:
nothing in their training says which human clip a robot motion came from. This fourth
model steps outside that family in the way ACE, one of the methods the paper evaluates,
does. It keeps the same network but trains it against two things at once:

  * a critic, a second network that learns to tell real robot motion from produced motion.
    The model is trained to make the critic fail, which pushes its output to look like
    real robot motion instead of collapsing onto an average;
  * a short summary of the source motion, which ties the output to the particular human
    clip it came from.

ACE's summary of a motion also includes joint rotations and angular velocities, which
cannot be recovered from joint positions alone, so this model uses a ten-number,
position-based subset of it: per frame, how high the hips are, how fast they move, and for
each hand and foot how high it is and how far out it reaches. Everything is divided by the
body's own size, so the human and the robot are comparable, and facing direction is left
out, because the two skeletons use different up axes and their headings cannot be matched.

This is the version fixed in advance. Its training did not converge; the paper reports it
as it came out, beside a second version with different training settings
(train_adversarial_second_setup.py).

Runs in the main environment, on a GPU:
    python -m robots.human_to_g1.train_adversarial
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from robots.human_to_g1.data import DEFAULT_DATA, load_corpus
from robots.human_to_g1.train import (HUMAN_JOINTS, ROBOT_POINTS, save_checkpoint,
                                      write_motions)
from robots.models import Critic, DirectMap, root_center, windows_per_clip

NAME = "adversarial_objective"
# Hands and feet on each side, and which axis points up for each skeleton.
HUMAN_ENDS = [21, 26, 14, 18]    # left foot, right foot, left hand, right hand
ROBOT_ENDS = [6, 12, 22, 29]     # left ankle, right ankle, left wrist, right wrist
HUMAN_UP, ROBOT_UP = 1, 2


def body_scale(clips, side):
    """How big a skeleton is, in metres: the typical spread of its joints about the root."""
    spreads = []
    for clip in clips.values():
        centred = root_center(clip[side].astype(np.float32))
        radius = np.sqrt((centred ** 2).sum(-1))
        spreads.append(np.sqrt((radius ** 2).mean(-1)))
    return float(np.median(np.concatenate(spreads)))


def _up_axis(axis, reference):
    """A unit vector along `axis`, on the same device and in the same type as `reference`."""
    unit = torch.zeros(3, device=reference.device, dtype=reference.dtype)
    unit[axis] = 1.0
    return unit


def motion_summary(motion, side, size):
    """Ten numbers per frame describing a motion, free of facing and of body size."""
    axis = HUMAN_UP if side == "human" else ROBOT_UP
    ends = HUMAN_ENDS if side == "human" else ROBOT_ENDS
    root = motion[:, :, 0:1, :]
    relative = motion - root
    root_height = motion[:, :, 0, axis:axis + 1]
    step = torch.cat([torch.zeros_like(motion[:, :1, 0, :]),
                      motion[:, 1:, 0, :] - motion[:, :-1, 0, :]], dim=1)
    root_speed = torch.linalg.norm(step, dim=-1, keepdim=True)
    ends_relative = relative[:, :, ends, :]
    ends_height = ends_relative[..., axis]
    sideways = ends_relative - ends_height.unsqueeze(-1) * _up_axis(axis, ends_relative)
    ends_reach = torch.linalg.norm(sideways, dim=-1)
    summary = torch.cat([root_height, root_speed,
                         ends_height.reshape(*ends_height.shape[:2], -1),
                         ends_reach.reshape(*ends_reach.shape[:2], -1)], dim=-1)
    return summary / size


def summary_loss(human, robot, human_size, robot_size):
    """How far the produced robot motion's summary is from the source clip's summary."""
    return torch.linalg.norm(motion_summary(human, "human", human_size)
                             - motion_summary(robot, "g1", robot_size), dim=-1).mean()


def train(clips, steps, batch, lr, device, seed, critic_weight, summary_weight):
    """Train the model and its critic in turn, one step each, for `steps` steps."""
    human_windows, human_scale = windows_per_clip(clips, "human")
    robot_windows, robot_scale = windows_per_clip(clips, "g1")
    human_size, robot_size = body_scale(clips, "human"), body_scale(clips, "g1")
    print(f"  body size: human {human_size:.3f} m, robot {robot_size:.3f} m; "
          f"critic weight {critic_weight}, summary weight {summary_weight}")

    # The critic only ever sees real robot windows drawn independently of the human input,
    # so it never learns which human clip a robot window belongs to.
    inputs = [(clip_id, i, w) for clip_id, windows in human_windows.items()
              for i, w in enumerate(windows)]
    real = [w for windows in robot_windows.values() for w in windows]

    model = DirectMap(HUMAN_JOINTS, ROBOT_POINTS).to(device)
    critic = Critic(ROBOT_POINTS).to(device)
    model_opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    critic_opt = torch.optim.AdamW(critic.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = nn.BCEWithLogitsLoss()
    rng = np.random.RandomState(seed)
    model.train()
    critic.train()
    started = time.time()
    for step in range(steps):
        picks = rng.randint(0, len(inputs), batch)
        source_m = np.stack([inputs[i][2] for i in picks])
        real_picks = rng.randint(0, len(real), batch)
        real_m = np.stack([real[i] for i in real_picks])
        source_m = torch.from_numpy(source_m).float().to(device)
        real_m = torch.from_numpy(real_m).float().to(device)
        source = source_m / human_scale
        real_scaled = real_m / robot_scale

        with torch.no_grad():
            produced = model(source)
        on_real = critic(real_scaled)
        on_produced = critic(produced)
        critic_loss = (loss_fn(on_real, torch.ones_like(on_real))
                       + loss_fn(on_produced, torch.zeros_like(on_produced)))
        critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
        critic_opt.step()

        produced = model(source)
        fooling = loss_fn(critic(produced), torch.ones_like(on_produced))
        matching = summary_loss(source_m, produced * robot_scale, human_size, robot_size)
        model_loss = critic_weight * fooling + summary_weight * matching
        model_opt.zero_grad()
        model_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        model_opt.step()

        if (step + 1) % max(1, steps // 5) == 0:
            print(f"  {NAME} [{step + 1}/{steps}] fooling={fooling.item():.3f} "
                  f"matching={matching.item():.3f} critic={critic_loss.item():.3f} "
                  f"({time.time() - started:.0f}s)")
    return model, human_scale, robot_scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--critic_weight", type=float, default=1.0)
    ap.add_argument("--summary_weight", type=float, default=1.0)
    ap.add_argument("--data_dir", default=str(DEFAULT_DATA))
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--checkpoint_dir", default=None)
    ap.add_argument("--max_groups", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir) if args.out_dir else data_dir / "generated"
    checkpoints = (Path(args.checkpoint_dir) if args.checkpoint_dir
                   else data_dir / "checkpoints")
    clips, groups = load_corpus(data_dir, max_groups=args.max_groups)
    print(f"{len(clips)} clips over {len(groups)} actions on {device}, "
          f"{args.steps} steps of {args.batch}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model, human_scale, robot_scale = train(clips, args.steps, args.batch, args.lr,
                                            device, args.seed, args.critic_weight,
                                            args.summary_weight)
    save_checkpoint(checkpoints / f"{NAME}.pt", NAME, model, human_scale, robot_scale)
    write_motions(model, NAME, clips, human_scale, robot_scale, device, out_dir / NAME)
    print(f"  {NAME}: saved {checkpoints / f'{NAME}.pt'}, motions in {out_dir / NAME}")


if __name__ == "__main__":
    main()
