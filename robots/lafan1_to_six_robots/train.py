"""Train the three models for every robot and every setting, and write their motions.

The three models are the same as in the human-to-G1 study, and so are their size, their
number of steps and their optimiser. Only the number of joints changes from robot to robot.

  unpaired_objective   one autoencoder per skeleton, neither ever seeing the other
  averaging_objective  asked to match a robot clip drawn at random from the same action
                       group
  true_pair_model      asked to match the true counterpart of its input

An action group with only one clip cannot supply "another clip of the same group", so its
clips are left out of training for the second and third models, which keeps their training
data the same as each other's. The first model keeps them, since it never pairs anything.
Motions are produced for every clip regardless. Each trained model is saved under
`<data_dir>/checkpoints/<robot>/<setting>/`.

Runs in the main environment, on a GPU, in a few minutes per robot and setting:
    python -m robots.lafan1_to_six_robots.train
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from robots.lafan1_to_six_robots.data import (DEFAULT_DATA, SETTINGS, load_clip_list,
                                              tensor_path)
from robots.models import (FRAMES, AutoencoderPair, DirectMap, kl_divergence, root_center,
                           split_windows, windows_per_clip)

HUMAN_JOINTS = 22
MODELS = ["unpaired_objective", "averaging_objective", "true_pair_model"]


def load_corpus(data_dir, clip_list, robot, setting, max_groups=0):
    """Clips that have both a human and a robot file, restricted to one setting."""
    allowed_by_group = clip_list["settings"][setting]
    clips, groups = {}, {}
    for group in clip_list["action_groups"]:
        allowed = set(allowed_by_group.get(group["name"], []))
        kept = []
        for clip in group["clips"]:
            clip_id = clip["clip_id"]
            if clip_id not in allowed:
                continue
            human = tensor_path(data_dir, "human", clip_id)
            robot_path = tensor_path(data_dir, robot, clip_id)
            if human.exists() and robot_path.exists():
                clips[clip_id] = {"human": np.load(human), "robot": np.load(robot_path),
                                  "group": group["name"]}
                kept.append(clip_id)
        if kept:
            groups[group["name"]] = kept
        if max_groups and len(groups) >= max_groups:
            break
    return clips, groups


def train_direct_map(clips, robot_points, steps, batch, lr, device, seed, use_true_pairs,
                     name):
    """Train the one-pass human-to-robot model against one of the two kinds of target."""
    human_windows, human_scale = windows_per_clip(clips, "human")
    robot_windows, robot_scale = windows_per_clip(clips, "robot")
    inputs, of_clip, window_index, of_group = [], [], [], []
    for clip_id, windows in human_windows.items():
        for i, window in enumerate(windows):
            inputs.append(window)
            of_clip.append(clip_id)
            window_index.append(i)
            of_group.append(clips[clip_id]["group"])
    by_group = {}
    for clip_id, windows in robot_windows.items():
        by_group.setdefault(clips[clip_id]["group"], []).append((clip_id, windows))

    model = DirectMap(HUMAN_JOINTS, robot_points).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    rng = np.random.RandomState(seed)
    model.train()
    started = time.time()
    for step in range(steps):
        picks = rng.randint(0, len(inputs), batch)
        source = np.stack([inputs[i] for i in picks]) / human_scale
        target = np.empty((batch, FRAMES, robot_points, 3), np.float32)
        for b, i in enumerate(picks):
            if use_true_pairs:
                windows = robot_windows[of_clip[i]]
                target[b] = windows[min(window_index[i], len(windows) - 1)]
            else:
                pool = [x for x in by_group[of_group[i]] if x[0] != of_clip[i]] \
                    or by_group[of_group[i]]
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
                  f"({time.time() - started:.0f}s)", flush=True)
    return model, human_scale, robot_scale


def train_autoencoder_pair(clips, robot_points, steps, batch, lr, device, seed, beta=1.0):
    """Train one autoencoder per skeleton; neither is ever shown the other skeleton."""
    human_windows, human_scale = windows_per_clip(clips, "human")
    robot_windows, robot_scale = windows_per_clip(clips, "robot")
    human_pool = [w for windows in human_windows.values() for w in windows]
    robot_pool = [w for windows in robot_windows.values() for w in windows]

    model = AutoencoderPair(HUMAN_JOINTS, robot_points).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    rng = np.random.RandomState(seed)
    model.train()
    started = time.time()
    for step in range(steps):
        human = torch.from_numpy(np.stack([human_pool[i] for i in
                                           rng.randint(0, len(human_pool), batch)])
                                 / human_scale).float().to(device)
        robot = torch.from_numpy(np.stack([robot_pool[i] for i in
                                           rng.randint(0, len(robot_pool), batch)])
                                 / robot_scale).float().to(device)
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
                  f"({time.time() - started:.0f}s)", flush=True)
    return model, human_scale, robot_scale


@torch.no_grad()
def write_motions(model, name, clips, human_scale, robot_scale, robot_points, device,
                  out_dir):
    """One robot motion per human clip: every whole 64-frame window, 256 frames here."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    for clip_id, clip in clips.items():
        windows = split_windows(root_center(clip["human"].astype(np.float32)), FRAMES)
        source = torch.from_numpy(windows / human_scale).float().to(device)
        produced = model.cross(source) if name == "unpaired_objective" else model(source)
        motion = (produced.cpu().numpy() * robot_scale).reshape(-1, robot_points, 3)
        np.save(out_dir / f"{clip_id}.npy",
                motion[:clip["robot"].shape[0]].astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default=None, help="default: every robot in the clip list")
    ap.add_argument("--setting", default="both", choices=SETTINGS + ["both"])
    ap.add_argument("--model", default="all", choices=MODELS + ["all"])
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
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
    clip_list = load_clip_list(data_dir)
    robots = [args.robot] if args.robot else list(clip_list["meta"]["robots"].keys())
    settings = SETTINGS if args.setting == "both" else [args.setting]
    models = MODELS if args.model == "all" else [args.model]

    for robot in robots:
        robot_points = int(clip_list["meta"]["robots"][robot]["n_points"])
        for setting in settings:
            clips, groups = load_corpus(data_dir, clip_list, robot, setting,
                                        args.max_groups)
            paired = {c: d for c, d in clips.items() if len(groups[d["group"]]) >= 2}
            alone = len(clips) - len(paired)
            print(f"{robot} [{setting}] {robot_points} points: {len(clips)} clips "
                  f"({alone} in action groups with a single clip, left out of the second "
                  f"and third models), {len(groups)} action groups, {device}", flush=True)
            if len(groups) < 2:
                print("   skipped: needs at least two action groups")
                continue
            for name in models:
                torch.manual_seed(args.seed)
                np.random.seed(args.seed)
                if name == "unpaired_objective":
                    model, human_scale, robot_scale = train_autoencoder_pair(
                        clips, robot_points, args.steps, args.batch, args.lr, device,
                        args.seed)
                else:
                    model, human_scale, robot_scale = train_direct_map(
                        paired, robot_points, args.steps, args.batch, args.lr, device,
                        args.seed, use_true_pairs=(name == "true_pair_model"), name=name)
                folder = checkpoints / robot / setting
                folder.mkdir(parents=True, exist_ok=True)
                torch.save({"model": name, "robot": robot, "setting": setting,
                            "state_dict": model.state_dict(),
                            "human_scale": float(human_scale),
                            "robot_scale": float(robot_scale)},
                           folder / f"{name}.pt")
                write_motions(model, name, clips, human_scale, robot_scale, robot_points,
                              device, out_dir / robot / setting / name)
                print(f"   {name}: motions in {out_dir / robot / setting / name}",
                      flush=True)


if __name__ == "__main__":
    main()
