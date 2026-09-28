"""Train the adversarial model with a second set of training settings, and score each run.

The adversarial model of `train_adversarial.py` did not converge: its critic and the model
never reached a working balance, and its output grew without bound. This second version
changes the training settings only, to the ones ACE itself uses: a smaller learning rate
with different Adam settings, a penalty that keeps the critic's gradients small on real
motion, and a critic held fixed while the model learns. The objective, the summary of the
source motion, the data, the network, the batch size, the number of steps and the scoring
are all unchanged. The settings were fixed before any of its results were looked at.

Three seeds are run, and all three are reported, beside the first version rather than in
place of it. Each run is scored on the same four measures as score_four_measures.py.

Runs in the main environment, on a GPU. Each seed takes about five minutes, most of it
scoring:
    python -m robots.human_to_g1.train_adversarial_second_setup \
        --out output/robots/g1_adversarial_second_setup.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from robots.human_to_g1.data import DEFAULT_DATA, action_groups, load_clip_list, load_corpus
from robots.human_to_g1.methods import method_reader
from robots.human_to_g1.score import score_rows
from robots.human_to_g1.score_four_measures import (realism_and_action, realism_bank,
                                                    references)
from robots.human_to_g1.train import (HUMAN_JOINTS, ROBOT_POINTS, save_checkpoint,
                                      write_motions)
from robots.human_to_g1.train_adversarial import body_scale, summary_loss
from robots.models import Critic, DirectMap, windows_per_clip
from robots.scoring import summarize

GRADIENT_PENALTY = 0.1
DESCRIPTION = ("Training of the adversarial objective, in the version fixed in advance, did "
               "not converge. These are three seeds of a second version with different "
               "training settings, fixed before any of its results were looked at. They are "
               "reported beside the first version, not in place of it.")


def train(clips, steps, batch, lr, device, seed, critic_weight, summary_weight):
    """Train the model and its critic in turn with the second set of training settings."""
    human_windows, human_scale = windows_per_clip(clips, "human")
    robot_windows, robot_scale = windows_per_clip(clips, "g1")
    human_size, robot_size = body_scale(clips, "human"), body_scale(clips, "g1")
    print(f"  seed {seed}: body size human {human_size:.3f} m, robot {robot_size:.3f} m")
    inputs = [(clip_id, i, w) for clip_id, windows in human_windows.items()
              for i, w in enumerate(windows)]
    real = [w for windows in robot_windows.values() for w in windows]

    model = DirectMap(HUMAN_JOINTS, ROBOT_POINTS).to(device)
    critic = Critic(ROBOT_POINTS).to(device)
    model_opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.9))
    critic_opt = torch.optim.Adam(critic.parameters(), lr=lr, betas=(0.5, 0.9))
    loss_fn = nn.BCEWithLogitsLoss()
    rng = np.random.RandomState(seed)
    model.train()
    critic.train()
    started = time.time()
    for step in range(steps):
        picks = rng.randint(0, len(inputs), batch)
        source_m = torch.from_numpy(np.stack([inputs[i][2] for i in picks])).float().to(device)
        real_picks = rng.randint(0, len(real), batch)
        real_m = torch.from_numpy(np.stack([real[i] for i in real_picks])).float().to(device)
        source = source_m / human_scale
        real_scaled = (real_m / robot_scale).detach().requires_grad_(True)

        # The gradient penalty needs a second derivative, which the fast attention kernels
        # do not provide, so the critic's step runs under the plain one.
        with sdpa_kernel(SDPBackend.MATH):
            on_real = critic(real_scaled)
            real_loss = loss_fn(on_real, torch.ones_like(on_real))
            slope = torch.autograd.grad(on_real.sum(), real_scaled, create_graph=True,
                                        retain_graph=True)[0]
            penalty = 0.5 * GRADIENT_PENALTY * (slope ** 2).flatten(1).sum(dim=-1).mean()
            with torch.no_grad():
                produced = model(source)
            on_produced = critic(produced)
            critic_loss = (real_loss
                           + loss_fn(on_produced, torch.zeros_like(on_produced)) + penalty)
            critic_opt.zero_grad()
            critic_loss.backward()
        nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
        critic_opt.step()

        for p in critic.parameters():
            p.requires_grad_(False)
        produced = model(source)
        fooling = loss_fn(critic(produced), torch.ones(produced.shape[0], device=device))
        matching = summary_loss(source_m, produced * robot_scale, human_size, robot_size)
        model_loss = critic_weight * fooling + summary_weight * matching
        model_opt.zero_grad()
        model_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        model_opt.step()
        for p in critic.parameters():
            p.requires_grad_(True)

        if (step + 1) % max(1, steps // 5) == 0:
            print(f"  seed {seed} [{step + 1}/{steps}] fooling={fooling.item():.3f} "
                  f"matching={matching.item():.3f} critic={critic_loss.item():.3f} "
                  f"penalty={penalty.item():.4f} ({time.time() - started:.0f}s)")
    return model, human_scale, robot_scale


def score_run(name, groups, data_dir, generated, bank, threshold, stretched, by_action,
              action_of):
    """The same four measures the main table uses, for one run."""
    reader = method_reader(name, data_dir, generated)
    raw = summarize(name, score_rows(groups, data_dir, reader, "raw"))
    stretched_fit = summarize(name, score_rows(groups, data_dir, reader, "length_controlled"))
    extra = realism_and_action(reader, groups, bank, threshold, stretched, by_action,
                               action_of)
    return {
        "sif": raw.get("sif"), "sif_length_controlled": stretched_fit.get("sif"),
        "sif_ci95": raw.get("sif_ci95"), "p_value": raw.get("p_value"),
        "variation_mean": raw.get("variation_mean"), "variation": raw.get("variation"),
        "realistic_pct": extra["realistic_pct"],
        "mean_distance_to_nearest_real_clip": extra["mean_distance_to_nearest_real_clip"],
        "action_auc": extra["action_auc_mean"], "action_auc_ci": extra["action_auc_ci"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--critic_weight", type=float, default=1.0)
    ap.add_argument("--summary_weight", type=float, default=1.0)
    ap.add_argument("--data_dir", default=str(DEFAULT_DATA))
    ap.add_argument("--generated_dir", default=None)
    ap.add_argument("--checkpoint_dir", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)
    generated = Path(args.generated_dir) if args.generated_dir else data_dir / "generated"
    checkpoints = (Path(args.checkpoint_dir) if args.checkpoint_dir
                   else data_dir / "checkpoints")
    clips, training_groups = load_corpus(data_dir)
    print(f"{len(clips)} clips over {len(training_groups)} actions on {device}")

    groups = action_groups(load_clip_list(data_dir), data_dir, min_clips=2)
    bank, threshold, held_out_median, n_held_out = realism_bank(groups, data_dir)
    print(f"realistic below {threshold:.4f} (held-out real clips: median "
          f"{held_out_median:.4f}, {n_held_out} clips)")
    stretched, by_action = references(groups, data_dir)
    action_of = {c: g["action"] for g in groups for c in g["clip_ids"]}

    seeds = [int(s) for s in args.seeds.split(",")]
    runs = []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model, human_scale, robot_scale = train(clips, args.steps, args.batch, args.lr,
                                                device, seed, args.critic_weight,
                                                args.summary_weight)
        name = f"adversarial_second_setup_seed{seed}"
        save_checkpoint(checkpoints / f"{name}.pt", name, model, human_scale, robot_scale)
        write_motions(model, name, clips, human_scale, robot_scale, device, generated / name)
        run = score_run(name, groups, data_dir, generated, bank, threshold, stretched,
                        by_action, action_of)
        run["seed"] = seed
        runs.append(run)
        print(f"  seed {seed}: SIF {run['sif']:+.3f} "
              f"(length controlled {run['sif_length_controlled']:+.3f}), "
              f"variation {run['variation']:.4f}, realistic {run['realistic_pct']}%, "
              f"action AUC {run['action_auc']:.3f}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"description": DESCRIPTION, "seeds": seeds,
                   "realistic_threshold": threshold, "runs": runs}, f, indent=2)
    print(f"  written to {args.out}")


if __name__ == "__main__":
    main()
