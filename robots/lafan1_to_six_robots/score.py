"""Score every robot and every setting on Source-Instance Fidelity (SIF) and variation.

For one robot and one setting, each action group gives the human clips of one routine and
the robot motions a method produced for them. SIF is the correlation between how different
the human clips are from one another and how different the robot motions are.

Action groups with fewer than three clips are left out, because two clips give a single
distance and no correlation. Action groups with more than six clips are cut to six, so
every possible pairing of outputs to sources can be enumerated exactly for the shuffle
test.

Three things are reported beside the score: the quantities behind the five predictions the
authors wrote down before any trained model was scored (listed in the paper's appendix on
the robot studies), an interval for each robot separately, and a pooled figure across
robots whose interval resamples whole action groups, because the same human routines recur
on all six robots.

Runs in the main environment, on the processor, in about ten minutes:
    python -m robots.lafan1_to_six_robots.score --out output/robots/six_robots_sif.json
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from robots.lafan1_to_six_robots.data import (DEFAULT_DATA, HORIZON, SETTINGS,
                                              load_clip_list, load_tensor, tensor_path)
from robots.lafan1_to_six_robots.methods import METHODS, method_reader
from robots.scoring import resample_mean, score_group, summarize, variation_ratio

LENGTHS = ["raw", "length_controlled"]
MIN_CLIPS, MAX_CLIPS = 3, 6
DESCRIPTION = ("Source-Instance Fidelity (SIF) and variation for six humanoid robots, each "
               "in a sparse and a dense setting, with motions compared raw and length "
               "controlled. \"results\" holds the score of every model. "
               "\"prediction_checks\" holds the quantities behind the predictions the "
               "authors wrote down before any trained model was scored, including "
               "\"variation_vs_true_retarget\" (the paper's Q), how much of the true "
               "retarget's variation a model keeps. \"pooled\" averages each model's SIF "
               "over the six robots; \"n_scores\" counts the robot and action group pairs "
               "averaged. Keys read robot|setting|scoring.")


def check_complete(clip_list, data_dir, generated, robots, settings, models):
    """Stop before scoring if any model is missing motions, so no model is scored on less.

    Every produced motion must exist, have the shape the robot's skeleton implies and hold
    no missing values.
    """
    problems = []
    for robot in robots:
        n_points = int(clip_list["meta"]["robots"][robot]["n_points"])
        for setting in settings:
            clip_ids = sorted({c for v in clip_list["settings"][setting].values()
                               for c in v})
            for model in models:
                folder = Path(generated) / robot / setting / model
                for clip_id in clip_ids:
                    path = folder / f"{clip_id}.npy"
                    if not path.exists():
                        problems.append(f"missing {path}")
                        continue
                    motion = np.load(path)
                    if motion.shape != (HORIZON, n_points, 3):
                        problems.append(f"shape {motion.shape} should be "
                                        f"({HORIZON},{n_points},3): {path}")
                    elif not np.isfinite(motion).all():
                        problems.append(f"missing values in {path}")
    if problems:
        for p in problems[:20]:
            print("incomplete:", p)
        raise SystemExit(f"{len(problems)} problems found (first 20 shown); nothing scored")
    print("all produced motions present and well formed", flush=True)


def score_rows(clip_list, data_dir, generated, robot, setting, model, length):
    """Score one model on one robot and setting, action group by action group."""
    setting_clips = clip_list["settings"][setting]
    reader = method_reader(model, data_dir, robot, setting, generated)
    scored = []
    for entry in clip_list["action_groups"]:
        clip_ids = [c for c in setting_clips.get(entry["name"], [])
                    if tensor_path(data_dir, "human", c).exists()]
        if len(clip_ids) < MIN_CLIPS:
            continue
        clip_ids = clip_ids[:MAX_CLIPS]
        sources = [load_tensor(data_dir, "human", c) for c in clip_ids]
        outputs = [reader(c, clip_ids) for c in clip_ids]
        group = score_group(entry["name"], sources, outputs, length)
        if group is not None:
            group["robot"] = robot
            scored.append(group)
    return scored


def pooled_interval(groups, n_resamples=10000, seed=42):
    """Average SIF across robots, with a 95% interval that resamples whole action groups."""
    values = np.array([g["sif"] for g in groups], float)
    by_name = defaultdict(list)
    for i, group in enumerate(groups):
        by_name[group["name"]].append(i)
    names = list(by_name.keys())
    rng = np.random.RandomState(seed)
    drawn = np.empty(n_resamples)
    for b in range(n_resamples):
        picks = rng.choice(len(names), len(names), replace=True)
        idx = [i for p in picks for i in by_name[names[p]]]
        drawn[b] = values[np.asarray(idx)].mean()
    drawn.sort()
    return {"mean": float(values.mean()),
            "ci95": [float(drawn[int(0.025 * n_resamples)]),
                     float(drawn[int(0.975 * n_resamples)])],
            "n_action_groups": len(names)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robots", default=None, help="comma list; default every robot")
    ap.add_argument("--data_dir", default=str(DEFAULT_DATA))
    ap.add_argument("--generated_dir", default=None)
    ap.add_argument("--skip_completeness_check", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    generated = Path(args.generated_dir) if args.generated_dir else data_dir / "generated"
    clip_list = load_clip_list(data_dir)
    robots = (args.robots.split(",") if args.robots
              else list(clip_list["meta"]["robots"].keys()))
    models = [m for m in METHODS if m not in ("true_retarget", "random_clip")]
    if not args.skip_completeness_check:
        check_complete(clip_list, data_dir, generated, robots, SETTINGS, models)

    out = {"description": DESCRIPTION, "horizon_frames": HORIZON,
           "min_clips_per_action_group": MIN_CLIPS, "robots": robots,
           "results": {}, "prediction_checks": {}, "pooled": {}}
    pooled = defaultdict(list)

    for setting in SETTINGS:
        for length in LENGTHS:
            for robot in robots:
                key = f"{robot}|{setting}|{length}"
                out["results"][key] = []
                per_model = {}
                for model in METHODS:
                    groups = score_rows(clip_list, data_dir, generated, robot, setting,
                                        model, length)
                    row = summarize(model, groups)
                    row["per_action_group"] = {
                        g["name"]: {"sif": g["sif"], "variation": g["variation"]}
                        for g in groups}
                    per_model[model] = row["per_action_group"]
                    out["results"][key].append(row)
                    pooled[(setting, length, model)].extend(groups)
                    if row["n_action_groups"]:
                        print(f"{key:44s} {model:20s} "
                              f"action groups={row['n_action_groups']:3d} "
                              f"SIF={row['sif']:+.3f} "
                              f"variation={row['variation']:.3f} "
                              f"p={row['p_value']:.4f}", flush=True)
                    else:
                        print(f"{key:44s} {model:20s} no motions found", flush=True)

                reference = per_model.get("true_retarget", {})
                checks = {}
                for model, scores in per_model.items():
                    if not scores:
                        continue
                    entry = {"sif_interval": resample_mean(
                        [v["sif"] for v in scores.values()])}
                    if model != "true_retarget" and reference:
                        entry["variation_vs_true_retarget"] = variation_ratio(
                            {g: v["variation"] for g, v in scores.items()},
                            {g: v["variation"] for g, v in reference.items()})
                    checks[model] = entry
                out["prediction_checks"][key] = checks

    for (setting, length, model), groups in pooled.items():
        if not groups:
            continue
        by_robot = defaultdict(list)
        for group in groups:
            by_robot[group["robot"]].append(group["sif"])
        out["pooled"][f"{setting}|{length}|{model}"] = {
            "n_scores": len(groups),
            "sif_over_robots": pooled_interval(groups),
            "sif_per_robot": {r: float(np.mean(v)) for r, v in by_robot.items()},
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
