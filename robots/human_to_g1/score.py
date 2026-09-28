"""Score how much of each human performance survives the trip to the robot.

This computes Source-Instance Fidelity (SIF) and variation on their own, without the slower
action and realism measures of score_four_measures.py. For every action group, the human
clips are compared with one another and the robot motions are compared with one another,
and the two sets of distances are correlated. A method that keeps what made each
performance different scores near one; a method that returns the same robot motion
whatever it is given scores near zero. Alongside it, variation says how far apart the robot
motions are compared with their sources.

Two reference rows come with no model at all: the true robot counterpart of each clip,
which shows the measure recognises a genuine retarget, and a random robot clip of the same
action, which shows what knowing only the action name is worth.

Runs in the main environment, on the processor, in a few minutes. Once per length setting:
    python -m robots.human_to_g1.score --length raw --out <file.json>
    python -m robots.human_to_g1.score --length length_controlled --out <file.json>
"""
import argparse
import json
from pathlib import Path

from robots.human_to_g1.data import DEFAULT_DATA, action_groups, load_clip_list, load_tensor
from robots.human_to_g1.methods import METHODS, method_reader
from robots.scoring import score_group, summarize

DESCRIPTION = ("Source-Instance Fidelity (SIF) and variation for human motion capture "
               "retargeted to the Unitree G1: how much of each human performance survives, "
               "and how far apart the resulting robot motions are.")


def score_rows(groups, data_dir, reader, length):
    """Score every action group for one method."""
    scored = []
    for group in groups:
        sources = [load_tensor(data_dir, "human", c) for c in group["clip_ids"]]
        outputs = [reader(c, group) for c in group["clip_ids"]]
        result = score_group(group["action"], sources, outputs, length)
        if result is not None:
            scored.append(result)
    return scored


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default=",".join(METHODS[:5]))
    ap.add_argument("--length", default="raw", choices=["raw", "length_controlled"])
    ap.add_argument("--data_dir", default=str(DEFAULT_DATA))
    ap.add_argument("--generated_dir", default=None,
                    help="folder holding one subfolder of motions per model")
    ap.add_argument("--min_clips", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    generated = Path(args.generated_dir) if args.generated_dir else data_dir / "generated"
    groups = action_groups(load_clip_list(data_dir), data_dir, min_clips=args.min_clips)
    print(f"{len(groups)} action groups, comparing motions {args.length.replace('_', ' ')}")

    rows = []
    for method in [m.strip() for m in args.methods.split(",")]:
        reader = method_reader(method, data_dir, generated)
        row = summarize(method, score_rows(groups, data_dir, reader, args.length))
        rows.append(row)
        if row["n_action_groups"]:
            print(f"  {method:20s} action groups={row['n_action_groups']:3d}  "
                  f"SIF={row['sif']:+.4f}  variation={row['variation']:.3f}  "
                  f"p={row['p_value']:.4f}  "
                  f"interval[{row['sif_ci95'][0]:+.3f},{row['sif_ci95'][1]:+.3f}]")
        else:
            print(f"  {method:20s} no motions found")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"description": DESCRIPTION, "length": args.length,
                   "n_action_groups": len(groups), "rows": rows}, f, indent=2)
    print(f"  written to {args.out}")


if __name__ == "__main__":
    main()
