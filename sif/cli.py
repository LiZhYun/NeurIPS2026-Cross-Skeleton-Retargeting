"""Command line: score groups of motion clips listed in a JSON file.

    sif-score groups.json --length both --out results.json

groups.json:
    {"groups": [
        {"block": "Horse",
         "sources": ["horse/attack_1.npy", "horse/attack_2.npy", "horse/attack_3.npy"],
         "outputs": ["out/q1.npy", "out/q2.npy", "out/q3.npy"]},
        ...
    ]}
Each .npy file holds one clip of shape (frames, joints, 3 or more). Relative paths are read
from the folder that contains groups.json. "block" is optional; use the source skeleton when
several groups reuse the same source clips.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .score import score_groups

# The paper's names for the two ways of matching clip lengths
LENGTH_NAMES = {"raw": "raw", "fixed": "length-controlled"}


def _load(groups_file):
    spec = json.loads(Path(groups_file).read_text())
    root = Path(groups_file).resolve().parent
    groups = []
    for g in spec["groups"]:
        groups.append({"block": g.get("block"),
                       "sources": [np.load(root / p) for p in g["sources"]],
                       "outputs": [np.load(root / p) for p in g["outputs"]]})
    return groups


def _summary(result):
    return {"sif": result.sif, "ci95": list(result.ci), "p": result.p,
            "variation": result.variation, "n_groups": result.n_groups,
            "groups": [{"block": g.block, "sif": g.sif, "variation": g.variation,
                        "n_clips": g.n_clips} for g in result.groups]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("groups", help="JSON file listing the groups (format above)")
    ap.add_argument("--length", choices=["raw", "fixed", "both"], default="both",
                    help="raw: crop pairs to common length; fixed: resample to 64 frames "
                         "(the paper's length-controlled score)")
    ap.add_argument("--shuffles", type=int, default=10000,
                    help="number of random reshuffles for the p-value")
    ap.add_argument("--boot", type=int, default=10000,
                    help="number of resamples for the 95%% confidence interval")
    ap.add_argument("--seed", type=int, default=42, help="random seed")
    ap.add_argument("--out", help="write the results to this JSON file")
    args = ap.parse_args(argv)

    result = score_groups(_load(args.groups), args.length, n_shuffles=args.shuffles,
                          n_boot=args.boot, seed=args.seed)
    results = result if isinstance(result, dict) else {args.length: result}
    for mode, r in results.items():
        label = LENGTH_NAMES[mode]
        print(f"{label:>17}: SIF {r.sif:+.3f}  95% CI [{r.ci[0]:+.3f}, {r.ci[1]:+.3f}]  "
              f"p = {r.p:.4f}  variation {r.variation:.3f}  ({r.n_groups} groups)")
    if args.out:
        Path(args.out).write_text(json.dumps({m: _summary(r) for m, r in results.items()}, indent=2))


if __name__ == "__main__":
    main()
