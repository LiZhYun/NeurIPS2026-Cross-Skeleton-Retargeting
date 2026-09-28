"""Score one method's outputs on the Truebones benchmark.

    python -m benchmark.score --outputs outputs/my_method/set1891 --set 1891 --out my_method.json

--outputs is a folder with one file per query, query_0000.npy, query_0001.npy, ..., each holding
the motion the method produced for that query on the target skeleton, shape (frames, joints, 3 or
more). The query list is in benchmark/sets/truebones_{37,49,1891}.json. Each query asks for one source
clip (skel_a, src_fname) to be retargeted to one target skeleton (skel_b). Queries that share a
source skeleton, target skeleton and action form a group; SIF is computed per group and averaged.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from sif import score_groups

SETS = Path(__file__).resolve().parent / "sets"
DEFAULT_MOTIONS = "dataset/truebones/zoo/truebones_processed/motions"
# The paper's names for the two ways of matching clip lengths
LENGTH_NAMES = {"raw": "raw", "fixed": "length-controlled"}


def load_groups(set_name, outputs, motions):
    """Build the groups of a benchmark set, exactly as in the paper.

    Queries are grouped by (source skeleton, target skeleton, action) and taken in sorted group
    order. A group keeps one query per distinct source clip, at most six. Queries whose output
    file is missing are left out; a group needs at least two remaining clips (a group reduced to
    two clips has a single distance and scores 0).
    """
    spec = json.loads((SETS / f"truebones_{set_name}.json").read_text())
    by_group = defaultdict(list)
    for q in spec["queries"]:
        by_group[(q["skel_a"], q["skel_b"], q["src_action"])].append(q)
    groups = []
    for (skel_a, _, _), queries in sorted(by_group.items()):
        unique = {}
        for q in queries:
            unique.setdefault(q["src_fname"], q)
        sources, outs = [], []
        for q in list(unique.values())[:6]:
            src = Path(motions) / q["src_fname"]
            out = Path(outputs) / f"query_{q['query_id']:04d}.npy"
            if src.exists() and out.exists():
                sources.append(np.load(src).astype(np.float32))
                outs.append(np.load(out).astype(np.float32))
        if len(sources) >= 2:
            groups.append({"block": skel_a, "sources": sources, "outputs": outs})
    return groups


def main(argv=None):
    ap = argparse.ArgumentParser(description="Score a method's outputs on the Truebones benchmark.")
    ap.add_argument("--outputs", required=True, help="folder with query_XXXX.npy files")
    ap.add_argument("--set", choices=["37", "49", "1891"], default="1891")
    ap.add_argument("--motions", default=DEFAULT_MOTIONS, help="processed Truebones motions folder")
    ap.add_argument("--length", choices=["raw", "fixed", "both"], default="both",
                    help="raw: crop pairs to common length; fixed: resample to 64 frames "
                         "(the paper's length-controlled score)")
    ap.add_argument("--out", help="write the results to this JSON file")
    args = ap.parse_args(argv)

    groups = load_groups(args.set, args.outputs, args.motions)
    result = score_groups(groups, args.length, min_clips=2)
    results = result if isinstance(result, dict) else {args.length: result}
    for mode, r in results.items():
        label = LENGTH_NAMES[mode]
        print(f"{label:>17}: SIF {r.sif:+.3f}  95% CI [{r.ci[0]:+.3f}, {r.ci[1]:+.3f}]  "
              f"p = {r.p:.4f}  variation {r.variation:.3f}  ({r.n_groups} groups)")
    if args.out:
        summary = {mode: {"sif": r.sif, "ci95": list(r.ci), "p": r.p, "variation": r.variation,
                          "n_groups": r.n_groups} for mode, r in results.items()}
        Path(args.out).write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
