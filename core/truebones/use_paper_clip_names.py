"""Give the processed Truebones clips the names the paper uses.

Each processed clip is named after its animal, its motion and a running number, such as
Alligator___Walk2_18. The conversion hands out the running numbers in its own order, which
is not the order of the paper's run, so a fresh conversion numbers the clips differently. The Truebones
evaluation refers to clips by the paper's names, so run this once after the conversion:

    python -m core.truebones.use_paper_clip_names

It renames the files in motions/, bvhs/ and animations/ of the processed dataset, using
paper_clip_names.json (animal and raw BVH file name -> the paper's names, one per 200-frame
piece in time order). Only the running numbers change; clip contents are untouched. Running it
again changes nothing. Clips it cannot match are listed at the end and left as they are.
"""
import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

from core.truebones.param_utils import DATASET_DIR, MOTION_DIR, BVHS_DIR, ANIMATIONS_DIR

ROOT = Path(__file__).resolve().parents[2]
MAPPING = Path(__file__).resolve().parent / "paper_clip_names.json"

# processed file name = clip name + suffix
SUFFIXES = {MOTION_DIR: ".npy", BVHS_DIR: ".bvh", ANIMATIONS_DIR: "_from_ric.mp4"}


def split_clip_name(name, animals):
    """Split a processed clip name into (animal, motion name, running number), or None."""
    m = re.match(r"^(.*)_(\d+)$", name)
    if m is None:
        return None
    stem, number = m.group(1), int(m.group(2))
    for animal in sorted(animals, key=len, reverse=True):
        if stem.startswith(animal + "_"):
            return animal, stem[len(animal) + 1:], number
    return None


def plan_renames(clip_names, mapping):
    """Pair each local clip with its paper name. Returns (renames, problems)."""
    problems = []
    groups = defaultdict(list)
    for name in clip_names:
        parts = split_clip_name(name, mapping)
        if parts is None:
            problems.append(f"{name}: animal not in the paper's dataset")
            continue
        animal, motion, number = parts
        # AnyTop names a clip after the raw file name up to its first dot
        raw_files = [f for f in mapping[animal] if f.split(".")[0] == motion]
        if len(raw_files) != 1:
            problems.append(f"{name}: no raw file {motion}.bvh for {animal} in the paper's dataset")
            continue
        groups[(animal, raw_files[0])].append((number, name))

    renames = {}
    for (animal, raw_file), local in sorted(groups.items()):
        paper = mapping[animal][raw_file]
        if len(local) != len(paper):
            problems.append(f"{animal}/{raw_file}: {len(local)} pieces here but {len(paper)} in the paper, "
                            f"left as they are ({', '.join(n for _, n in sorted(local))})")
            continue
        for (_, old), new in zip(sorted(local), paper):
            renames[old] = new

    matched = set(renames.values())
    for animal in sorted(mapping):
        for raw_file in sorted(mapping[animal]):
            for name in mapping[animal][raw_file]:
                if name not in matched:
                    problems.append(f"{name}: paper clip not found here")
    return renames, problems


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", default=str(ROOT / DATASET_DIR),
                        help="processed Truebones folder (holds motions/, bvhs/ and animations/)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    with open(MAPPING) as f:
        mapping = json.load(f)
    suffix = SUFFIXES[MOTION_DIR]
    clip_names = sorted(f[:-len(suffix)] for f in os.listdir(data_dir / MOTION_DIR) if f.endswith(suffix))
    renames, problems = plan_renames(clip_names, mapping)
    changed = {old: new for old, new in renames.items() if old != new}

    # Rename in two steps so that a new name never overwrites a clip that is still to be renamed.
    moves = []
    for folder, suffix in SUFFIXES.items():
        for old, new in changed.items():
            src = data_dir / folder / (old + suffix)
            if src.exists():
                moves.append((src, src.with_name(src.name + ".renaming"), data_dir / folder / (new + suffix)))
            else:
                problems.append(f"{folder}/{old + suffix}: file missing, nothing to rename")
    for src, tmp, _ in moves:
        os.rename(src, tmp)
    for _, tmp, dst in moves:
        os.rename(tmp, dst)

    print(f"{len(clip_names)} clips found in {data_dir / MOTION_DIR}")
    print(f"{len(renames)} matched to the paper's names, {len(changed)} of them renamed")
    if problems:
        print(f"{len(problems)} things could not be matched:")
        for p in problems:
            print("  " + p)
    else:
        print("Every clip now has the paper's name.")


if __name__ == "__main__":
    main()
