"""Pull the chosen clips out of the two motion archives.

The collection ships its motions as large compressed archives. This study needs two of
them: `soma_uniform.tar.gz`, the human recordings as BVH files, and `g1.tar.gz`, the
matching Unitree G1 recordings as CSV tables. Reading a single clip out of such an archive
means reading the whole thing, so this step writes one list of wanted files per archive and
lets `tar` extract them all in a single pass. The two passes together take about half an
hour. The clips land under `<data_dir>/extracted/`, keeping the archives' own folder layout.

Needs clips.json in the data folder (see select_clips.py). Runs in either environment and
needs no GPU:
    python -m robots.human_to_g1.extract_clips --human_archive <soma_uniform.tar.gz> \
        --robot_archive <g1.tar.gz>
    python -m robots.human_to_g1.extract_clips --check
"""
import argparse
import subprocess
from pathlib import Path

from robots.human_to_g1.data import DEFAULT_DATA, load_clip_list


def wanted_members(clip_list):
    human = [c["human_path"] for a in clip_list["actions"] for c in a["clips"]]
    robot = [c["robot_path"] for a in clip_list["actions"] for c in a["clips"]]
    return human, robot


def extract(archive, members, into, list_path):
    """Extract exactly `members` from `archive` in one pass over the archive."""
    list_path.parent.mkdir(parents=True, exist_ok=True)
    list_path.write_text("\n".join(members) + "\n")
    into.mkdir(parents=True, exist_ok=True)
    print(f"reading {len(members)} clips out of {archive}")
    # tar applies -C only to the names that follow it, so it must come before -T.
    subprocess.run(["tar", "-xzf", str(archive), "-C", str(into), "-T", str(list_path)],
                   check=True)


def check(clip_list, into):
    """Report how many of the wanted clips are on disk."""
    human, robot = wanted_members(clip_list)
    ok = True
    for name, members in (("human", human), ("robot", robot)):
        missing = [m for m in members if not (into / m).exists()]
        print(f"{name}: {len(members) - len(missing)}/{len(members)} present")
        for m in missing[:10]:
            print("   missing", m)
        ok = ok and not missing
    print("complete" if ok else "incomplete")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human_archive", help="the human recordings, soma_uniform.tar.gz")
    ap.add_argument("--robot_archive", help="the robot recordings, g1.tar.gz")
    ap.add_argument("--data_dir", default=str(DEFAULT_DATA))
    ap.add_argument("--check", action="store_true",
                    help="only report which clips are already on disk")
    args = ap.parse_args()

    clip_list = load_clip_list(args.data_dir)
    into = Path(args.data_dir) / "extracted"
    lists = Path(args.data_dir) / "member_lists"

    if not args.check:
        if not (args.human_archive and args.robot_archive):
            ap.error("give both --human_archive and --robot_archive, or pass --check")
        human, robot = wanted_members(clip_list)
        extract(args.human_archive, human, into, lists / "human.txt")
        extract(args.robot_archive, robot, into, lists / "robot.txt")
    check(clip_list, into)


if __name__ == "__main__":
    main()
