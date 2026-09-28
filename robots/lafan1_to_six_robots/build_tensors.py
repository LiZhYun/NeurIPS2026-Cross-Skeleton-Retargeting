"""Turn the retargeted recordings into joint positions, and write the clip list.

For each recording this cuts out the two windows chosen by choose_windows.py, works out
where the human's joints are and where each robot's body parts are, and saves one file per
clip per skeleton. Retargeting keeps frame numbering, so window frame 100 is the same
instant on the human and on every robot; that is what lets a model be trained and scored
against a true counterpart.

The human is placed in the same world the retargeting tool used, with the up direction
along Z and distances in metres. Robot points are the robot's floating base, the root body
that moves and turns freely in the world rather than being fixed to it, plus the part each
joint moves: the same rule the human-to-G1 study uses.

A recording that failed to retarget on any robot is dropped from the whole study, so every
robot is built from the same clips.

The clip list also records the two settings. Sparse keeps at most three performers per
action group and only the first window; dense keeps everything.

Runs in the robot environment (robots/environment.yml), on the processor, in about two
minutes. It reads the retargeting tool's robot descriptions from external/GMR:
    python -m robots.lafan1_to_six_robots.build_tensors
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

from robots.lafan1_to_six_robots.data import DEFAULT_DATA, ROBOTS, ROOT, clip_list_path

SEED = 42
# The retargeting tool reads the recordings in a world where Z points up and distances are
# in metres; the files themselves use centimetres and Y up.
TO_WORLD = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], np.float64)
# One robot names its feet after the leg segment they belong to rather than after an ankle.
FOOT_NAMES = {"pal_talos": ("leg_left_6_link", "leg_right_6_link")}


def is_foot(name, robot):
    """Whether a robot body is a foot, for the report's ground-contact check."""
    if robot in FOOT_NAMES:
        return name in FOOT_NAMES[robot]
    lowered = name.lower()
    return ("ankle" in lowered or "foot" in lowered or "toe" in lowered
            or re.search(r"(^|_)ank_", lowered) is not None)


class RobotShape:
    """One robot's description and the points read out of it."""

    def __init__(self, robot, xml_for_robot):
        import mujoco as mj

        self.mj = mj
        self.robot = robot
        self.model = mj.MjModel.from_xml_path(str(xml_for_robot[robot]))
        self.data = mj.MjData(self.model)
        m = self.model
        base, joint_bodies = [], []
        for i in range(m.njnt):
            if m.jnt_type[i] == mj.mjtJoint.mjJNT_FREE:
                base.append(int(m.jnt_bodyid[i]))
            else:
                joint_bodies.append(int(m.jnt_bodyid[i]))
        assert len(base) == 1, (robot, "expected one freely moving root body", base)
        self.body_ids = [base[0]] + joint_bodies
        self.body_names = [mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, b) for b in self.body_ids]
        self.n_points = len(self.body_ids)

        chosen = set(self.body_ids)
        self.segments = []
        row_of = {b: i for i, b in enumerate(self.body_ids)}
        for i, body in enumerate(self.body_ids[1:], start=1):
            parent = int(m.body_parentid[body])
            while parent not in chosen and parent != 0:
                parent = int(m.body_parentid[parent])
            if parent in chosen and row_of[parent] != i:
                self.segments.append((i, row_of[parent]))
        self.foot_rows = [i for i, n in enumerate(self.body_names) if is_foot(n, robot)]
        self.repeated_points = self.n_points - len(set(self.body_ids))

    def positions(self, pose):
        """Joint angles over time to body positions in metres."""
        assert pose.shape[1] == self.model.nq, (self.robot, pose.shape, self.model.nq)
        out = np.empty((pose.shape[0], self.n_points, 3), np.float64)
        for t in range(pose.shape[0]):
            self.data.qpos[:] = pose[t]
            self.mj.mj_forward(self.model, self.data)
            out[t] = self.data.xpos[self.body_ids]
        return out


def segment_variation(positions, segments):
    """Largest relative wobble in a robot segment's length; a rigid body gives nearly zero."""
    values = []
    for child, parent in segments:
        length = np.linalg.norm(positions[:, child] - positions[:, parent], axis=1)
        if length.mean() > 1e-6:
            values.append(float(length.std() / length.mean()))
    return max(values) if values else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=str(DEFAULT_DATA))
    ap.add_argument("--bvh_dir", default=None)
    ap.add_argument("--retargeted_dir", default=None)
    ap.add_argument("--gmr_dir", default=str(ROOT / "external/GMR"))
    args = ap.parse_args()

    if args.gmr_dir and args.gmr_dir not in sys.path:
        sys.path.insert(0, args.gmr_dir)
    from general_motion_retargeting.params import ROBOT_XML_DICT
    from general_motion_retargeting.utils.lafan_vendor import utils as lafan_utils
    from general_motion_retargeting.utils.lafan_vendor.extract import read_bvh

    data_dir = Path(args.data_dir)
    bvh_dir = Path(args.bvh_dir) if args.bvh_dir else data_dir / "lafan1_bvh"
    retargeted = (Path(args.retargeted_dir) if args.retargeted_dir
                  else data_dir / "retargeted")
    tensors = data_dir / "tensors"

    started = time.time()
    recordings = json.loads((data_dir / "takes.json").read_text())
    window = recordings["window_frames"]
    takes = sorted(recordings["takes"], key=lambda t: (t["routine"], t["performer"]))

    dropped = {}
    for take in takes:
        for robot in ROBOTS:
            path = retargeted / robot / f"{take['routine']}_s{take['performer']}.npz"
            if not path.exists():
                dropped.setdefault(take["source_file"], []).append(
                    f"{robot}: no retargeted result")
    usable = [t for t in takes if t["source_file"] not in dropped]

    shapes = {r: RobotShape(r, ROBOT_XML_DICT) for r in ROBOTS}
    (tensors / "human").mkdir(parents=True, exist_ok=True)
    for robot in ROBOTS:
        (tensors / robot).mkdir(parents=True, exist_ok=True)

    groups, clip_reports, human_meta = {}, [], None
    totals = {r: {"worst_wobble": 0.0, "lowest_foot": np.inf, "feet": [], "base": [],
                  "clips": 0, "bad": 0} for r in ROBOTS}
    human_totals = {"clips": 0, "bad": 0}

    for take in usable:
        routine, performer = take["routine"], take["performer"]
        animation = read_bvh(str(bvh_dir / take["source_file"]))
        _, human_world = lafan_utils.quat_fk(animation.quats, animation.pos,
                                             animation.parents)
        human = (human_world @ TO_WORLD.T) / 100.0
        joint_names = list(animation.bones)
        parents = [int(p) for p in animation.parents]
        if human_meta is None:
            human_meta = {"n_joints": len(joint_names), "joint_names": joint_names,
                          "parents": parents}
        assert human.shape[0] == take["n_frames"], (take["source_file"], human.shape)
        assert human.shape[1] == 22, (take["source_file"], human.shape)

        poses = {}
        for robot in ROBOTS:
            saved = np.load(retargeted / robot / f"{routine}_s{performer}.npz")
            pose = np.asarray(saved["qpos"], np.float64)
            assert pose.shape[0] == take["n_frames"], (robot, take["source_file"])
            poses[robot] = pose

        groups.setdefault(routine, {"name": routine, "action": take["action"],
                                    "routine_number": take["routine_number"], "clips": []})
        for k, start in enumerate(take["window_starts"]):
            clip_id = f"{routine}_s{performer}_w{k}"
            human_clip = human[start:start + window].astype(np.float32)
            np.save(tensors / "human" / f"{clip_id}.npy", human_clip)
            human_totals["clips"] += 1
            human_totals["bad"] += int(np.isnan(human_clip).any())
            report = {"clip_id": clip_id, "start_frame": start,
                      "human_frames": int(human_clip.shape[0]), "robots": {}}
            for robot in ROBOTS:
                shape = shapes[robot]
                robot_clip = shape.positions(poses[robot][start:start + window])
                np.save(tensors / robot / f"{clip_id}.npy", robot_clip.astype(np.float32))
                wobble = segment_variation(robot_clip, shape.segments)
                lowest = (float(robot_clip[:, shape.foot_rows, 2].min())
                          if shape.foot_rows else None)
                totals[robot]["worst_wobble"] = max(totals[robot]["worst_wobble"],
                                                    wobble or 0.0)
                if lowest is not None:
                    totals[robot]["lowest_foot"] = min(totals[robot]["lowest_foot"], lowest)
                    totals[robot]["feet"].append(lowest)
                totals[robot]["base"].append(float(robot_clip[:, 0, 2].mean()))
                totals[robot]["clips"] += 1
                totals[robot]["bad"] += int(np.isnan(robot_clip).any())
                report["robots"][robot] = {"frames": int(robot_clip.shape[0]),
                                           "segment_length_variation": wobble,
                                           "lowest_foot_m": lowest}
            clip_reports.append(report)
            groups[routine]["clips"].append({"clip_id": clip_id, "performer": performer,
                                             "source_file": take["source_file"],
                                             "start_frame": start})

    rng = np.random.default_rng(SEED)
    sparse, dense = {}, {}
    for name in sorted(groups):
        performers = sorted({c["performer"] for c in groups[name]["clips"]})
        dense[name] = [c["clip_id"] for c in groups[name]["clips"]]
        kept = (performers if len(performers) <= 3
                else sorted(rng.choice(performers, size=3, replace=False).tolist()))
        sparse[name] = [f"{name}_s{p}_w0" for p in kept]

    clip_list = {
        "meta": {
            "fps": 30, "window_frames": window,
            "robots": {r: {"n_points": shapes[r].n_points,
                           "body_names": shapes[r].body_names} for r in ROBOTS},
            "human": human_meta, "seed": SEED, "source": "LAFAN1",
            "windows": "a quarter and about two thirds of the way through each recording",
            "world": "Z up, metres, the world the retargeting tool uses",
            "robot_points": "the robot's freely moving root body plus the part each joint "
                            "moves",
            "sparse_rule": "per action group in name order: every performer if there are "
                           "at most three, otherwise three drawn with seed 42; first window "
                           "only",
        },
        "action_groups": [groups[g] for g in sorted(groups)],
        "settings": {"sparse": sparse, "dense": dense},
    }
    clip_list_path(data_dir).write_text(json.dumps(clip_list, indent=1))

    summaries = {}
    for robot in ROBOTS:
        path = retargeted / robot / "summary.json"
        if path.exists():
            summaries[robot] = json.loads(path.read_text())
    overall = {}
    for robot in ROBOTS:
        totals_r = totals[robot]
        overall[robot] = {
            "clips": totals_r["clips"], "clips_with_missing_values": totals_r["bad"],
            "segment_length_variation": totals_r["worst_wobble"],
            "lowest_foot_m": None if not totals_r["feet"] else totals_r["lowest_foot"],
            "mean_lowest_foot_m": None if not totals_r["feet"]
            else float(np.mean(totals_r["feet"])),
            "mean_base_height_m": float(np.mean(totals_r["base"])),
            "n_points": shapes[robot].n_points, "base_body": shapes[robot].body_names[0],
            "repeated_points": shapes[robot].repeated_points,
            "foot_bodies": [shapes[robot].body_names[i] for i in shapes[robot].foot_rows],
            "retarget_elapsed_seconds": summaries.get(robot, {}).get("elapsed_seconds"),
            "retarget_processor_seconds": summaries.get(robot, {}).get("processor_seconds"),
        }
    report = {
        "n_takes_total": len(takes), "n_takes_used": len(usable), "dropped_takes": dropped,
        "human": {"clips": human_totals["clips"],
                  "clips_with_missing_values": human_totals["bad"], "n_joints": 22,
                  "note": "metres, Z up, the world the retargeting tool uses"},
        "robots": overall,
        "note_on_feet": ("a foot's height is measured at the body's own origin, not at the "
                         "sole, so each robot carries a constant offset"),
        "build_seconds": round(time.time() - started, 1),
        "clips": clip_reports,
    }
    (data_dir / "conversion_report.json").write_text(json.dumps(report, indent=1))

    print(f"{len(groups)} action groups from {len(usable)} of {len(takes)} recordings")
    print(f"clips: {sum(len(v) for v in dense.values())} dense, "
          f"{sum(len(v) for v in sparse.values())} sparse "
          f"({human_totals['clips']} human clips)")
    for robot in ROBOTS:
        o = overall[robot]
        print(f"  {robot:18s} clips={o['clips']} points={o['n_points']} "
              f"segment wobble={o['segment_length_variation']:.1e} "
              f"lowest foot={o['lowest_foot_m']} base={o['mean_base_height_m']:.3f} m")
    if dropped:
        print(f"dropped {len(dropped)} recordings:")
        for name, why in dropped.items():
            print("  ", name, why)
    print(f"clip list: {clip_list_path(data_dir)}")
    print(f"report:    {data_dir / 'conversion_report.json'}")


if __name__ == "__main__":
    main()
