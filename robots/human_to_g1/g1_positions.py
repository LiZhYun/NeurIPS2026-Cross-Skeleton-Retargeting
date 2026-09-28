"""Read a Unitree G1 recording and work out where every body part is, frame by frame.

The collection stores the robot side as a table with one row per frame: where the pelvis
is, how it is turned, and the angle of each of the robot's twenty-nine joints. Feeding
those into the robot's own physical description gives a point for each body part, in
metres, with the up direction along Z. Thirty points are kept: the pelvis and the part
that each joint moves.

Joints are matched to the description by name, so the order of the columns does not
matter. The robot description is fetched by `fetch_g1_model.sh`; it is not part of this
repository.

The conversion step uses this module; it can also be run on a single file to see what the
file contains:
    python -m robots.human_to_g1.g1_positions <file.csv>
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = (ROOT / "external/holosoma/src/holosoma_retargeting/holosoma_retargeting"
                        "/models/g1/g1_29dof.xml")

# The order in which the three pelvis angles are applied, as used for the paper's numbers.
ROOT_ANGLE_ORDER = "XYZ"


def _read_table(path):
    """The column names and the numbers of a recording's table."""
    with open(path) as f:
        rows = list(csv.reader(f))
    return rows[0], np.asarray([[float(x) for x in r] for r in rows[1:]], np.float64)


class RobotModel:
    """The robot description, with the body parts this study reads out of it."""

    def __init__(self, xml=DEFAULT_MODEL):
        import mujoco

        self.mujoco = mujoco
        self.model_path = str(xml)
        self.model = mujoco.MjModel.from_xml_path(str(xml))
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.joints, joint_bodies = [], []
        for i in range(m.njnt):
            if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
                continue
            self.joints.append(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i))
            joint_bodies.append(int(m.jnt_bodyid[i]))
        pelvis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.body_ids = [pelvis] + joint_bodies
        self.body_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
                           for b in self.body_ids]

    def positions(self, header, table):
        """Turn the recording into body positions in metres, one row per frame."""
        mujoco = self.mujoco
        col = {name: k for k, name in enumerate(header)}
        base = table[:, [col["root_translateX"], col["root_translateY"],
                         col["root_translateZ"]]] / 100.0
        angles = table[:, [col["root_rotateX"], col["root_rotateY"], col["root_rotateZ"]]]
        xyzw = Rotation.from_euler(ROOT_ANGLE_ORDER, angles, degrees=True).as_quat()
        base_quat = xyzw[:, [3, 0, 1, 2]]
        joint_angles = np.stack([table[:, col[f"{j}_dof"]] for j in self.joints],
                                axis=1) * (np.pi / 180.0)

        out = np.empty((table.shape[0], len(self.body_ids), 3), np.float64)
        for t in range(table.shape[0]):
            self.data.qpos[:3] = base[t]
            self.data.qpos[3:7] = base_quat[t]
            self.data.qpos[7:] = joint_angles[t]
            mujoco.mj_forward(self.model, self.data)
            out[t] = self.data.xpos[self.body_ids]
        return out


_MODEL = None


def load_positions_m(path, xml=DEFAULT_MODEL):
    """Body positions in metres, together with the names of the points."""
    global _MODEL
    if _MODEL is None or _MODEL.model_path != str(xml):
        _MODEL = RobotModel(xml)
    header, table = _read_table(path)
    return _MODEL.positions(header, table), _MODEL.body_names


def main():
    ap = argparse.ArgumentParser(description="Report what one Unitree G1 recording contains.")
    ap.add_argument("csv", help="a G1 recording from the collection (.csv)")
    ap.add_argument("--g1_model", default=str(DEFAULT_MODEL),
                    help="the robot description (default: the one fetch_g1_model.sh fetches)")
    args = ap.parse_args()
    positions, names = load_positions_m(args.csv, args.g1_model)
    print(f"{Path(args.csv).name}: {positions.shape[0]} frames, "
          f"{positions.shape[1]} body points")
    print("points:", names)


if __name__ == "__main__":
    main()
