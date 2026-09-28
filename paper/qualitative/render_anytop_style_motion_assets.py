#!/usr/bin/env python3
"""Render the skeleton images used in the first two figures of the paper.

Each image shows one pose of a motion as a coloured skeleton, with a small picture of
the animal's mesh in the corner for scale and recognition. The skeleton comes either
from a real Truebones clip or from a motion a method produced; the mesh picture comes
from the matching Truebones model file and carries no skeleton on top.

Run it with Blender:

  blender -b --python paper/qualitative/render_anytop_style_motion_assets.py -- --help
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "outputs"     # outputs/<method>/set49/query_XXXX.npy, as generate writes them
PROC = ROOT / "dataset/truebones/zoo/truebones_processed"
MOTION_DIR = PROC / "motions"
COND_FILE = PROC / "cond.npy"
FBX_ROOT = ROOT / "datasets/truebones/zoo/Truebone_Z-OO"
OUT_ROOT = ROOT / "paper/output/motion_mesh_assets"


SCENE_FLOOR_X = (-6.40, 6.40)
SCENE_FLOOR_Y = (-6.10, 6.65)
SCENE_SKELETON_Y = -0.30
SCENE_CAMERA_LOC = Vector((0.0, -6.35, 2.12))
SCENE_CAMERA_TARGET = Vector((0.0, 0.72, 1.02))


PALETTE = {
    "torso": (0.05, 0.43, 0.82, 1.0),
    "head": (0.10, 0.10, 0.10, 1.0),
    "tail": (0.10, 0.78, 0.22, 1.0),
    "left_fore": (1.00, 0.22, 0.26, 1.0),
    "right_fore": (1.00, 0.76, 0.05, 1.0),
    "left_hind": (0.58, 0.23, 0.95, 1.0),
    "right_hind": (0.18, 0.82, 0.24, 1.0),
    "wing": (0.00, 0.78, 0.96, 1.0),
    "appendage": (0.70, 0.30, 0.95, 1.0),
    "claw": (1.00, 0.22, 0.55, 1.0),
    "other": (0.12, 0.50, 0.92, 1.0),
}


FIG1_SPECS = [
    {
        "name": "source_bird_attack2",
        "skel": "Bird",
        "motion": MOTION_DIR / "Bird___Attack2_110.npy",
        "mesh": FBX_ROOT / "Bird/BIRD-Attack2.fbx",
        "phase": 0.55,
    },
    {
        "name": "source_bird_attack4_q077",
        "skel": "Bird",
        "motion": MOTION_DIR / "Bird___Attack4_107.npy",
        "mesh": FBX_ROOT / "Bird/BIRD-Attack4.fbx",
        "phase": 0.55,
    },
    {
        "name": "map_ace_i_kingcobra",
        "skel": "KingCobra",
        "motion": OUTPUT_ROOT / "ace_i" / "set49" / "query_0076.npy",
        "mesh": FBX_ROOT / "KingCobra/Cobra-Attack2.fbx",
        "phase": 0.55,
    },
    {
        "name": "map_ace_i_kingcobra_q077",
        "skel": "KingCobra",
        "motion": OUTPUT_ROOT / "ace_i" / "set49" / "query_0077.npy",
        "mesh": FBX_ROOT / "KingCobra/Cobra-Attack2.fbx",
        "phase": 0.55,
    },
    {
        "name": "map_moref_t_kingcobra",
        "skel": "KingCobra",
        "motion": OUTPUT_ROOT / "moreflow_t" / "set49" / "query_0076.npy",
        "mesh": FBX_ROOT / "KingCobra/Cobra-Attack2.fbx",
        "phase": 0.55,
    },
    {
        "name": "map_alflow_kingcobra",
        "skel": "KingCobra",
        "motion": OUTPUT_ROOT / "al_flow" / "set49" / "query_0076.npy",
        "mesh": FBX_ROOT / "KingCobra/Cobra-Attack2.fbx",
        "phase": 0.55,
    },
    {
        "name": "map_alflow_kingcobra_q077",
        "skel": "KingCobra",
        "motion": OUTPUT_ROOT / "al_flow" / "set49" / "query_0077.npy",
        "mesh": FBX_ROOT / "KingCobra/Cobra-Attack2.fbx",
        "phase": 0.55,
    },
    {
        "name": "map_anchor_kingcobra",
        "skel": "KingCobra",
        "motion": OUTPUT_ROOT / "anchor" / "set49" / "query_0076.npy",
        "mesh": FBX_ROOT / "KingCobra/Cobra-Attack2.fbx",
        "phase": 0.55,
    },
    {
        "name": "source_spiderg_attack4",
        "skel": "SpiderG",
        "motion": MOTION_DIR / "SpiderG___Attack4_937.npy",
        "mesh": FBX_ROOT / "SpiderG/Spider-Attack4.fbx",
        "phase": 0.52,
    },
    {
        "name": "map_ace_i_tricera_q004",
        "skel": "Tricera",
        "motion": OUTPUT_ROOT / "ace_i" / "set49" / "query_0004.npy",
        "mesh": FBX_ROOT / "Tricera/Tricera-Attack4.fbx",
        "phase": 0.52,
    },
    {
        "name": "map_moref_t_tricera_q004",
        "skel": "Tricera",
        "motion": OUTPUT_ROOT / "moreflow_t" / "set49" / "query_0004.npy",
        "mesh": FBX_ROOT / "Tricera/Tricera-Attack4.fbx",
        "phase": 0.52,
    },
    {
        "name": "map_anchor_tricera_q004",
        "skel": "Tricera",
        "motion": OUTPUT_ROOT / "anchor" / "set49" / "query_0004.npy",
        "mesh": FBX_ROOT / "Tricera/Tricera-Attack4.fbx",
        "phase": 0.52,
    },
    {
        "name": "source_coyote_attack2",
        "skel": "Coyote",
        "motion": MOTION_DIR / "Coyote___Attack2_229.npy",
        "mesh": FBX_ROOT / "Coyote/Coyote-Attack2.fbx",
        "phase": 0.52,
    },
    {
        "name": "map_ace_i_ostrich_q119",
        "skel": "Ostrich",
        "motion": OUTPUT_ROOT / "ace_i" / "set49" / "query_0119.npy",
        "mesh": FBX_ROOT / "Ostrich/Ostrich-Attack2.fbx",
        "phase": 0.52,
    },
    {
        "name": "map_moref_t_ostrich_q119",
        "skel": "Ostrich",
        "motion": OUTPUT_ROOT / "moreflow_t" / "set49" / "query_0119.npy",
        "mesh": FBX_ROOT / "Ostrich/Ostrich-Attack2.fbx",
        "phase": 0.52,
    },
    {
        "name": "source_dragon_attack3",
        "skel": "Dragon",
        "motion": MOTION_DIR / "Dragon___Attack3_289.npy",
        "mesh": FBX_ROOT / "Dragon/Wyvern-Attack3.fbx",
        "phase": 0.50,
    },
    {
        "name": "map_ace_i_jaguar_q147",
        "skel": "Jaguar",
        "motion": OUTPUT_ROOT / "ace_i" / "set49" / "query_0147.npy",
        "mesh": FBX_ROOT / "Jaguar/Jaguar-Attack2.fbx",
        "phase": 0.50,
    },
    {
        "name": "map_moref_t_jaguar_q147",
        "skel": "Jaguar",
        "motion": OUTPUT_ROOT / "moreflow_t" / "set49" / "query_0147.npy",
        "mesh": FBX_ROOT / "Jaguar/Jaguar-Attack2.fbx",
        "phase": 0.50,
    },
]


FIG1_COYOTE_OSTRICH_SPECS = [
    {
        "name": "source_coyote_attack1_q117",
        "skel": "Coyote",
        "motion": MOTION_DIR / "Coyote___Attack1_231.npy",
        "mesh": FBX_ROOT / "Coyote/Coyote-Attack1.fbx",
        "phase": 0.52,
    },
    {
        "name": "source_coyote_attack2_q119",
        "skel": "Coyote",
        "motion": MOTION_DIR / "Coyote___Attack2_229.npy",
        "mesh": FBX_ROOT / "Coyote/Coyote-Attack2.fbx",
        "phase": 0.52,
    },
    {
        "name": "map_ace_i_ostrich_q117",
        "skel": "Ostrich",
        "motion": OUTPUT_ROOT / "ace_i" / "set49" / "query_0117.npy",
        "mesh": FBX_ROOT / "Ostrich/Ostrich-Attack2.fbx",
        "phase": 0.52,
    },
    {
        "name": "map_ace_i_ostrich_q119",
        "skel": "Ostrich",
        "motion": OUTPUT_ROOT / "ace_i" / "set49" / "query_0119.npy",
        "mesh": FBX_ROOT / "Ostrich/Ostrich-Attack2.fbx",
        "phase": 0.52,
    },
    {
        "name": "map_alflow_ostrich_q119",
        "skel": "Ostrich",
        "motion": OUTPUT_ROOT / "al_flow" / "set49" / "query_0119.npy",
        "mesh": FBX_ROOT / "Ostrich/Ostrich-Attack2.fbx",
        "phase": 0.52,
    },
]


FIG2_GROUPS = [
    (
        "kingcobra",
        "KingCobra",
        [
            ("target_attack2", "KingCobra___Attack2_503.npy", FBX_ROOT / "KingCobra/Cobra-Attack2.fbx", 0.55),
            ("target_attack", "KingCobra___Attack_504.npy", FBX_ROOT / "KingCobra/Cobra-Attack.fbx", 0.55),
            ("target_bite", "KingCobra___Bite_506.npy", FBX_ROOT / "KingCobra/Cobra-Bite.fbx", 0.55),
            ("target_circlebite", "KingCobra___CircleBite_507.npy", FBX_ROOT / "KingCobra/Cobra-CircleBite.fbx", 0.55),
        ],
    ),
    (
        "tricera",
        "Tricera",
        [
            ("target_attack", "Tricera___Attack_1040.npy", FBX_ROOT / "Tricera/Tricera-Attack.fbx", 0.52),
            ("target_attack2", "Tricera___Attack2_1039.npy", FBX_ROOT / "Tricera/Tricera-Attack2.fbx", 0.52),
            ("target_attack3", "Tricera___Attack3_1033.npy", FBX_ROOT / "Tricera/Tricera-Attack3.fbx", 0.52),
            ("target_attack4", "Tricera___Attack4_1036.npy", FBX_ROOT / "Tricera/Tricera-Attack4.fbx", 0.52),
        ],
    ),
    (
        "ostrich",
        "Ostrich",
        [
            ("target_attack", "Ostrich___Attack_590.npy", FBX_ROOT / "Ostrich/Ostrich-Attack.fbx", 0.52),
            ("target_attack2", "Ostrich___Attack2_588.npy", FBX_ROOT / "Ostrich/Ostrich-Attack2.fbx", 0.52),
            ("target_attack3", "Ostrich___Attack3_581.npy", FBX_ROOT / "Ostrich/Ostrich-Attack3.fbx", 0.52),
            ("target_attack4", "Ostrich___Attack4_587.npy", FBX_ROOT / "Ostrich/Ostrich-Attack4.fbx", 0.52),
        ],
    ),
]


DIVERSE_SPECS = [
    ("diverse_monkey_attack", "Monkey", MOTION_DIR / "Monkey___B2Attack_574.npy", FBX_ROOT / "Monkey/Monkey_B02.fbx", 0.52),
    ("diverse_fox_attack", "Fox", MOTION_DIR / "Fox_-_Attack2_357.npy", FBX_ROOT / "Fox/FoxA_A02.fbx", 0.52),
    ("diverse_scorpion_stinger", "Scorpion", MOTION_DIR / "Scorpion___Stinger_837.npy", FBX_ROOT / "Scorpion/scorpion-Stinger.fbx", 0.58),
    ("diverse_dragon_fly", "Dragon", MOTION_DIR / "Dragon___Fly_295.npy", FBX_ROOT / "Dragon/Wyvern-Fly.fbx", 0.48),
    ("diverse_crab_walk", "Crab", MOTION_DIR / "Crab___Walk_235.npy", FBX_ROOT / "Crab/Crab-Walk.fbx", 0.52),
    ("diverse_horse_hoofscrape", "Horse", MOTION_DIR / "Horse___HoofScrape_454.npy", FBX_ROOT / "Horse/HorseALL-HoofScrape.fbx", 0.54),
    ("diverse_ant_jump", "Ant", MOTION_DIR / "Ant___JumpForward_58.npy", FBX_ROOT / "Ant/Ant-JumpForward.fbx", 0.50),
    ("diverse_ostrich_run", "Ostrich", MOTION_DIR / "Ostrich___Run_591.npy", FBX_ROOT / "Ostrich/Ostrich-Run.fbx", 0.52),
    ("diverse_pteranodon_flyloop", "Pteranodon", MOTION_DIR / "Pteranodon___FlyLoop_657.npy", FBX_ROOT / "Pteranodon/Pteranodon-FlyLoop.fbx", 0.50),
    ("diverse_tricera_attack3", "Tricera", MOTION_DIR / "Tricera___Attack3_1033.npy", FBX_ROOT / "Tricera/Tricera-Attack3.fbx", 0.50),
    ("diverse_hound_barking", "Hound", MOTION_DIR / "Hound___Barking_470.npy", FBX_ROOT / "Hound/Hound-Barking.fbx", 0.52),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true",
                        help="render only a few images of Figure 1 and of the extra animals")
    parser.add_argument("--size", type=int, default=1500, help="image width in pixels")
    parser.add_argument("--set", choices=["all", "fig1", "fig1-coyote-ostrich", "fig2", "diverse"],
                        default="all",
                        help="which images to render: Figure 1, the Coyote to Ostrich images of "
                             "Figure 1, Figure 2, or the extra animals")
    parser.add_argument("--phase-only", action="store_true",
                        help="Figure 1: render only the extra images at other points in time")
    parser.add_argument("--base-only", action="store_true",
                        help="Figure 1: skip the extra images at other points in time")
    parser.add_argument("--fig1-start", type=int, default=0,
                        help="Figure 1: index of the first image to render")
    parser.add_argument("--fig1-end", type=int, default=None,
                        help="Figure 1: index one past the last image to render")
    parser.add_argument("--fig2-start", type=int, default=0,
                        help="Figure 2: index of the first skeleton group to render")
    parser.add_argument("--fig2-end", type=int, default=None,
                        help="Figure 2: index one past the last skeleton group to render")
    parser.add_argument("--skeleton-yaw-offset-deg", type=float, default=25.0,
                        help="extra turn of the skeleton about the vertical axis, in degrees")
    parser.add_argument("--mesh-yaw-correction-deg", type=float, default=None,
                        help="turn of the mesh picture about the vertical axis, in degrees; "
                             "by default it is matched to the skeleton")
    parser.add_argument("--suffix", default="", help="text appended to every file name")
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(argv)


def clear_scene() -> None:
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for collection in (bpy.data.meshes, bpy.data.materials, bpy.data.curves, bpy.data.cameras, bpy.data.lights):
        for item in list(collection):
            if item.users == 0:
                collection.remove(item)


def make_mat(name: str, color, alpha=1.0, emission=False):
    existing = bpy.data.materials.get(name)
    if existing:
        return existing
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (color[0], color[1], color[2], alpha)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    if emission:
        nodes.clear()
        out = nodes.new(type="ShaderNodeOutputMaterial")
        em = nodes.new(type="ShaderNodeEmission")
        em.inputs["Color"].default_value = (color[0], color[1], color[2], alpha)
        em.inputs["Strength"].default_value = 0.65
        mat.node_tree.links.new(em.outputs["Emission"], out.inputs["Surface"])
    else:
        bsdf = nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = (color[0], color[1], color[2], alpha)
            bsdf.inputs["Alpha"].default_value = alpha
            if "Roughness" in bsdf.inputs:
                bsdf.inputs["Roughness"].default_value = 0.68
    mat.blend_method = "BLEND" if alpha < 1.0 else "OPAQUE"
    mat.use_screen_refraction = alpha < 1.0
    return mat


def semantic_label(name: str, skel: str) -> str:
    s = name.lower().replace("-", "_")
    left = any(tok in s for tok in ("_l_", "_l", "left", ".l"))
    right = any(tok in s for tok in ("_r_", "_r", "right", ".r"))
    winged = skel.lower() in {"bird", "pteranodon", "dragon", "bat", "giantbee"}
    if any(tok in s for tok in ("claw", "talon", "horn", "antler", "fang", "pincer", "stinger", "sting")):
        return "claw"
    if any(tok in s for tok in ("head", "neck", "jaw", "mouth", "tongue", "beak", "skull", "face")):
        return "head"
    if "tail" in s:
        return "tail"
    if winged and any(tok in s for tok in ("wing", "upperarm", "forearm", "hand", "finger")):
        return "wing"
    if any(tok in s for tok in ("arm", "forearm", "hand", "finger", "clavicle", "shoulder")):
        if left:
            return "left_fore"
        if right:
            return "right_fore"
        return "appendage"
    if any(tok in s for tok in ("thigh", "calf", "leg", "foot", "toe", "hoof", "knee")):
        if left:
            return "left_hind"
        if right:
            return "right_hind"
        return "appendage"
    if any(tok in s for tok in ("spine", "pelvis", "hips", "hip", "chest", "body", "cog", "root")):
        return "torso"
    if any(tok in s for tok in ("feeler", "antenna")):
        return "appendage"
    return "other"


def color_for_joint(joint_name: str, skel: str):
    return PALETTE[semantic_label(joint_name, skel)]


def load_positions(path: Path, n_joints: int) -> np.ndarray:
    arr = np.load(path).astype(np.float32)
    return arr[:, :n_joints, :3].copy()


def resample(pos: np.ndarray, n=72) -> np.ndarray:
    src = np.linspace(0.0, 1.0, pos.shape[0])
    tgt = np.linspace(0.0, 1.0, n)
    out = np.empty((n, pos.shape[1], 3), dtype=np.float32)
    for j in range(pos.shape[1]):
        for d in range(3):
            out[:, j, d] = np.interp(tgt, src, pos[:, j, d])
    return out


def representative_phases(pos: np.ndarray, n: int = 4) -> tuple[float, ...]:
    """Pick temporally spread frames from high-motion parts of the clip."""
    t = pos.shape[0]
    if t <= n:
        return tuple(np.linspace(0.08, 0.92, n))
    velocity = np.linalg.norm(np.diff(pos, axis=0), axis=2).mean(axis=1)
    if not np.isfinite(velocity).all() or float(np.max(velocity)) < 1e-8:
        return tuple(np.linspace(0.12, 0.88, n))
    win = max(3, int(round(t * 0.06)))
    if win % 2 == 0:
        win += 1
    kernel = np.ones(win, dtype=np.float32) / win
    score = np.convolve(velocity, kernel, mode="same")
    phases = []
    margin = max(2, int(round(t * 0.05)))
    lo_all, hi_all = margin, max(margin + 1, t - margin - 1)
    for k in range(n):
        lo = int(round(lo_all + (hi_all - lo_all) * k / n))
        hi = int(round(lo_all + (hi_all - lo_all) * (k + 1) / n))
        hi = max(lo + 1, hi)
        seg = score[min(lo, len(score) - 1) : min(hi, len(score))]
        if seg.size:
            frame = min(lo + int(np.argmax(seg)) + 1, t - 1)
        else:
            frame = int(round((k + 0.5) * (t - 1) / n))
        phases.append(frame / (t - 1))
    phases = np.maximum.accumulate(np.asarray(phases, dtype=np.float32))
    return tuple(float(np.clip(p, 0.04, 0.96)) for p in phases)


def _indices_matching(names, tokens):
    out = []
    for i, name in enumerate(names):
        s = name.lower().replace("-", "_")
        if any(tok in s for tok in tokens):
            out.append(i)
    return out


def _clip_indices(indices, n_joints):
    return [i for i in indices if 0 <= i < n_joints]


def orientation_angle(poses: list[np.ndarray], names) -> float:
    n_joints = poses[0].shape[0]
    head = _clip_indices(
        _indices_matching(names, ("head", "neck", "jaw", "mouth", "beak", "skull", "eye", "face")),
        n_joints,
    )
    tail = _clip_indices(_indices_matching(names, ("tail", "tai", "stinger", "sting")), n_joints)
    body = _clip_indices(_indices_matching(names, ("hips", "pelvis", "spine", "body", "cog", "root")), n_joints)
    vectors = []
    for pose in poses:
        if head and tail:
            v = pose[head].mean(axis=0) - pose[tail].mean(axis=0)
        elif head and body:
            v = pose[head].mean(axis=0) - pose[body].mean(axis=0)
        elif tail and body:
            v = pose[body].mean(axis=0) - pose[tail].mean(axis=0)
        else:
            xy = pose[:, :2] - pose[:, :2].mean(axis=0, keepdims=True)
            _, _, vt = np.linalg.svd(xy, full_matrices=False)
            v = np.array([vt[0, 0], vt[0, 1], 0.0], dtype=np.float32)
        vxy = np.array([v[0], v[1]], dtype=np.float32)
        if np.linalg.norm(vxy) > 1e-5:
            vectors.append(vxy / np.linalg.norm(vxy))
    if not vectors:
        return 0.0
    forward = np.mean(np.stack(vectors), axis=0)
    if np.linalg.norm(forward) < 1e-5:
        return 0.0
    forward = forward / np.linalg.norm(forward)
    target = np.array([0.24, -1.0], dtype=np.float32)
    target = target / np.linalg.norm(target)
    return float(math.atan2(target[1], target[0]) - math.atan2(forward[1], forward[0]))


def rotate_pose_xy(pose: np.ndarray, angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    out = pose.copy()
    out[:, :2] = out[:, :2] @ rot.T
    return out


def is_airborne_spec(spec: dict) -> bool:
    text = " ".join(str(spec.get(k, "")) for k in ("name", "skel", "motion", "mesh")).lower()
    if any(tok in text for tok in ("fly", "flight", "jump", "pteranodon", "bird")):
        return True
    return False


def transform_poses(
    pos: np.ndarray,
    phases: tuple[float, ...],
    names,
    airborne: bool,
    skeleton_yaw_offset: float = 0.0,
) -> tuple[list[np.ndarray], float]:
    frames = [int(round(np.clip(p, 0, 1) * (pos.shape[0] - 1))) for p in phases]
    poses = []
    for frame in frames:
        pose = pos[frame].copy()
        # Processed motions use y-up; Blender uses z-up.
        pose = np.stack([pose[:, 0], pose[:, 2], pose[:, 1]], axis=1)
        center = pose.mean(axis=0)
        pose[:, 0] -= center[0]
        pose[:, 1] -= center[1]
        poses.append(pose)
    all_points = np.concatenate(poses, axis=0)
    scale = max(float(np.ptp(all_points, axis=0).max()), 1e-5)
    out = []
    for pose in poses:
        p = pose / scale * 3.85
        p[:, 2] -= p[:, 2].min()
        out.append(p)
    display_yaw = orientation_angle(out, names) + skeleton_yaw_offset
    out = [rotate_pose_xy(p, display_yaw) for p in out]
    z_offset = 0.34 if airborne else 0.0
    for p in out:
        p[:, 2] -= p[:, 2].min()
        p[:, 2] += z_offset
    return out, display_yaw


def add_tube(name: str, p0: Vector, p1: Vector, radius: float, mat):
    if (p1 - p0).length < radius:
        return None
    curve = bpy.data.curves.new(name, "CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = radius
    curve.bevel_resolution = 6
    spline = curve.splines.new("POLY")
    spline.points.add(1)
    spline.points[0].co = (p0.x, p0.y, p0.z, 1)
    spline.points[1].co = (p1.x, p1.y, p1.z, 1)
    obj = bpy.data.objects.new(name, curve)
    obj.data.materials.append(mat)
    bpy.context.collection.objects.link(obj)
    return obj


def add_joint(name: str, p: Vector, radius: float, mat) -> None:
    bpy.ops.mesh.primitive_uv_sphere_add(segments=18, ring_count=9, radius=radius, location=p)
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(mat)


def add_skeleton(pose: np.ndarray, parents, names, skel: str, offset=Vector((0, 0, 0))) -> list:
    objects = []
    joint_mat = make_mat("joint_black", (0.03, 0.03, 0.035, 1), 1.0)
    shadow_mat = make_mat("skeleton_shadow", (0.12, 0.13, 0.14, 1), 0.20)
    shadow_offset = Vector((0.045, -0.035, 0.0))
    for j, p in enumerate(parents):
        if p >= 0 and p != j and p < len(parents):
            mat = make_mat(f"sem_{semantic_label(names[j], skel)}", color_for_joint(names[j], skel), 1.0)
            p0 = Vector(tuple(pose[p])) + offset
            p1 = Vector(tuple(pose[j])) + offset
            sp0 = Vector((p0.x, p0.y, 0.015)) + shadow_offset
            sp1 = Vector((p1.x, p1.y, 0.015)) + shadow_offset
            shadow = add_tube(f"shadow_{j}", sp0, sp1, 0.028, shadow_mat)
            if shadow is not None:
                objects.append(shadow)
            bone = add_tube(f"bone_{j}", p0, p1, 0.028, mat)
            if bone is not None:
                objects.append(bone)
    for j, xyz in enumerate(pose):
        add_joint(f"joint_{j}", Vector(tuple(xyz)) + offset, 0.046, joint_mat)
        objects.append(bpy.context.object)
    return objects


def motion_strip_spacing(footprint: float) -> float:
    return float(np.clip(footprint * 1.20, 2.10, 3.15))


def fit_poses_to_strip_width(poses: list[np.ndarray], max_width: float = 7.10, max_height: float = 2.70) -> list[np.ndarray]:
    if len(poses) <= 1:
        return poses
    footprint = max(float(np.ptp(pose[:, 0])) for pose in poses)
    height = max(float(np.ptp(pose[:, 2])) for pose in poses)
    spacing = motion_strip_spacing(footprint)
    total_width = footprint + spacing * (len(poses) - 1)
    if total_width <= max_width and height <= max_height:
        return poses
    factor = min(max_width / max(total_width, 1e-5), max_height / max(height, 1e-5))
    fitted = []
    for pose in poses:
        p = pose.copy()
        zmin = float(p[:, 2].min())
        p[:, :2] *= factor
        p[:, 2] = (p[:, 2] - zmin) * factor + zmin
        fitted.append(p)
    return fitted


def add_motion_skeletons(poses: list[np.ndarray], parents, names, skel: str) -> list:
    poses = fit_poses_to_strip_width(poses)
    if len(poses) == 1:
        return add_skeleton(poses[0], parents, names, skel)
    objects = []
    footprint = max(float(np.ptp(pose[:, 0])) for pose in poses)
    spacing = motion_strip_spacing(footprint)
    start = -0.5 * spacing * (len(poses) - 1)
    for i, pose in enumerate(poses):
        offset = Vector((start + i * spacing, SCENE_SKELETON_Y, 0.0))
        objects.extend(add_skeleton(pose, parents, names, skel, offset=offset))
    return objects


def bbox(objects) -> tuple[Vector, Vector, Vector]:
    pts = []
    for obj in objects:
        if hasattr(obj, "bound_box"):
            pts.extend([obj.matrix_world @ Vector(c) for c in obj.bound_box])
    if not pts:
        return Vector((0, 0, 0)), Vector((1, 1, 1)), Vector((-0.5, -0.5, 0))
    minv = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    maxv = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    return (minv + maxv) * 0.5, maxv, minv


def copy_mesh_mat(mat):
    if mat is None:
        return make_mat("mesh_thumb_neutral", (0.80, 0.75, 0.66, 1), 1.0)
    new = mat.copy()
    new.diffuse_color = (mat.diffuse_color[0], mat.diffuse_color[1], mat.diffuse_color[2], 1.0)
    new.use_nodes = True
    bsdf = new.node_tree.nodes.get("Principled BSDF")
    if bsdf and "Alpha" in bsdf.inputs:
        bsdf.inputs["Alpha"].default_value = 1.0
    new.blend_method = "OPAQUE"
    return new


def rotate_objects_about(objects, center: Vector, angle: float) -> None:
    if abs(angle) < 1e-5:
        return
    transform = Matrix.Translation(center) @ Matrix.Rotation(angle, 4, "Z") @ Matrix.Translation(-center)
    for obj in objects:
        obj.matrix_world = transform @ obj.matrix_world


def freeze_mesh_thumbnail(fbx: Path, phase: float, target_center: Vector, target_width: float, zrot: float = 0.0) -> list:
    before = set(bpy.data.objects)
    bpy.ops.import_scene.fbx(filepath=str(fbx), automatic_bone_orientation=False)
    imported = [obj for obj in bpy.data.objects if obj not in before]
    meshes = [obj for obj in imported if obj.type == "MESH"]
    scene = bpy.context.scene
    frame = int(round(scene.frame_start + np.clip(phase, 0, 1) * (scene.frame_end - scene.frame_start)))
    scene.frame_set(frame)
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    frozen = []
    for mesh in meshes:
        eval_obj = mesh.evaluated_get(depsgraph)
        new_mesh = bpy.data.meshes.new_from_object(eval_obj, depsgraph=depsgraph)
        new_obj = bpy.data.objects.new(f"thumb_{mesh.name}", new_mesh)
        new_obj.matrix_world = mesh.matrix_world.copy()
        source_mats = list(mesh.data.materials)
        if source_mats:
            for mat in source_mats:
                new_obj.data.materials.append(copy_mesh_mat(mat))
        else:
            new_obj.data.materials.append(make_mat("mesh_thumb_neutral", (0.80, 0.75, 0.66, 1), 1.0))
        bpy.context.collection.objects.link(new_obj)
        frozen.append(new_obj)
    for obj in imported:
        bpy.data.objects.remove(obj, do_unlink=True)
    if not frozen:
        return []
    center, maxv, minv = bbox(frozen)
    width = max((maxv - minv).x, (maxv - minv).y, (maxv - minv).z, 1e-5)
    scale = target_width / width
    for obj in frozen:
        obj.scale *= scale
    bpy.context.view_layer.update()
    center, _, _ = bbox(frozen)
    rotate_objects_about(frozen, center, zrot)
    bpy.context.view_layer.update()
    center, _, _ = bbox(frozen)
    delta = target_center - center
    for obj in frozen:
        obj.location += delta
    return frozen


def blend_color(c0, c1, t: float):
    t = max(0.0, min(1.0, t))
    return tuple(c0[i] * (1.0 - t) + c1[i] * t for i in range(3)) + (1.0,)


def add_floor() -> None:
    x0, x1 = SCENE_FLOOR_X
    y0, y1 = SCENE_FLOOR_Y
    z = -0.006
    nx, ny = 10, 8
    near_a = (0.78, 0.81, 0.84)
    near_b = (0.91, 0.92, 0.93)
    white = (1.0, 1.0, 1.0)
    for i in range(nx):
        for j in range(ny):
            row = j / max(ny - 1, 1)
            fade = row**1.85
            base = near_a if (i + j) % 2 == 0 else near_b
            mat = make_mat(f"floor_{i}_{j}_fade", blend_color(base, white, fade), 1.0, emission=True)
            mesh = bpy.data.meshes.new(f"floor_{i}_{j}")
            xa, xb = x0 + (x1 - x0) * i / nx, x0 + (x1 - x0) * (i + 1) / nx
            ya, yb = y0 + (y1 - y0) * j / ny, y0 + (y1 - y0) * (j + 1) / ny
            mesh.from_pydata([(xa, ya, z), (xb, ya, z), (xb, yb, z), (xa, yb, z)], [], [(0, 1, 2, 3)])
            mesh.update()
            obj = bpy.data.objects.new(f"floor_{i}_{j}", mesh)
            obj.data.materials.append(mat)
            bpy.context.collection.objects.link(obj)


def compose_floor_band(render_path: Path) -> None:
    helper = Path(__file__).resolve().parent / "composite_anytop_floor.py"
    subprocess.run([sys.executable, str(helper), str(render_path)], check=True)


def look_at(obj, target: Vector) -> None:
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def setup_scene(out_path: Path, size: int):
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.eevee.taa_render_samples = 48
    scene.render.resolution_x = size
    scene.render.resolution_y = int(size * 0.44)
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "Medium High Contrast"
    scene.world = bpy.data.worlds.new("World")
    scene.world.color = (1, 1, 1)
    center = SCENE_CAMERA_TARGET.copy()
    bpy.ops.object.camera_add(location=SCENE_CAMERA_LOC)
    cam = bpy.context.object
    look_at(cam, center)
    cam.data.type = "PERSP"
    cam.data.lens = 34
    cam.data.sensor_fit = "VERTICAL"
    cam.data.sensor_width = 32
    scene.camera = cam
    bpy.ops.object.light_add(type="AREA", location=(-2.4, -3.7, 5.0))
    light = bpy.context.object
    light.name = "Key_Area"
    light.data.energy = 650
    light.data.size = 5.5
    scene.render.filepath = str(out_path)
    aspect = scene.render.resolution_x / scene.render.resolution_y
    quat = cam.matrix_world.to_quaternion()
    right = quat @ Vector((1, 0, 0))
    up = quat @ Vector((0, 1, 0))
    forward = quat @ Vector((0, 0, -1))
    depth = (center - cam.location).length
    vspan = 2.0 * depth * math.tan(cam.data.angle_y * 0.5)
    return {
        "center": center,
        "vspan": vspan,
        "hspan": vspan * aspect,
        "right": right,
        "up": up,
        "forward": forward,
        "camera": cam,
    }


def render_panel(
    spec: dict,
    out_dir: Path,
    cond: dict,
    size: int,
    phase_variant: float | None = None,
    view_suffix: str = "",
    skeleton_yaw_offset: float = 0.0,
    mesh_yaw_correction: float | None = None,
) -> Path:
    clear_scene()
    out_dir.mkdir(parents=True, exist_ok=True)
    skel = spec["skel"]
    parents = cond[skel]["parents"]
    names = cond[skel]["joints_names"]
    phase = spec["phase"] if phase_variant is None else phase_variant
    pos = load_positions(spec["motion"], len(parents))
    if phase_variant is None:
        phases = tuple(spec.get("phases", representative_phases(pos)))
    else:
        delta = 0.11
        phases = tuple(float(np.clip(phase + (i - 1.5) * delta, 0.05, 0.95)) for i in range(4))
    airborne = bool(spec.get("airborne", is_airborne_spec(spec)))
    poses, zrot = transform_poses(pos, phases, names, airborne, skeleton_yaw_offset=skeleton_yaw_offset)
    add_motion_skeletons(poses, parents, names, skel)
    name = spec["name"]
    if phase_variant is not None:
        name += f"_p{int(round(phase_variant * 100)):02d}"
    if view_suffix:
        name += f"_{view_suffix}"
    out_path = out_dir / f"{name}.png"
    view = setup_scene(out_path, size)
    # Place the mesh-only thumbnail in camera space so it remains in the
    # upper-left corner for side and isometric views.
    thumb_center = (
        view["camera"].location
        + view["forward"] * ((view["center"] - view["camera"].location).length * 0.78)
        - view["right"] * (view["hspan"] * 0.30)
        + view["up"] * (view["vspan"] * 0.28)
    )
    if mesh_yaw_correction is None:
        mesh_yaw_correction = float(spec.get("mesh_yaw_correction", math.pi - math.radians(38) - skeleton_yaw_offset))
    freeze_mesh_thumbnail(spec["mesh"], phase, thumb_center, view["vspan"] * 0.180, zrot=zrot + mesh_yaw_correction)
    bpy.ops.render.render(write_still=True)
    compose_floor_band(out_path)
    print(f"saved {out_path}")
    return out_path


def render_fig2_prototype(
    cond: dict,
    size: int,
    skeleton_yaw_offset: float = 0.0,
    mesh_yaw_correction: float | None = None,
    suffix: str = "",
    group_start: int = 0,
    group_end: int | None = None,
) -> None:
    out_dir = OUT_ROOT / "figure2_prototype"
    end = len(FIG2_GROUPS) if group_end is None else group_end
    for group, skel, entries in FIG2_GROUPS[group_start:end]:
        parents = cond[skel]["parents"]
        clips = [resample(load_positions(MOTION_DIR / fname, len(parents)), 72) for _, fname, _, _ in entries]
        proto = np.mean(np.stack(clips, axis=0), axis=0)
        proto_path = OUT_ROOT / f"_tmp_{group}_prototype.npy"
        np.save(proto_path, proto)
        specs = []
        for name, fname, mesh, phase in entries:
            specs.append(
                {
                    "name": f"{group}_{name}",
                    "skel": skel,
                    "motion": MOTION_DIR / fname,
                    "mesh": mesh,
                    "phase": phase,
                }
            )
        specs.append(
            {
                "name": f"{group}_prototype_mean",
                "skel": skel,
                "motion": proto_path,
                "mesh": entries[0][2],
                "phase": entries[0][3],
            }
        )
        for spec in specs:
            render_panel(
                spec,
                out_dir,
                cond,
                size,
                view_suffix=suffix,
                skeleton_yaw_offset=skeleton_yaw_offset,
                mesh_yaw_correction=mesh_yaw_correction,
            )
        if proto_path.exists():
            proto_path.unlink()


def write_index() -> None:
    index = OUT_ROOT / "INDEX.md"
    index.write_text(
        """# Skeleton images for Figures 1 and 2

These images are the pieces the first two figures of the paper are put together from.

- `figure1_mapping/` shows one source clip and what several methods made of it.
- `figure2_prototype/` shows several clips of one skeleton and action, and their average.
- `diverse_pool/` shows further animals and actions.

Each image shows the motion as a skeleton coloured by body part, with a small picture of
the animal's mesh in the upper-left corner.
""",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    cond = np.load(COND_FILE, allow_pickle=True).item()
    mesh_yaw_correction = None
    if args.mesh_yaw_correction_deg is not None:
        mesh_yaw_correction = math.radians(args.mesh_yaw_correction_deg)
    if args.set in {"all", "fig1"}:
        out = OUT_ROOT / "figure1_mapping"
        base_specs = FIG1_SPECS[:2] if args.quick else FIG1_SPECS
        end = len(base_specs) if args.fig1_end is None else args.fig1_end
        specs = base_specs[args.fig1_start : end]
        if not args.phase_only:
            for spec in specs:
                render_panel(
                    spec,
                    out,
                    cond,
                    args.size,
                    view_suffix=args.suffix,
                    skeleton_yaw_offset=math.radians(args.skeleton_yaw_offset_deg),
                    mesh_yaw_correction=mesh_yaw_correction,
                )
        if not args.quick and not args.base_only:
            # Extra phase variants for manual selection without changing the mapping meaning.
            for spec in specs:
                for phase in (0.32, 0.74):
                    render_panel(
                        spec,
                        out,
                        cond,
                        args.size,
                        phase_variant=phase,
                        view_suffix=args.suffix,
                        skeleton_yaw_offset=math.radians(args.skeleton_yaw_offset_deg),
                        mesh_yaw_correction=mesh_yaw_correction,
                    )
    if args.set in {"all", "fig1-coyote-ostrich"}:
        out = OUT_ROOT / "figure1_mapping"
        for spec in FIG1_COYOTE_OSTRICH_SPECS:
            render_panel(
                spec,
                out,
                cond,
                args.size,
                view_suffix=args.suffix,
                skeleton_yaw_offset=math.radians(args.skeleton_yaw_offset_deg),
                mesh_yaw_correction=mesh_yaw_correction,
            )
    if args.set in {"all", "fig2"}:
        render_fig2_prototype(
            cond,
            args.size,
            skeleton_yaw_offset=math.radians(args.skeleton_yaw_offset_deg),
            mesh_yaw_correction=mesh_yaw_correction,
            suffix=args.suffix,
            group_start=args.fig2_start,
            group_end=args.fig2_end,
        )
    if args.set in {"all", "diverse"}:
        out = OUT_ROOT / "diverse_pool"
        specs = DIVERSE_SPECS[:3] if args.quick else DIVERSE_SPECS
        for name, skel, motion, mesh, phase in specs:
            render_panel(
                {"name": name, "skel": skel, "motion": motion, "mesh": mesh, "phase": phase},
                out,
                cond,
                args.size,
                view_suffix=args.suffix,
                skeleton_yaw_offset=math.radians(args.skeleton_yaw_offset_deg),
                mesh_yaw_correction=mesh_yaw_correction,
            )
    write_index()


if __name__ == "__main__":
    main()
