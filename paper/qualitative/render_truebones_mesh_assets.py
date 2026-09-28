#!/usr/bin/env python3
"""Render Truebones animals as a mesh with a coloured skeleton on top.

Three poses of one clip are frozen as still meshes and placed side by side, with a
skeleton drawn over them. Bone colours follow what the bone is, such as a leg or the
spine, read from its name rather than from the shape of the skeleton tree. These
images appear in the first two figures of the paper.

Run it with Blender:

  blender -b --python paper/qualitative/render_truebones_mesh_assets.py -- --quick
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import bpy
from mathutils import Vector


ROOT = Path(__file__).resolve().parents[2]
TRUEBONES = ROOT / "datasets/truebones/zoo/Truebone_Z-OO"
OUT = ROOT / "paper" / "output" / "motion_mesh_assets"


SEMANTIC_COLORS = {
    "torso": (0.19, 0.39, 0.62, 1.0),
    "head": (0.18, 0.56, 0.45, 1.0),
    "tail": (0.88, 0.46, 0.12, 1.0),
    "left_fore": (0.86, 0.27, 0.28, 1.0),
    "right_fore": (0.95, 0.74, 0.22, 1.0),
    "left_hind": (0.45, 0.36, 0.65, 1.0),
    "right_hind": (0.35, 0.63, 0.32, 1.0),
    "wing": (0.30, 0.67, 0.78, 1.0),
    "appendage": (0.66, 0.45, 0.70, 1.0),
    "claw": (0.90, 0.48, 0.63, 1.0),
    "other": (0.42, 0.45, 0.49, 1.0),
}

MESH_ALPHA = 0.42
FRAME_ALPHAS = (0.34, 0.48, 0.64)


ASSETS = [
    {
        "name": "fig1_bird_attack2",
        "fbx": TRUEBONES / "Bird/BIRD-Attack2.fbx",
        "views": [("iso", -42, 22), ("side", -82, 15)],
    },
    {
        "name": "fig2_kingcobra_attack2",
        "fbx": TRUEBONES / "KingCobra/Cobra-Attack2.fbx",
        "views": [("iso", -64, 20), ("side", -90, 13)],
    },
    {
        "name": "fig2_kingcobra_bite",
        "fbx": TRUEBONES / "KingCobra/Cobra-Bite.fbx",
        "views": [("iso", -64, 20), ("side", -90, 13)],
    },
    {
        "name": "diverse_monkey_attack",
        "fbx": TRUEBONES / "Monkey/Monkey_B02.fbx",
        "views": [("iso", -52, 20)],
    },
    {
        "name": "diverse_fox_attack",
        "fbx": TRUEBONES / "Fox/FoxA_A02.fbx",
        "views": [("iso", -58, 18)],
    },
    {
        "name": "diverse_scorpion_stinger",
        "fbx": TRUEBONES / "Scorpion/scorpion-Stinger.fbx",
        "views": [("iso", -46, 22), ("top", -48, 42)],
    },
    {
        "name": "diverse_dragon_fly",
        "fbx": TRUEBONES / "Dragon/Wyvern-Fly.fbx",
        "views": [("iso", -54, 20), ("top", -52, 38)],
    },
    {
        "name": "diverse_crab_walk",
        "fbx": TRUEBONES / "Crab/Crab-Walk.fbx",
        "views": [("iso", -45, 24), ("top", -45, 48)],
    },
    {
        "name": "diverse_horse_hoofscrape",
        "fbx": TRUEBONES / "Horse/HorseALL-HoofScrape.fbx",
        "views": [("iso", -58, 18)],
    },
    {
        "name": "diverse_ant_jump",
        "fbx": TRUEBONES / "Ant/Ant-JumpForward.fbx",
        "views": [("iso", -54, 24), ("top", -54, 44)],
    },
    {
        "name": "diverse_ostrich_run",
        "fbx": TRUEBONES / "Ostrich/Ostrich-Run.fbx",
        "views": [("iso", -60, 18)],
    },
    {
        "name": "diverse_pteranodon_flyloop",
        "fbx": TRUEBONES / "Pteranodon/Pteranodon-FlyLoop.fbx",
        "views": [("iso", -58, 18), ("top", -58, 36)],
    },
    {
        "name": "diverse_tricera_attack3",
        "fbx": TRUEBONES / "Tricera/Tricera-Attack3.fbx",
        "views": [("iso", -58, 18)],
    },
    {
        "name": "diverse_hound_barking",
        "fbx": TRUEBONES / "Hound/Hound-Barking.fbx",
        "views": [("iso", -58, 18)],
    },
]

QUICK_NAMES = {
    "fig1_bird_attack2",
    "fig2_kingcobra_attack2",
    "diverse_dragon_fly",
    "diverse_scorpion_stinger",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Render a small representative subset.")
    parser.add_argument("--only", nargs="*", default=None, help="Optional asset names to render.")
    parser.add_argument("--size", type=int, default=1600, help="Square render resolution in pixels.")
    argv = []
    if "--" in __import__("sys").argv:
        argv = __import__("sys").argv[__import__("sys").argv.index("--") + 1 :]
    return parser.parse_args(argv)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for datablock in (
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.curves,
        bpy.data.cameras,
        bpy.data.lights,
        bpy.data.images,
    ):
        for item in list(datablock):
            if item.users == 0:
                datablock.remove(item)


def make_material(name: str, color, alpha: float = 1.0, emission: bool = False):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (color[0], color[1], color[2], alpha)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    if emission:
        nodes.clear()
        out = nodes.new(type="ShaderNodeOutputMaterial")
        em = nodes.new(type="ShaderNodeEmission")
        em.inputs["Color"].default_value = (color[0], color[1], color[2], alpha)
        em.inputs["Strength"].default_value = 0.9
        mat.node_tree.links.new(em.outputs["Emission"], out.inputs["Surface"])
    else:
        bsdf = nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Alpha"].default_value = alpha
            bsdf.inputs["Base Color"].default_value = (color[0], color[1], color[2], alpha)
    mat.blend_method = "BLEND"
    mat.use_screen_refraction = True
    mat.show_transparent_back = True
    return mat


def copy_mesh_material(src_mat, alpha: float, fallback_color):
    if src_mat is None:
        return make_material("mesh_neutral", fallback_color, alpha)
    mat = src_mat.copy()
    mat.diffuse_color = (
        src_mat.diffuse_color[0],
        src_mat.diffuse_color[1],
        src_mat.diffuse_color[2],
        alpha,
    )
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        if "Alpha" in bsdf.inputs:
            bsdf.inputs["Alpha"].default_value = alpha
        if "Base Color" in bsdf.inputs:
            c = bsdf.inputs["Base Color"].default_value
            bsdf.inputs["Base Color"].default_value = (c[0], c[1], c[2], alpha)
    mat.blend_method = "BLEND"
    mat.use_screen_refraction = True
    mat.show_transparent_back = True
    return mat


def semantic_label(name: str) -> str:
    s = name.lower().replace(".", "_").replace("-", "_")
    left = any(tok in s for tok in ("_l", "left", " l ", ".l"))
    right = any(tok in s for tok in ("_r", "right", " r ", ".r"))
    if any(tok in s for tok in ("stinger", "sting", "fang", "pincer", "claw", "talon", "horn", "antler")):
        return "claw"
    if any(tok in s for tok in ("head", "neck", "jaw", "mouth", "tongue", "beak", "skull", "face")):
        return "head"
    if "tail" in s:
        return "tail"
    if any(tok in s for tok in ("wing", "feather")):
        return "wing"
    if any(tok in s for tok in ("arm", "forearm", "hand", "finger", "clavicle", "shoulder")):
        if left:
            return "left_fore"
        if right:
            return "right_fore"
        return "appendage"
    if any(tok in s for tok in ("thigh", "calf", "leg", "foot", "toe", "hoof", "knee", "horse")):
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


def material_for_semantic(label: str):
    key = f"sem_{label}"
    mat = bpy.data.materials.get(key)
    if mat is None:
        mat = make_material(key, SEMANTIC_COLORS[label], 1.0, emission=True)
    return mat


def import_fbx(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    bpy.ops.import_scene.fbx(filepath=str(path), automatic_bone_orientation=False)
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not armatures or not meshes:
        raise RuntimeError(f"FBX did not import armature and mesh: {path}")
    return armatures[0], meshes


def object_bbox_world(obj) -> list[Vector]:
    return [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]


def bbox_center_and_size(objects) -> tuple[Vector, Vector, Vector, Vector]:
    pts = []
    for obj in objects:
        if hasattr(obj, "bound_box"):
            pts.extend(object_bbox_world(obj))
    if not pts:
        return Vector((0, 0, 0)), Vector((1, 1, 1)), Vector((-0.5, -0.5, 0)), Vector((0.5, 0.5, 1))
    minv = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    maxv = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    return (minv + maxv) * 0.5, maxv - minv, minv, maxv


def add_tube(name: str, p0: Vector, p1: Vector, radius: float, mat) -> None:
    if (p1 - p0).length < radius * 1.5:
        return
    curve = bpy.data.curves.new(name, type="CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = 2
    curve.bevel_depth = radius
    curve.bevel_resolution = 5
    spline = curve.splines.new(type="POLY")
    spline.points.add(1)
    spline.points[0].co = (p0.x, p0.y, p0.z, 1.0)
    spline.points[1].co = (p1.x, p1.y, p1.z, 1.0)
    obj = bpy.data.objects.new(name, curve)
    obj.data.materials.append(mat)
    bpy.context.collection.objects.link(obj)


def add_joint(name: str, p: Vector, radius: float, mat) -> None:
    bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8, radius=radius, location=p)
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(mat)


def sample_frames(scene, phases=(0.14, 0.50, 0.86)) -> list[int]:
    start, end = int(scene.frame_start), int(scene.frame_end)
    if end <= start:
        return [start, start, start]
    return [int(round(start + p * (end - start))) for p in phases]


def freeze_meshes_at_frame(scene, meshes, frame: int, alpha: float):
    scene.frame_set(frame)
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    static = []
    for mesh_obj in meshes:
        eval_obj = mesh_obj.evaluated_get(depsgraph)
        new_mesh = bpy.data.meshes.new_from_object(eval_obj, depsgraph=depsgraph)
        new_obj = bpy.data.objects.new(f"mesh_f{frame:04d}_{mesh_obj.name}", new_mesh)
        new_obj.matrix_world = mesh_obj.matrix_world.copy()
        new_obj.data.materials.clear()
        source_mats = list(mesh_obj.data.materials)
        if source_mats:
            for mat in source_mats:
                new_obj.data.materials.append(copy_mesh_material(mat, alpha, (0.78, 0.70, 0.60, 1.0)))
        else:
            new_obj.data.materials.append(make_material("mesh_neutral", (0.78, 0.70, 0.60, 1.0), alpha))
        bpy.context.collection.objects.link(new_obj)
        static.append(new_obj)
    return static


def make_skeleton_at_frame(scene, armature, frame: int, bone_radius: float, joint_radius: float):
    scene.frame_set(frame)
    bpy.context.view_layer.update()
    objects = []
    for pb in armature.pose.bones:
        label = semantic_label(pb.name)
        mat = material_for_semantic(label)
        p0 = armature.matrix_world @ pb.head
        p1 = armature.matrix_world @ pb.tail
        before = len(bpy.context.scene.objects)
        add_tube(f"bone_f{frame:04d}_{pb.name}", p0, p1, bone_radius, mat)
        if len(bpy.context.scene.objects) > before:
            objects.append(bpy.context.scene.objects[-1])
        add_joint(f"joint_f{frame:04d}_{pb.name}", p0, joint_radius, mat)
        objects.append(bpy.context.object)
    return objects


def translate_objects(objects, delta: Vector) -> None:
    for obj in objects:
        obj.location += delta


def add_checker_floor(minv: Vector, maxv: Vector) -> None:
    x0, x1 = minv.x - 0.25, maxv.x + 0.25
    y0, y1 = minv.y - 0.25, maxv.y + 0.25
    z = minv.z - 0.035
    nx, ny = 10, 6
    dx, dy = (x1 - x0) / nx, (y1 - y0) / ny
    mats = [
        make_material("floor_a", (0.93, 0.95, 0.97, 1.0), 0.74),
        make_material("floor_b", (0.88, 0.91, 0.95, 1.0), 0.74),
    ]
    for i in range(nx):
        for j in range(ny):
            mesh = bpy.data.meshes.new(f"floor_{i}_{j}")
            verts = [
                (x0 + i * dx, y0 + j * dy, z),
                (x0 + (i + 1) * dx, y0 + j * dy, z),
                (x0 + (i + 1) * dx, y0 + (j + 1) * dy, z),
                (x0 + i * dx, y0 + (j + 1) * dy, z),
            ]
            mesh.from_pydata(verts, [], [(0, 1, 2, 3)])
            mesh.update()
            obj = bpy.data.objects.new(f"floor_{i}_{j}", mesh)
            obj.data.materials.append(mats[(i + j) % 2])
            bpy.context.collection.objects.link(obj)


def look_at(obj, target: Vector) -> None:
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def add_camera_and_light(center: Vector, minv: Vector, maxv: Vector, azim: float, elev: float, size: int) -> None:
    span = max((maxv - minv).x, (maxv - minv).y, (maxv - minv).z, 1.0)
    r = span * 2.9
    az = math.radians(azim)
    el = math.radians(elev)
    loc = Vector((center.x + r * math.cos(el) * math.cos(az), center.y + r * math.cos(el) * math.sin(az), center.z + r * math.sin(el)))
    bpy.ops.object.camera_add(location=loc)
    cam = bpy.context.object
    look_at(cam, center)
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = span * 1.18
    bpy.context.scene.camera = cam

    bpy.ops.object.light_add(type="AREA", location=(center.x - span, center.y - span, center.z + span * 2.2))
    key = bpy.context.object
    key.name = "Key_Area"
    key.data.energy = 430
    key.data.size = span * 3.0

    bpy.context.scene.render.resolution_x = size
    bpy.context.scene.render.resolution_y = int(size * 0.62)


def render_asset(asset: dict, view_name: str, azim: float, elev: float, size: int) -> Path:
    clear_scene()
    armature, meshes = import_fbx(asset["fbx"])
    scene = bpy.context.scene
    frames = sample_frames(scene)

    # Get a representative scale from the middle frame before freezing.
    mid_static = freeze_meshes_at_frame(scene, meshes, frames[1], MESH_ALPHA)
    _, rep_size, _, _ = bbox_center_and_size(mid_static)
    for obj in mid_static:
        bpy.data.objects.remove(obj, do_unlink=True)
    scale = max(rep_size.x, rep_size.y, rep_size.z, 0.5)
    spacing = scale * 1.18
    bone_radius = scale * 0.010
    joint_radius = scale * 0.030

    frame_groups = []
    for idx, frame in enumerate(frames):
        frame_meshes = freeze_meshes_at_frame(scene, meshes, frame, FRAME_ALPHAS[idx])
        frame_skel = make_skeleton_at_frame(scene, armature, frame, bone_radius, joint_radius)
        group = frame_meshes + frame_skel
        center, _, minv, _ = bbox_center_and_size(group)
        target = Vector(((idx - 1) * spacing, 0.0, -minv.z))
        delta = Vector((target.x - center.x, target.y - center.y, target.z))
        translate_objects(group, delta)
        frame_groups.extend(group)

    for obj in meshes + [armature]:
        obj.hide_render = True
        obj.hide_viewport = True

    bpy.context.view_layer.update()
    center, _, minv, maxv = bbox_center_and_size(frame_groups)
    add_checker_floor(minv, maxv)
    add_camera_and_light(center, minv, maxv, azim, elev, size)

    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.eevee.taa_render_samples = 64
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "Medium High Contrast"
    scene.world = bpy.data.worlds.new("World")
    scene.world.color = (1, 1, 1)
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.compression = 12

    out = OUT / f"{asset['name']}_{view_name}.png"
    scene.render.filepath = str(out)
    bpy.ops.render.render(write_still=True)
    print(f"saved {out}")
    return out


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.only:
        selected = [a for a in ASSETS if a["name"] in set(args.only)]
    elif args.quick:
        selected = [a for a in ASSETS if a["name"] in QUICK_NAMES]
    else:
        selected = ASSETS
    for asset in selected:
        for view_name, azim, elev in asset["views"]:
            render_asset(asset, view_name, azim, elev, args.size)


if __name__ == "__main__":
    main()
