"""Drawing helpers for the robot figure.

Everything here draws the robots' real shapes with MuJoCo, off screen, so the pictures
show the machines rather than stick figures:

  * a still that shows several moments of one motion side by side in a single scene, so
    the floor, the lights and the shadows are shared and the earlier moments are genuinely
    faded rather than pasted on afterwards;
  * a video of the same motion at 30 frames per second;
  * a simple figure built from capsules for the human performer, since the recordings give
    joint positions and no body shape;
  * a way to turn predicted body positions back into robot joint angles, by asking the
    robot to reach those positions with its joints inside their real limits.

A robot pose here is MuJoCo's list of joint values for one frame: the position and turn of
the robot's root body (the floating base, which moves freely in the world rather than being
fixed to it), then the angle of every joint. A MuJoCo scene is made of shapes it calls
geoms; the stills copy the shapes of earlier moments into the scene of the last one.
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco as mj
CHECKER_LIGHT = (1.0, 1.0, 1.0)
CHECKER_DARK = (0.930, 0.936, 0.950)


def _set_look(spec, ortho_height=None, offsamples=8, floor_repeat=1.6, floor_size=None):
    """Give every robot the same look.

    A white sky, a soft light checked floor, one key light that casts shadows, heavy
    smoothing of edges, and, when `ortho_height` is given, a camera without perspective
    that shows that many metres from bottom to top.
    """
    v = spec.visual
    v.quality.offsamples = offsamples
    v.quality.shadowsize = 8192
    v.global_.offwidth = 2400
    v.global_.offheight = 1400
    v.map.znear = 0.02
    v.map.zfar = 60.0
    v.map.shadowclip = 2.5
    v.map.shadowscale = 1.0
    v.headlight.ambient = [0.50, 0.50, 0.52]
    v.headlight.diffuse = [0.50, 0.50, 0.51]
    v.headlight.specular = [0.16, 0.16, 0.16]
    v.global_.orthographic = int(ortho_height is not None)
    if ortho_height is not None:
        v.global_.fovy = float(ortho_height)

    for tex in spec.textures:
        if tex.type == mj.mjtTexture.mjTEXTURE_SKYBOX:
            tex.builtin = int(mj.mjtBuiltin.mjBUILTIN_GRADIENT)
            tex.rgb1 = [1.0, 1.0, 1.0]
            tex.rgb2 = [1.0, 1.0, 1.0]
            tex.mark = int(mj.mjtMark.mjMARK_NONE)

    floor_materials = set()
    for g in spec.geoms:
        if g.type != mj.mjtGeom.mjGEOM_PLANE:
            continue
        if g.material:
            floor_materials.add(g.material)
        if floor_size is not None:
            g.size = [float(floor_size[0]), float(floor_size[1]), 0.05]
    for mat in spec.materials:
        if mat.name not in floor_materials:
            continue
        mat.texrepeat = [floor_repeat, floor_repeat]
        mat.reflectance = 0.0
        mat.rgba = [1.0, 1.0, 1.0, 1.0]
        for texture_name in mat.textures:
            if not texture_name:
                continue
            for tex in spec.textures:
                if tex.name == texture_name:
                    tex.builtin = int(mj.mjtBuiltin.mjBUILTIN_CHECKER)
                    tex.rgb1 = list(CHECKER_LIGHT)
                    tex.rgb2 = list(CHECKER_DARK)
                    tex.mark = int(mj.mjtMark.mjMARK_NONE)
                    tex.width = tex.height = 512

    for light in spec.lights:
        light.active = 0
    spec.worldbody.add_light(pos=[-1.8, -1.5, 5.0], dir=[1.7, 1.4, -4.8], castshadow=1,
                             type=int(mj.mjtLightType.mjLIGHT_DIRECTIONAL),
                             diffuse=[0.42, 0.42, 0.43], specular=[0.10, 0.10, 0.10],
                             ambient=[0.0, 0.0, 0.0])
    spec.stat.extent = 3.0
    return spec


def load_model(xml_path, ortho_height=None, offsamples=8, floor_repeat=1.6,
               floor_size=None):
    """Read a robot description and give it the figure's look."""
    spec = mj.MjSpec.from_file(str(xml_path))
    _set_look(spec, ortho_height, offsamples, floor_repeat, floor_size)
    return spec.compile()


def body_rows(model):
    """The points the study reads off a robot, in order: the root body, then the body
    each joint moves. This is the order the saved joint positions use."""
    free, hinge = [], []
    for j in range(model.njnt):
        if model.jnt_type[j] == mj.mjtJoint.mjJNT_FREE:
            free.append(int(model.jnt_bodyid[j]))
        else:
            hinge.append(int(model.jnt_bodyid[j]))
    ids = [free[0]] + hinge
    names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, b) for b in ids]
    return ids, names


def body_positions(model, data, poses, body_ids):
    """Where the listed bodies are in every frame, for a sequence of robot poses."""
    out = np.empty((len(poses), len(body_ids), 3))
    for t in range(len(poses)):
        data.qpos[:] = poses[t]
        mj.mj_forward(model, data)
        out[t] = data.xpos[body_ids]
    return out


def lowest_point(model, data, poses):
    """The lowest height any part of the robot's shape may reach, frame by frame.

    Each shape is treated as the sphere that encloses it, so the value errs on the low side.
    """
    shapes = np.array([g for g in range(model.ngeom)
                       if model.geom_type[g] != mj.mjtGeom.mjGEOM_PLANE])
    radius = model.geom_rbound[shapes]
    z = np.empty(len(poses))
    for t in range(len(poses)):
        data.qpos[:] = poses[t]
        mj.mj_forward(model, data)
        z[t] = float((data.geom_xpos[shapes, 2] - radius).min())
    return z


def tint_rgb(rgb, tint, strength=0.78):
    """Recolour a shape while keeping its own light and dark structure."""
    lum = float(np.clip(np.mean(rgb[:3]), 0.0, 1.0))
    shaded = np.asarray(tint, float) * (0.45 + 0.95 * lum)
    return np.clip((1.0 - strength) * np.asarray(rgb[:3]) + strength * shaded, 0.0, 1.0)


GEOM_FIELDS = ("type", "dataid", "objtype", "objid", "category", "texcoord", "segid",
               "emission", "specular", "shininess", "reflectance", "camdist",
               "transparent")


def _copy_geom(dst, src):
    """Copy one drawn shape into a free place in another scene."""
    for f in GEOM_FIELDS:
        setattr(dst, f, getattr(src, f))
    dst.size[:] = src.size
    dst.pos[:] = src.pos
    dst.mat[:] = src.mat
    dst.rgba[:] = src.rgba
    dst.matid = src.matid
    dst.modelrbound = src.modelrbound
    dst.label = ''          # places are reused; an old label would be drawn


class Scene:
    """Draws a robot off screen, and can put several poses of it in one scene."""

    def __init__(self, model, width=1700, height=480, max_poses=8):
        self.model = model
        self.data = mj.MjData(model)
        self.w, self.h = width, height
        self.renderer = mj.Renderer(model, height=height, width=width,
                                    max_geom=model.ngeom * (max_poses + 1) + 400)
        self.opt = mj.MjvOption()
        self.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = False
        self.scratch = mj.MjvScene(model, maxgeom=model.ngeom + 200)

    @staticmethod
    def camera(lookat, distance, azimuth, elevation):
        """A camera looking at `lookat` from `distance` metres, angles in degrees."""
        cam = mj.MjvCamera()
        cam.type = mj.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = lookat
        cam.distance = distance
        cam.azimuth = azimuth
        cam.elevation = elevation
        return cam

    def render_single(self, pose, cam, tint=None, shadow=True):
        """One picture of one pose."""
        self.data.qpos[:] = pose
        mj.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera=cam, scene_option=self.opt)
        sc = self.renderer.scene
        sc.flags[mj.mjtRndFlag.mjRND_SHADOW] = int(shadow)
        if tint is not None:
            for i in range(sc.ngeom):
                g = sc.geoms[i]
                if int(g.category) == int(mj.mjtCatBit.mjCAT_DYNAMIC):
                    g.rgba[:3] = tint_rgb(np.array(g.rgba), tint)
                    g.matid = -1
        return self.renderer.render().copy()

    def render_multi(self, poses, cam, tint=None, alphas=None, fades=None, shadow=True):
        """One picture holding several poses.

        The last pose builds the scene, with its floor and lights; the robot's shapes in
        the earlier poses are copied into it. `alphas` makes those earlier poses
        see-through; `fades` instead washes them towards white, which keeps the shadows
        clean.
        """
        n = len(poses)
        if alphas is None and fades is None:
            fades = np.linspace(0.55, 0.0, n)
        if alphas is None:
            alphas = np.ones(n)
        if fades is None:
            fades = np.zeros(n)
        self.data.qpos[:] = poses[-1]
        mj.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera=cam, scene_option=self.opt)
        sc = self.renderer.scene
        sc.flags[mj.mjtRndFlag.mjRND_SHADOW] = int(shadow)

        def paint(geom, alpha, fade):
            rgb = np.array(geom.rgba[:3])
            if tint is not None:
                rgb = tint_rgb(np.array(geom.rgba), tint)
            if fade > 1e-6:
                rgb = rgb * (1.0 - fade) + fade
            if tint is not None or fade > 1e-6:
                geom.rgba[:3] = rgb
                geom.matid = -1
            geom.rgba[3] = alpha
            geom.transparent = 1 if alpha < 0.999 else 0

        for i in range(sc.ngeom):
            g = sc.geoms[i]
            if int(g.category) == int(mj.mjtCatBit.mjCAT_DYNAMIC):
                paint(g, float(alphas[-1]), float(fades[-1]))

        for k in range(n - 1):
            self.data.qpos[:] = poses[k]
            mj.mj_forward(self.model, self.data)
            mj.mjv_updateScene(self.model, self.data, self.opt, None, cam,
                               int(mj.mjtCatBit.mjCAT_DYNAMIC), self.scratch)
            for i in range(self.scratch.ngeom):
                src = self.scratch.geoms[i]
                if int(src.category) != int(mj.mjtCatBit.mjCAT_DYNAMIC):
                    continue
                if sc.ngeom >= sc.maxgeom:
                    break
                dst = sc.geoms[sc.ngeom]
                _copy_geom(dst, src)
                paint(dst, float(alphas[k]), float(fades[k]))
                sc.ngeom = sc.ngeom + 1
        return self.renderer.render().copy()

    def close(self):
        self.renderer.close()


def spread_offsets(n, spacing, azimuth_deg):
    """Even sideways offsets, in metres on the floor, along the camera's left-to-right
    direction, so the first moment lands on the left of the picture and time reads left
    to right."""
    a = np.deg2rad(azimuth_deg)
    right = np.array([np.sin(a), -np.cos(a)])
    k = np.arange(n) - (n - 1) / 2.0
    return k[:, None] * spacing * right[None, :]


def stage_poses(poses, frames, offsets, ground_dz=0.0):
    """Take the given frames of a motion and stand them side by side.

    Each chosen pose loses its own position on the floor and is placed at its sideways
    offset instead; `ground_dz` raises or lowers every pose by the same amount.
    """
    out = []
    for i, f in enumerate(frames):
        q = np.array(poses[f], float)
        q[0:2] = offsets[i]
        q[2] += ground_dz
        out.append(q)
    return out


HUMAN_RADIUS = {
    'spine': 0.075, 'leg': 0.058, 'arm': 0.044, 'foot': 0.036, 'neck': 0.045,
}


def _human_radius(child_name):
    """How thick to draw the bone that ends at the named joint."""
    n = child_name.lower()
    if 'toe' in n or 'foot' in n:
        return HUMAN_RADIUS['foot']
    if 'leg' in n:
        return HUMAN_RADIUS['leg']
    if 'hand' in n or 'forearm' in n or 'arm' in n or 'shoulder' in n:
        return HUMAN_RADIUS['arm']
    if 'neck' in n or 'head' in n:
        return HUMAN_RADIUS['neck']
    return HUMAN_RADIUS['spine']


def build_mannequin(joint_names, parents, bone_len, rgba=(0.56, 0.60, 0.66, 1.0),
                    ortho_height=None, offsamples=8, floor_repeat=1.6, floor_size=None):
    """A MuJoCo model of the human: one capsule per bone and a sphere for the head.

    Each capsule is a body that is placed directly, frame by frame, rather than moved by
    joints. bone_len[i] is the rest length of the bone that ends at joint i. Returns the
    model, the list of joints that end a bone (body k of the model draws bone bones[k]),
    and the index of the head joint.
    """
    bones = [i for i in range(len(parents)) if parents[i] >= 0]
    head = joint_names.index('Head') if 'Head' in joint_names else None
    col = ' '.join(f'{c:g}' for c in rgba)
    bodies = []
    for k, j in enumerate(bones):
        length = max(float(bone_len[j]), 0.02)
        r = _human_radius(joint_names[j])
        bodies.append(
            f'    <body name="bone{k}" mocap="true" pos="0 0 0">\n'
            f'      <geom type="capsule" fromto="0 0 0 0 0 {length:.5f}" size="{r:.4f}" '
            f'rgba="{col}" contype="0" conaffinity="0"/>\n'
            f'    </body>')
    if head is not None:
        bodies.append(
            f'    <body name="head" mocap="true" pos="0 0 0">\n'
            f'      <geom type="sphere" size="0.108" rgba="{col}" '
            f'contype="0" conaffinity="0"/>\n    </body>')
    c1 = ' '.join(f'{c:g}' for c in CHECKER_LIGHT)
    c2 = ' '.join(f'{c:g}' for c in CHECKER_DARK)
    ortho = (f'orthographic="true" fovy="{ortho_height:g}"' if ortho_height else '')
    fsz = f'{floor_size[0]:g} {floor_size[1]:g} 0.05' if floor_size else '0 0 0.01'
    xml = f"""<mujoco model="human_figure">
  <compiler angle="radian"/>
  <statistic extent="3.0"/>
  <visual>
    <headlight diffuse="0.50 0.50 0.51" ambient="0.50 0.50 0.52" specular="0.16 0.16 0.16"/>
    <quality offsamples="{offsamples}" shadowsize="8192"/>
    <global offwidth="2400" offheight="1400" {ortho}/>
    <map znear="0.02" zfar="60" shadowclip="2.5" shadowscale="1"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="1 1 1" rgb2="1 1 1" width="800" height="800"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="none" rgb1="{c1}"
             rgb2="{c2}" width="512" height="512"/>
    <material name="groundplane" texture="groundplane" texuniform="true"
              texrepeat="{floor_repeat:g} {floor_repeat:g}" reflectance="0"/>
  </asset>
  <worldbody>
    <geom name="floor" size="{fsz}" type="plane" material="groundplane"
          contype="0" conaffinity="0"/>
    <light pos="-1.8 -1.5 5.0" dir="1.7 1.4 -4.8" directional="true"
           diffuse="0.42 0.42 0.43" specular="0.10 0.10 0.10" ambient="0 0 0"
           castshadow="true"/>
{chr(10).join(bodies)}
  </worldbody>
</mujoco>
"""
    model = mj.MjModel.from_xml_string(xml)
    return model, bones, head


def mannequin_pose(model, data, pos, bones, parents, head_idx, offset_xy=(0.0, 0.0)):
    """Place every capsule of the human figure for one frame.

    `pos` holds the joint positions of that frame in metres, shape (joints, 3).
    """
    p = pos.copy()
    p[:, 0] += offset_xy[0]
    p[:, 1] += offset_xy[1]
    z = np.array([0.0, 0.0, 1.0])
    for k, j in enumerate(bones):
        a, b = p[parents[j]], p[j]
        v = b - a
        length = np.linalg.norm(v)
        q = np.array([1.0, 0.0, 0.0, 0.0])
        if length > 1e-8:
            u = v / length
            c = float(np.dot(z, u))
            if c < -0.999999:
                q = np.array([0.0, 1.0, 0.0, 0.0])
            else:
                ax = np.cross(z, u)
                q = np.array([1.0 + c, ax[0], ax[1], ax[2]])
                q /= np.linalg.norm(q)
        mid = model.body_mocapid[mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, f'bone{k}')]
        data.mocap_pos[mid] = a
        data.mocap_quat[mid] = q
    if head_idx is not None:
        mid = model.body_mocapid[mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, 'head')]
        data.mocap_pos[mid] = p[head_idx] + np.array([0.0, 0.0, 0.055])
        data.mocap_quat[mid] = np.array([1.0, 0.0, 0.0, 0.0])
    mj.mj_forward(model, data)


class MannequinScene:
    """The same drawing as Scene, for the human figure built by build_mannequin."""

    def __init__(self, model, bones, parents, head_idx, width=1700, height=480,
                 max_poses=8):
        self.model, self.bones, self.parents, self.head = model, bones, parents, head_idx
        self.data = mj.MjData(model)
        self.renderer = mj.Renderer(model, height=height, width=width,
                                    max_geom=model.ngeom * (max_poses + 1) + 200)
        self.opt = mj.MjvOption()
        self.scratch = mj.MjvScene(model, maxgeom=model.ngeom + 100)

    def _pose(self, pos, offset_xy):
        mannequin_pose(self.model, self.data, pos, self.bones, self.parents, self.head,
                       offset_xy)

    def render_single(self, pos, cam, offset_xy=(0, 0)):
        """One picture of the figure in one frame."""
        self._pose(pos, offset_xy)
        self.renderer.update_scene(self.data, camera=cam, scene_option=self.opt)
        self.renderer.scene.flags[mj.mjtRndFlag.mjRND_SHADOW] = 1
        return self.renderer.render().copy()

    def render_multi(self, pos_list, offsets, cam, alphas=None, fades=None):
        """One picture holding several frames, placed at the given sideways offsets."""
        n = len(pos_list)
        if alphas is None and fades is None:
            fades = np.linspace(0.55, 0.0, n)
        if alphas is None:
            alphas = np.ones(n)
        if fades is None:
            fades = np.zeros(n)
        self._pose(pos_list[-1], offsets[-1])
        self.renderer.update_scene(self.data, camera=cam, scene_option=self.opt)
        sc = self.renderer.scene
        sc.flags[mj.mjtRndFlag.mjRND_SHADOW] = 1

        def paint(geom, alpha, fade):
            if fade > 1e-6:
                geom.rgba[:3] = np.array(geom.rgba[:3]) * (1.0 - fade) + fade
                geom.matid = -1
            geom.rgba[3] = alpha
            geom.transparent = 1 if alpha < 0.999 else 0

        for i in range(sc.ngeom):
            g = sc.geoms[i]
            if int(g.category) == int(mj.mjtCatBit.mjCAT_DYNAMIC):
                paint(g, float(alphas[-1]), float(fades[-1]))
        for k in range(n - 1):
            self._pose(pos_list[k], offsets[k])
            mj.mjv_updateScene(self.model, self.data, self.opt, None, cam,
                               int(mj.mjtCatBit.mjCAT_DYNAMIC), self.scratch)
            for i in range(self.scratch.ngeom):
                src = self.scratch.geoms[i]
                if int(src.category) != int(mj.mjtCatBit.mjCAT_DYNAMIC):
                    continue
                if sc.ngeom >= sc.maxgeom:
                    break
                dst = sc.geoms[sc.ngeom]
                _copy_geom(dst, src)
                paint(dst, float(alphas[k]), float(fades[k]))
                sc.ngeom = sc.ngeom + 1
        return self.renderer.render().copy()

    def close(self):
        self.renderer.close()


def fit_joint_angles(model, targets, body_names, iters=24, posture_cost=2e-2):
    """Find robot poses whose bodies sit at the given positions, frame by frame.

    `targets` has shape (frames, bodies, 3), in the order of `body_names`. Only positions
    are matched, never orientations; joint limits are respected; and each frame starts from
    the answer for the previous one. The solver is the inverse kinematics of the mink
    library. Returns the poses (frames, pose values), a summary of the fit error in metres
    (root mean square, median, 95th percentile and largest over frames), and the fit error
    of each frame.
    """
    import mink

    cfg = mink.Configuration(model)
    cfg.update(model.qpos0.copy())

    tasks = [mink.FrameTask(frame_name=n, frame_type="body", position_cost=1.0,
                            orientation_cost=0.0, lm_damping=1.0) for n in body_names]
    posture = mink.PostureTask(model, cost=posture_cost)
    posture.set_target(cfg.q.copy())
    limits = [mink.ConfigurationLimit(model)]
    dt = model.opt.timestep
    identity = mink.SO3.identity()

    n_frames = len(targets)
    poses = np.empty((n_frames, model.nq))
    per_frame = np.empty(n_frames)
    for t in range(n_frames):
        for task, p in zip(tasks, targets[t]):
            task.set_target(mink.SE3.from_rotation_and_translation(identity, np.asarray(p)))
        posture.set_target(cfg.q.copy())
        for _ in range(iters):
            vel = mink.solve_ik(cfg, tasks + [posture], dt, "daqp", damping=1e-3,
                                limits=limits)
            cfg.integrate_inplace(vel, dt)
        poses[t] = cfg.q.copy()
        err = np.concatenate([task.compute_error(cfg)[:3] for task in tasks])
        per_frame[t] = float(np.sqrt((err ** 2).sum() / len(tasks)))
    summary = {'rms_m': float(np.sqrt((per_frame ** 2).mean())),
               'median_m': float(np.median(per_frame)),
               'percentile_95_m': float(np.percentile(per_frame, 95)),
               'max_m': float(per_frame.max())}
    return poses, summary, per_frame


def write_video(path, frames, fps=30):
    """Write pictures as an H.264 video at `fps` frames per second."""
    import imageio.v2 as imageio
    path = str(path)
    imageio.mimwrite(path, list(frames), fps=fps, quality=8, macro_block_size=1,
                     codec='libx264', output_params=['-pix_fmt', 'yuv420p'])
    return path


def downscale(img, factor=2):
    """Shrink a picture by a whole factor, for the grid videos."""
    from PIL import Image
    im = Image.fromarray(img)
    return np.asarray(im.resize((im.width // factor, im.height // factor),
                                Image.LANCZOS))
