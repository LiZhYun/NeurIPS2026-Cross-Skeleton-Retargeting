"""Draw the first half of the figure: one dance, six robots.

One ten-second dance window is shown as its true retarget on all six robots, next to the
human performance it came from. Each picture puts five moments of the same motion side by
side in one scene, with the earlier moments faded, so a single still carries the whole
movement. A video of each row is written as well.

The robots differ a lot in size, so each row is framed in proportion to its own robot: the
camera and the spacing scale with the robot's median base height over the clip, the height
of its root body above the floor. Nothing about the motion is changed.

Paths written into the record are relative to the output folder.

Runs in the robot environment (robots/environment.yml) on a machine with a graphics card:
    python -m robots.render.panel_six_robots
"""
import argparse
import json
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from robots.lafan1_to_six_robots.data import DEFAULT_DATA, ROBOTS, ROOT, load_clip_list
from robots.render import viz_core as V
from robots.render.paths import DEFAULT_OUT, VIDEO_SUFFIX

CLIP = "dance2_s2_w0"
RECORDING = "dance2_s2"
START_FRAME = 1692
N_FRAMES = 300
KEY_FRAMES = np.linspace(0, N_FRAMES - 1, 5).astype(int)

AZIMUTH, ELEVATION = 325.0, -11.0
VIEW_HEIGHT, SPACING = 2.1, 1.25
FADES = np.linspace(0.38, 0.0, 5)
STILL_W, STILL_H = 1800, 520
VIDEO_W, VIDEO_H = 960, 540
REFERENCE_HEIGHT = 0.793     # the base height the framing numbers were chosen for


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=str(DEFAULT_DATA))
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT))
    ap.add_argument("--gmr_dir", default=str(ROOT / "external/GMR"))
    ap.add_argument("--no_video", action="store_true", help="draw the stills only")
    args = ap.parse_args()

    if args.gmr_dir and args.gmr_dir not in sys.path:
        sys.path.insert(0, args.gmr_dir)
    from general_motion_retargeting.params import ROBOT_XML_DICT

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    stills = out_dir / "frames/six_robots"
    videos = out_dir / "videos/six_robots"
    stills.mkdir(parents=True, exist_ok=True)
    videos.mkdir(parents=True, exist_ok=True)

    report = {"clip_id": CLIP, "recording": RECORDING, "start_frame": START_FRAME,
              "fps": 30, "n_frames": N_FRAMES, "key_frames": KEY_FRAMES.tolist(),
              "key_times_s": (KEY_FRAMES / 30.0).round(2).tolist(),
              "camera": {"azimuth": AZIMUTH, "elevation": ELEVATION,
                         "projection": "orthographic",
                         "view_height_m_at_reference_scale": VIEW_HEIGHT,
                         "pose_spacing_m_at_reference_scale": SPACING},
              "rows": {}}
    tiles = {}

    human_meta = load_clip_list(data_dir)["meta"]["human"]
    human = np.load(data_dir / "tensors/human" / f"{CLIP}.npy").astype(np.float64)
    parents, names = human_meta["parents"], human_meta["joint_names"]
    bone_length = np.zeros(len(names))
    for j, parent in enumerate(parents):
        if parent >= 0:
            bone_length[j] = np.median(
                np.linalg.norm(human[:, j] - human[:, parent], axis=-1))

    model, bones, head = V.build_mannequin(names, parents, bone_length,
                                           ortho_height=VIEW_HEIGHT, floor_repeat=1.1)
    scene = V.MannequinScene(model, bones, parents, head, STILL_W, STILL_H, max_poses=6)
    offsets = V.spread_offsets(len(KEY_FRAMES), SPACING, AZIMUTH)
    poses = []
    for frame in KEY_FRAMES:
        pose = human[frame].copy()
        pose[:, :2] -= human[frame, 0, :2]
        poses.append(pose)
    imageio.imwrite(stills / "human.png",
                    scene.render_multi(poses, offsets,
                                       V.Scene.camera([0, 0, 0.85], 6.0, AZIMUTH,
                                                      ELEVATION), fades=FADES))
    scene.close()
    report["rows"]["human"] = {"still": str((stills / "human.png").relative_to(out_dir)),
                               "height_m": float(human[:, :, 2].max())}

    if not args.no_video:
        model, bones, head = V.build_mannequin(names, parents, bone_length,
                                               floor_size=(14.0, 14.0))
        scene = V.MannequinScene(model, bones, parents, head, VIDEO_W, VIDEO_H, max_poses=1)
        frames = []
        for t in range(N_FRAMES):
            camera = V.Scene.camera([human[t, 0, 0], human[t, 0, 1], 0.88], 3.0,
                                    AZIMUTH, -12.0)
            frames.append(scene.render_single(human[t], camera))
        video = videos / f"human{VIDEO_SUFFIX}"
        V.write_video(video, frames)
        tiles["human"] = [V.downscale(f) for f in frames]
        scene.close()
        report["rows"]["human"]["video"] = str(video.relative_to(out_dir))
    print("human done", flush=True)

    for robot in ROBOTS:
        started = time.time()
        pose = np.load(data_dir / f"retargeted/{robot}/{RECORDING}.npz")["qpos"]
        pose = np.asarray(pose[START_FRAME:START_FRAME + N_FRAMES], np.float64)
        scale = float(np.median(pose[:, 2])) / REFERENCE_HEIGHT

        model = V.load_model(ROBOT_XML_DICT[robot], ortho_height=VIEW_HEIGHT * scale,
                             floor_repeat=1.1 / scale)
        scene = V.Scene(model, STILL_W, STILL_H, max_poses=6)
        offsets = V.spread_offsets(len(KEY_FRAMES), SPACING * scale, AZIMUTH)
        camera = V.Scene.camera([0, 0, 0.85 * scale], 6.0 * scale, AZIMUTH, ELEVATION)
        imageio.imwrite(stills / f"{robot}.png",
                        scene.render_multi(V.stage_poses(pose, KEY_FRAMES, offsets),
                                           camera, fades=FADES))
        scene.close()
        report["rows"][robot] = {"still": str((stills / f"{robot}.png").relative_to(out_dir)),
                                 "median_base_height_m": float(np.median(pose[:, 2])),
                                 "framing_scale": scale}

        if not args.no_video:
            model = V.load_model(ROBOT_XML_DICT[robot],
                                 floor_size=(14.0 * scale, 14.0 * scale),
                                 floor_repeat=1.1 / scale)
            scene = V.Scene(model, VIDEO_W, VIDEO_H, max_poses=1)
            frames = []
            for t in range(N_FRAMES):
                camera = V.Scene.camera([pose[t, 0], pose[t, 1], 0.82 * scale],
                                        3.1 * scale, AZIMUTH, -12.0)
                frames.append(scene.render_single(pose[t], camera))
            video = videos / f"{robot}{VIDEO_SUFFIX}"
            V.write_video(video, frames)
            tiles[robot] = [V.downscale(f) for f in frames]
            scene.close()
            report["rows"][robot]["video"] = str(video.relative_to(out_dir))
        print(f"{robot} done in {time.time() - started:.0f}s (framing {scale:.2f})",
              flush=True)

    order = ["human"] + ROBOTS
    if not args.no_video:
        height, width = tiles["human"][0].shape[:2]
        grid = []
        for t in range(N_FRAMES):
            canvas = np.full((2 * height, 4 * width, 3), 255, np.uint8)
            for i, key in enumerate(order):
                row, col = divmod(i, 4)
                canvas[row * height:(row + 1) * height,
                       col * width:(col + 1) * width] = tiles[key][t]
            grid.append(canvas)
        grid_video = Path("videos") / f"six_robots_grid{VIDEO_SUFFIX}"
        V.write_video(out_dir / grid_video, grid)
        report["grid_video"] = str(grid_video)
    report["grid_order"] = order

    (out_dir / "reports").mkdir(parents=True, exist_ok=True)
    (out_dir / "reports/six_robots.json").write_text(json.dumps(report, indent=2))
    print("first half of the figure complete", flush=True)


if __name__ == "__main__":
    main()
