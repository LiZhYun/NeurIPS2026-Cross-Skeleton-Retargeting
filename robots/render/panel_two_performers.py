"""Draw the second half of the figure: two performers, same routine, one robot.

Two people perform the same dance. Their true retargets on the Unitree G1 keep them
visibly apart. The three trained models are drawn below, so a reader can see which of them
keeps the two performances apart and which returns nearly the same motion twice. The
models' motions are the ones the six-robot training step wrote for the dense setting.

The models produce body positions rather than joint angles, so each frame is put back onto
the robot by asking its joints, inside their real limits, to reach those positions. How far
off that fit is, is reported in centimetres and printed on the figure. The only other
change is one constant height shift per clip, which puts each row's lowest point at the
same height as the true retarget's, because the produced motions carry no height of their
own.

Paths written into the record are relative to the output folder (drawn files) or to the
data folder (the motions drawn).

Runs in the robot environment (robots/environment.yml) on a machine with a graphics card:
    python -m robots.render.panel_two_performers
"""
import argparse
import json
import sys
import time
from pathlib import Path

import imageio.v2 as imageio
import mujoco as mj
import numpy as np

from robots.lafan1_to_six_robots.data import DEFAULT_DATA, HORIZON, ROOT
from robots.render import viz_core as V
from robots.render.paths import DEFAULT_OUT, VIDEO_SUFFIX

ACTION_GROUP, WINDOW, SETTING = "dance2", "w0", "dense"
PERFORMERS = [("dance2_s1_w0", "dance2_s1", 1, 1692),
              ("dance2_s2_w0", "dance2_s2", 2, 1692)]
KEY_FRAMES = np.linspace(0, HORIZON - 1, 5).astype(int)

ROWS = [("true_retarget", None, (0.70, 0.72, 0.76), "true retarget"),
        ("true_pair_model", "true_pair_model", (0.180, 0.490, 0.357),
         "model, true pairs"),
        ("averaging_objective", "averaging_objective", (0.710, 0.416, 0.165),
         "averaging objective"),
        ("unpaired_objective", "unpaired_objective", (0.710, 0.416, 0.165),
         "unpaired objective")]

STILL_W, STILL_H = 1800, 520
VIDEO_W, VIDEO_H = 960, 540
FADES = np.linspace(0.38, 0.0, len(KEY_FRAMES))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=str(DEFAULT_DATA))
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT))
    ap.add_argument("--gmr_dir", default=str(ROOT / "external/GMR"))
    ap.add_argument("--azimuth", type=float, default=318.0,
                    help="camera direction around the robot, in degrees")
    ap.add_argument("--elevation", type=float, default=-11.0,
                    help="camera angle above the floor, in degrees (negative looks down)")
    ap.add_argument("--spacing", type=float, default=1.25,
                    help="distance between the five poses of a still, in metres")
    ap.add_argument("--view_height", type=float, default=2.1,
                    help="height of the scene a still shows, in metres")
    ap.add_argument("--fit_steps", type=int, default=30,
                    help="solver steps per frame when fitting joint angles to positions")
    ap.add_argument("--no_video", action="store_true", help="draw the stills only")
    args = ap.parse_args()

    if args.gmr_dir and args.gmr_dir not in sys.path:
        sys.path.insert(0, args.gmr_dir)
    from general_motion_retargeting.params import ROBOT_XML_DICT

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    stills = out_dir / "frames/two_performers"
    videos = out_dir / "videos/two_performers"
    stills.mkdir(parents=True, exist_ok=True)
    videos.mkdir(parents=True, exist_ok=True)

    model = V.load_model(ROBOT_XML_DICT["unitree_g1"], ortho_height=args.view_height,
                         floor_repeat=1.1)
    video_model = V.load_model(ROBOT_XML_DICT["unitree_g1"], floor_size=(14.0, 14.0),
                               floor_repeat=1.1)
    _, body_names = V.body_rows(model)
    probe = mj.MjData(model)

    report = {"action_group": ACTION_GROUP, "window": WINDOW, "setting": SETTING,
              "robot": "unitree_g1",
              "performers": [{"clip_id": c, "recording": r, "performer": p,
                              "start_frame": s} for c, r, p, s in PERFORMERS],
              "n_frames": HORIZON, "key_frames": KEY_FRAMES.tolist(), "fps": 30,
              "camera": {"azimuth": args.azimuth, "elevation": args.elevation,
                         "projection": "orthographic", "view_height_m": args.view_height,
                         "pose_spacing_m": args.spacing},
              "tiles": {}}
    tiles, poses = {}, {}

    still_camera = V.Scene.camera([0.0, 0.0, 0.85], 6.0, args.azimuth, args.elevation)
    video_camera = V.Scene.camera([0.0, 0.0, 0.82], 3.1, args.azimuth, -12.0)
    offsets = V.spread_offsets(len(KEY_FRAMES), args.spacing, args.azimuth)
    still_scene = V.Scene(model, STILL_W, STILL_H, max_poses=6)
    video_scene = None if args.no_video else V.Scene(video_model, VIDEO_W, VIDEO_H,
                                                     max_poses=1)

    for clip_id, recording, performer, start in PERFORMERS:
        true_pose = np.asarray(
            np.load(data_dir / f"retargeted/unitree_g1/{recording}.npz")["qpos"]
            [start:start + HORIZON], np.float64)
        ground = float(np.percentile(V.lowest_point(model, probe, true_pose), 5))
        for row_key, model_name, tint, label in ROWS:
            started = time.time()
            info = {"label": label, "tint": tint}
            if model_name is None:
                pose = true_pose
                shift = 0.0
                info["source"] = f"retargeted/unitree_g1/{recording}.npz"
            else:
                produced_file = (Path("generated/unitree_g1") / SETTING /
                                 model_name / f"{clip_id}.npy")
                produced = np.load(data_dir / produced_file).astype(np.float64)
                pose, fit, _ = V.fit_joint_angles(model, produced, body_names,
                                                  iters=args.fit_steps)
                shift = ground - float(
                    np.percentile(V.lowest_point(model, probe, pose), 5))
                info["source"] = str(produced_file)
                info["fit_error_m"] = fit
                info["ground_shift_m"] = shift
                info["ground_reference_m"] = ground
            lowest = V.lowest_point(model, probe, pose)
            info["lowest_point_m"] = {
                "fifth_percentile": float(np.percentile(lowest, 5) + shift),
                "min": float(lowest.min() + shift)}

            tag = f"{clip_id}__{row_key}"
            poses[tag] = pose
            image = still_scene.render_multi(
                V.stage_poses(pose, KEY_FRAMES, offsets, ground_dz=shift), still_camera,
                tint=tint, fades=FADES)
            imageio.imwrite(stills / f"{tag}.png", image)
            info["still"] = str((stills / f"{tag}.png").relative_to(out_dir))

            if not args.no_video:
                frames = []
                for t in range(HORIZON):
                    frame_pose = np.array(pose[t])
                    frame_pose[0:2] = 0.0
                    frame_pose[2] += shift
                    frames.append(video_scene.render_single(frame_pose, video_camera,
                                                            tint=tint))
                video = videos / f"{tag}{VIDEO_SUFFIX}"
                V.write_video(video, frames)
                tiles[tag] = [V.downscale(f) for f in frames]
                info["video"] = str(video.relative_to(out_dir))
            report["tiles"][tag] = info
            print(f"{tag}: {time.time() - started:.0f}s "
                  f"{info.get('fit_error_m', '')}", flush=True)

    still_scene.close()
    if video_scene is not None:
        video_scene.close()

    body_ids, _ = V.body_rows(model)
    first, second = PERFORMERS[0][0], PERFORMERS[1][0]
    separation = {}
    for row_key, *_ in ROWS:
        a = V.body_positions(model, probe, poses[f"{first}__{row_key}"], body_ids)
        b = V.body_positions(model, probe, poses[f"{second}__{row_key}"], body_ids)
        a = a - a[:, 0:1]
        b = b - b[:, 0:1]
        separation[row_key] = {
            "performer_gap_m": float(np.linalg.norm(a - b, axis=-1).mean()),
            "motion_a_m_per_frame": float(
                np.linalg.norm(np.diff(a, axis=0), axis=-1).mean()),
            "motion_b_m_per_frame": float(
                np.linalg.norm(np.diff(b, axis=0), axis=-1).mean()),
            "key_frame_gap_m": [float(np.linalg.norm(a[k] - b[k], axis=-1).mean())
                                for k in KEY_FRAMES]}
    report["performer_separation"] = separation
    print(json.dumps(separation, indent=2), flush=True)

    if not args.no_video:
        height, width = next(iter(tiles.values()))[0].shape[:2]
        grid = []
        for t in range(HORIZON):
            canvas = np.full((4 * height, 2 * width, 3), 255, np.uint8)
            for r, (row_key, *_rest) in enumerate(ROWS):
                for c, (clip_id, *_rest2) in enumerate(PERFORMERS):
                    canvas[r * height:(r + 1) * height, c * width:(c + 1) * width] = \
                        tiles[f"{clip_id}__{row_key}"][t]
            grid.append(canvas)
        grid_video = Path("videos") / f"two_performers_grid{VIDEO_SUFFIX}"
        V.write_video(out_dir / grid_video, grid)
        report["grid_video"] = str(grid_video)

    (out_dir / "reports").mkdir(parents=True, exist_ok=True)
    (out_dir / "reports/two_performers.json").write_text(json.dumps(report, indent=2))
    print("second half of the figure complete", flush=True)


if __name__ == "__main__":
    main()
