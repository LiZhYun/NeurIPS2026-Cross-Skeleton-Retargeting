"""Animated version of the source-clip comparison figure.

Same layout as fig_qualitative_method_panorama.py: one source clip on top and the
motion each method produced below. Here the skeletons move instead of being drawn as
a strip of poses. Outputs have different lengths, so every cell is played over the
same normalised time, from the start of its own clip to its end.

Writes an MP4; needs ffmpeg.

    python -m paper.qualitative.video_qualitative_method_panorama --outputs <folder>
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from matplotlib.animation import FuncAnimation, FFMpegWriter

ROOT = Path(__file__).resolve().parents[2]
# Defaults, all overridable on the command line.
OUTPUT_ROOT = ROOT / 'outputs'     # outputs/<method>/set49/query_XXXX.npy, as generate writes them
COND_FILE = ROOT / 'dataset/truebones/zoo/truebones_processed/cond.npy'
MOTION_DIR = ROOT / 'dataset/truebones/zoo/truebones_processed/motions'
QUERY_SET = ROOT / 'benchmark/sets/truebones_49.json'
OUT_DIR = ROOT / 'paper/output/videos'


def output_path(folder, query_id):
    """Where a method's answer to one query lives."""
    return OUTPUT_ROOT / folder / 'set49' / f'query_{query_id:04d}.npy'


def parse_args():
    """Read the folders to work from, falling back to the ones next to this repository."""
    global OUTPUT_ROOT, MOTION_DIR, COND_FILE, QUERY_SET, OUT_DIR
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--outputs', default=str(OUTPUT_ROOT),
                    help='folder holding <method>/set49/query_XXXX.npy for every method')
    ap.add_argument('--motions', default=str(MOTION_DIR),
                    help='folder of processed Truebones motions')
    ap.add_argument('--cond', default=str(COND_FILE),
                    help='file describing every skeleton')
    ap.add_argument('--query-set', default=str(QUERY_SET),
                    help='the benchmark query set to read the queries from')
    ap.add_argument('--out-dir', default=str(OUT_DIR), help='where to write the result')
    a = ap.parse_args()
    OUTPUT_ROOT = Path(a.outputs)
    MOTION_DIR = Path(a.motions)
    COND_FILE = Path(a.cond)
    QUERY_SET = Path(a.query_set)
    OUT_DIR = Path(a.out_dir)

QUERY_ID = 76
FPS = 30
LOOPS = 3                  # play through the motion 3x for a longer video
PHASE_FRAMES = 90          # 90 normalised phase steps per loop -> 3 sec/loop
SOURCE_COLOR = '#222222'

METHODS = [
    ('AL-Flow',         '#2ca02c', 'al_flow'),
    ('AL-Flow-Src',     '#98df8a', 'al_flow_src'),
    ('AL-Flow-Src-G',   '#5fb15f', 'al_flow_src_g'),
    ('AnyTop',          '#d62728', 'anytop'),
    ('MoReFlow-T',      '#ff7f0e', 'moreflow_t'),
    ('MoReFlow-I',      '#ffbb78', 'moreflow_i'),
    ('ACE-T',           '#1f77b4', 'ace_t'),
    ('ACE-I',           '#6baed6', 'ace_i'),
    ('DPG-SB-v3',       '#9467bd', 'dpg_sb'),
    ('Motion2Motion-Direct', '#8c564b', 'motion2motion_direct'),
    ('Motion2Motion-BVH', '#c49c94', 'motion2motion_bvh'),
    ('ANCHOR',          '#D4A017', 'anchor'),
]


def setup_mpl():
    plt.rcParams.update({
        'text.usetex': False,        # animation: skip LaTeX for speed
        'font.family': 'serif',
        'font.size': 8,
        'axes.unicode_minus': False,
    })


def get_positions(motion, n_joints):
    return motion[:, :n_joints, 0:3].astype(np.float32).copy()


def best_view_aggregate(pos_list):
    all_pos = np.concatenate([p.reshape(-1, 3) for p in pos_list], axis=0)
    extents = np.ptp(all_pos, axis=0)
    if extents[1] >= max(extents[0], extents[2]) * 0.5:
        h = 0 if extents[0] > extents[2] else 2
        return h, 1
    else:
        order = np.argsort(extents)[::-1]
        return order[0], order[1]


def make_skeleton_artists(ax, pos, parents, color, axes_override,
                          lw=1.0, ms=12, root_color='#cccccc'):
    """Draw one skeleton that update_skeleton can move later, with the path of its root
    joint as a fixed grey line. Returns the bone lines, the joint markers and the two
    axes the poses are projected on."""
    h_axis, v_axis = axes_override
    n_j = pos.shape[1]
    # One Line2D per bone
    lines = []
    for j in range(n_j):
        pp = parents[j]
        if pp >= 0 and pp < n_j and pp != j:
            (line,) = ax.plot([0, 0], [0, 0],
                              color=color, linewidth=lw, alpha=0.92,
                              solid_capstyle='round', zorder=3)
            lines.append((j, pp, line))
    scatter = ax.scatter(np.zeros(n_j), np.zeros(n_j),
                         c=color, s=ms, alpha=0.92,
                         edgecolors='none', zorder=4)
    # Root trail (full path, static across the animation)
    root = pos[:, 0, :]
    ax.plot(root[:, h_axis], root[:, v_axis],
            color=root_color, linewidth=0.8, alpha=0.55, zorder=1)
    # Set xlim/ylim once based on the full trajectory extent
    h_extent = float(pos[..., h_axis].max() - pos[..., h_axis].min())
    v_extent = float(pos[..., v_axis].max() - pos[..., v_axis].min())
    body_size = max(0.3, max(h_extent, v_extent))
    pad = 0.15 * body_size
    ax.set_xlim(pos[..., h_axis].min() - pad, pos[..., h_axis].max() + pad)
    ax.set_ylim(pos[..., v_axis].min() - pad, pos[..., v_axis].max() + pad)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return lines, scatter, h_axis, v_axis


def update_skeleton(pos_frame, lines, scatter, h_axis, v_axis):
    for j, pp, line in lines:
        line.set_data([pos_frame[j, h_axis], pos_frame[pp, h_axis]],
                      [pos_frame[j, v_axis], pos_frame[pp, v_axis]])
    scatter.set_offsets(np.c_[pos_frame[:, h_axis], pos_frame[:, v_axis]])


def main():
    parse_args()
    setup_mpl()
    cond = np.load(COND_FILE, allow_pickle=True).item()
    manifest = json.loads(QUERY_SET.read_text())
    q = next(qq for qq in manifest['queries'] if qq['query_id'] == QUERY_ID)
    skel_a, skel_b = q['skel_a'], q['skel_b']
    src_parents = cond[skel_a]['parents']
    tgt_parents = cond[skel_b]['parents']
    n_a = len(src_parents); n_b = len(tgt_parents)

    # Source motion
    src_motion = np.load(MOTION_DIR / q['src_fname'])
    src_pos = get_positions(src_motion, n_a)
    src_axes = best_view_aggregate([src_pos])

    # Method outputs
    method_pos = []
    for name, color, out_dir in METHODS:
        out_path = output_path(out_dir, QUERY_ID)
        if out_path.exists():
            arr = np.load(out_path).astype(np.float32)
            method_pos.append(get_positions(arr, n_b))
        else:
            method_pos.append(None)
    valid = [p for p in method_pos if p is not None]
    tgt_axes = best_view_aggregate(valid) if valid else (0, 1)

    n_methods = len(METHODS)
    grid_cols = 6
    grid_rows = (n_methods + grid_cols - 1) // grid_cols

    # Layout: 1080p target (1920x1080 at 120 dpi -> figsize 16x9)
    fig_w, fig_h = 16.0, 9.0
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=120)
    gs = GridSpec(3, 1, figure=fig,
                  height_ratios=[2.4, 0.45, grid_rows * 1.55],
                  hspace=0.18,
                  left=0.02, right=0.99, top=0.93, bottom=0.06)

    # Source panel
    ax_src = fig.add_subplot(gs[0])
    src_lines, src_scatter, src_h, src_v = make_skeleton_artists(
        ax_src, src_pos, src_parents, SOURCE_COLOR, src_axes,
        lw=1.4, ms=20)
    ax_src.set_title(
        f'Source clip: {skel_a} attack', loc='left', fontsize=14,
        fontweight='bold', pad=4)

    # Arrow / explanation row
    ax_arr = fig.add_subplot(gs[1])
    ax_arr.set_xticks([]); ax_arr.set_yticks([])
    for s in ax_arr.spines.values():
        s.set_visible(False)
    ax_arr.text(0.5, 0.55,
                r'$\downarrow$  same source clip $\rightarrow$ '
                '12 trained methods, 12 visibly different KingCobra outputs',
                ha='center', va='center', fontsize=12,
                transform=ax_arr.transAxes)

    # Method grid
    inner = gs[2].subgridspec(grid_rows, grid_cols, wspace=0.08, hspace=0.30)
    method_artists = []
    for mi, (name, color, _) in enumerate(METHODS):
        r, c = mi // grid_cols, mi % grid_cols
        ax = fig.add_subplot(inner[r, c])
        pos = method_pos[mi]
        if pos is not None:
            lines, scatter, hax, vax = make_skeleton_artists(
                ax, pos, tgt_parents, color, tgt_axes, lw=1.2, ms=16)
            method_artists.append((pos, lines, scatter, hax, vax))
        else:
            ax.text(0.5, 0.5, 'missing', transform=ax.transAxes,
                    ha='center', va='center', fontsize=10, color='gray')
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            method_artists.append(None)
        ax.set_title(name, fontsize=11, color=color, fontweight='bold', pad=4)

    # Family colour legend at the bottom
    family_swatches = [
        ('AL-Flow', '#2ca02c'), ('AnyTop', '#d62728'),
        ('MoReFlow', '#ff7f0e'), ('ACE', '#1f77b4'),
        ('DPG-SB-v3', '#9467bd'), ('Motion2Motion', '#8c564b'),
        ('ANCHOR', '#D4A017'),
    ]
    legend_handles = [Patch(facecolor=c, edgecolor='none', label=n)
                      for n, c in family_swatches]
    fig.legend(handles=legend_handles, loc='lower center', ncol=7,
               frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, 0.005),
               handlelength=1.4, handleheight=1.0,
               columnspacing=1.6, handletextpad=0.5)

    # Animation update function
    total_frames = LOOPS * PHASE_FRAMES

    def update(frame_idx):
        phase = (frame_idx % PHASE_FRAMES) / PHASE_FRAMES
        # Source
        T_src = src_pos.shape[0]
        f_src = int(round(phase * (T_src - 1)))
        update_skeleton(src_pos[f_src], src_lines, src_scatter, src_h, src_v)
        # Methods
        for ma in method_artists:
            if ma is None:
                continue
            pos, lines, scatter, hax, vax = ma
            T_cell = pos.shape[0]
            f_cell = int(round(phase * (T_cell - 1)))
            update_skeleton(pos[f_cell], lines, scatter, hax, vax)
        return []

    anim = FuncAnimation(fig, update, frames=total_frames,
                         interval=1000.0 / FPS, blit=False)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / 'main_panorama.mp4'
    writer = FFMpegWriter(fps=FPS, codec='libx264',
                          bitrate=4000,
                          extra_args=['-pix_fmt', 'yuv420p',
                                      '-preset', 'medium',
                                      '-crf', '20'])
    print(f'Rendering {total_frames} frames -> {out_path}')
    anim.save(str(out_path), writer=writer, dpi=120)
    plt.close(fig)
    print(f'Saved {out_path.name}')


if __name__ == '__main__':
    main()
