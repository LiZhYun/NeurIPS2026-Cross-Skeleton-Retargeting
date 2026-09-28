"""Animated version of the failure-mode figure.

Same layout as fig_qualitative_failuremodes.py: one row per failure mode, three source
clips on the left and the three motions one method produced from them on the right.
Here the skeletons move instead of being drawn as a strip of poses, every cell played
over the same normalised time.

Writes an MP4; needs ffmpeg.

    python -m paper.qualitative.video_qualitative_failuremodes --outputs <folder>
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch
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

QUERY_IDS = [76, 77, 79]   # Bird -> KingCobra attack triple
SOURCE_COLOR = '#222222'
FPS = 30
LOOPS = 3
PHASE_FRAMES = 90

ROWS = [
    ('Latent collapse',          'AL-Flow',
     '#2ca02c', 'al_flow'),
    ('Decoder collapse',         'AnyTop',
     '#d62728', 'anytop'),
    ('Source-conditional (partial)', 'MoReFlow-T',
     '#ff7f0e', 'moreflow_t'),
    ('Collapsed variation',      'ACE-I',
     '#1f77b4', 'ace_i'),
]


def setup_mpl():
    plt.rcParams.update({
        'text.usetex': False,        # animation: skip LaTeX for speed
        'font.family': 'serif',
        'font.size': 9,
        'axes.unicode_minus': False,
    })


def load_query_meta(qid):
    m = json.loads(QUERY_SET.read_text())
    for q in m['queries']:
        if q['query_id'] == qid:
            return q
    raise ValueError(f'qid {qid} not in manifest')


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
                          lw=1.0, ms=14, root_color='#cccccc'):
    h_axis, v_axis = axes_override
    n_j = pos.shape[1]
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
    root = pos[:, 0, :]
    ax.plot(root[:, h_axis], root[:, v_axis],
            color=root_color, linewidth=0.7, alpha=0.55, zorder=1)
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

    sources = []
    for qid in QUERY_IDS:
        q = load_query_meta(qid)
        src_motion = np.load(MOTION_DIR / q['src_fname'])
        n_a = len(cond[q['skel_a']]['parents'])
        src_pos = get_positions(src_motion, n_a)
        n_b = len(cond[q['skel_b']]['parents'])
        sources.append({
            'qid': qid, 'src_pos': src_pos,
            'src_parents': cond[q['skel_a']]['parents'],
            'tgt_parents': cond[q['skel_b']]['parents'],
            'n_b': n_b, 'src_fname': q['src_fname'],
        })

    src_axes = best_view_aggregate([s['src_pos'] for s in sources])

    method_outputs = {}
    target_pos_for_aggregate = []
    for ri, (mode_label, method_label, color, out_dir) in enumerate(ROWS):
        method_outputs[ri] = {}
        for sd in sources:
            out_path = output_path(out_dir, sd['qid'])
            if out_path.exists():
                m = np.load(out_path).astype(np.float32)
                pos = get_positions(m, sd['n_b'])
                method_outputs[ri][sd['qid']] = pos
                target_pos_for_aggregate.append(pos)
            else:
                method_outputs[ri][sd['qid']] = None
    tgt_axes = best_view_aggregate(target_pos_for_aggregate)

    n_rows = len(ROWS)
    # 1080p target
    fig_w, fig_h = 16.0, 9.0
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=120)

    gs = GridSpec(n_rows, 7, figure=fig,
                  width_ratios=[1.0, 1.0, 1.0, 0.18, 1.0, 1.0, 1.0],
                  wspace=0.06, hspace=0.18,
                  left=0.13, right=0.99, top=0.92, bottom=0.04)

    src_artists = {}
    out_artists = {}

    for ri, (mode_label, method_label, color, out_dir) in enumerate(ROWS):
        for ci_src, sd in enumerate(sources):
            ax = fig.add_subplot(gs[ri, ci_src])
            lines, scatter, hax, vax = make_skeleton_artists(
                ax, sd['src_pos'], sd['src_parents'],
                SOURCE_COLOR, src_axes,
                lw=1.0, ms=8)
            src_artists[(ri, ci_src)] = (sd['src_pos'], lines, scatter, hax, vax)

        for ci_out, sd in enumerate(sources):
            ax = fig.add_subplot(gs[ri, 4 + ci_out])
            pos = method_outputs[ri][sd['qid']]
            if pos is not None:
                lines, scatter, hax, vax = make_skeleton_artists(
                    ax, pos, sd['tgt_parents'], color, tgt_axes,
                    lw=1.4, ms=20)
                out_artists[(ri, ci_out)] = (pos, lines, scatter, hax, vax)
            else:
                ax.text(0.5, 0.5, 'missing', transform=ax.transAxes,
                        ha='center', va='center', fontsize=9, color='gray')
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(False)
                out_artists[(ri, ci_out)] = None

        # Row label (failure mode + method)
        y_row = 0.04 + (n_rows - 1 - ri + 0.5) / n_rows * (0.92 - 0.04)
        fig.text(0.005, y_row,
                 f'{mode_label}\n{method_label}',
                 fontsize=10, color=color, va='center', ha='left',
                 fontweight='bold')

    # Gutter arrows per row
    for ri in range(n_rows):
        ax_g = fig.add_subplot(gs[ri, 3])
        ax_g.set_xticks([]); ax_g.set_yticks([])
        ax_g.set_xlim(0, 1); ax_g.set_ylim(0, 1)
        for s in ax_g.spines.values():
            s.set_visible(False)
        ax_g.axvline(0.5, color='#888888', linewidth=0.8, ymin=0.05, ymax=0.95)
        arr = FancyArrowPatch((0.10, 0.5), (0.90, 0.5),
                               arrowstyle='->', mutation_scale=14,
                               color='#444444', linewidth=1.0,
                               transform=ax_g.transAxes)
        ax_g.add_patch(arr)

    # Group headers
    left, right = 0.13, 0.99
    fig.text(left + (right - left) * (1.5 / 7.18), 0.955,
             'Source clip (Bird, attack)',
             ha='center', va='bottom', fontsize=12, fontweight='bold')
    fig.text(left + (right - left) * (5.5 / 7.18), 0.955,
             'Output on KingCobra target skeleton',
             ha='center', va='bottom', fontsize=12, fontweight='bold')

    total_frames = LOOPS * PHASE_FRAMES

    def update(frame_idx):
        phase = (frame_idx % PHASE_FRAMES) / PHASE_FRAMES
        for art in src_artists.values():
            pos, lines, scatter, hax, vax = art
            T_cell = pos.shape[0]
            f_cell = int(round(phase * (T_cell - 1)))
            update_skeleton(pos[f_cell], lines, scatter, hax, vax)
        for art in out_artists.values():
            if art is None:
                continue
            pos, lines, scatter, hax, vax = art
            T_cell = pos.shape[0]
            f_cell = int(round(phase * (T_cell - 1)))
            update_skeleton(pos[f_cell], lines, scatter, hax, vax)
        return []

    anim = FuncAnimation(fig, update, frames=total_frames,
                         interval=1000.0 / FPS, blit=False)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / 'main_failuremodes.mp4'
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
