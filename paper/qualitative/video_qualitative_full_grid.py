"""Animated version of the full grid, one video per source group.

Each video shows one group: the three source clips on top, then one row per method
with its three outputs. The skeletons move instead of being drawn as a strip of poses,
every cell played over the same normalised time.

Writes one MP4 per group; needs ffmpeg.

    python -m paper.qualitative.video_qualitative_full_grid --outputs <folder>
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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

FPS = 30
LOOPS = 3
PHASE_FRAMES = 90
SOURCE_COLOR = '#222222'

METHODS = [
    ('AL-Flow',           '#2ca02c', 'al_flow'),
    ('AL-Flow-Src',       '#98df8a', 'al_flow_src'),
    ('AL-Flow-Src-G',     '#5fb15f', 'al_flow_src_g'),
    ('AnyTop',            '#d62728', 'anytop'),
    ('MoReFlow-T',        '#ff7f0e', 'moreflow_t'),
    ('MoReFlow-I',        '#ffbb78', 'moreflow_i'),
    ('ACE-T',             '#1f77b4', 'ace_t'),
    ('ACE-I',             '#6baed6', 'ace_i'),
    ('DPG-SB-v3',         '#9467bd', 'dpg_sb'),
    ('Motion2Motion-Direct', '#8c564b', 'motion2motion_direct'),
    ('Motion2Motion-BVH', '#c49c94', 'motion2motion_bvh'),
    ('ANCHOR',            '#D4A017', 'anchor'),
    ('random-same-cluster',         '#888888', 'random_same_cluster'),
    ('random-same-exact-action',    '#bbbbbb', 'random_same_exact_action'),
]

TRIPLES = [
    ('spiderg_tricera_attack',     'SpiderG -> Tricera, attack',         [0, 1, 2]),
    ('sandmouse_ostrich_idle',     'SandMouse -> Ostrich, idle',         [10, 11, 12]),
    ('dragon_giantbee_idle',       'Dragon -> Giantbee, idle',           [15, 16, 17]),
    ('horse_ant_idle',             'Horse -> Ant, idle',                 [96, 97, 98]),
    ('coyote_ostrich_attack',      'Coyote -> Ostrich, attack',          [117, 118, 119]),
    ('rhino_spiderg_attack',       'Rhino -> SpiderG, attack',           [141, 142, 143]),
    ('hippo_lion_attack',          'Hippopotamus -> Lion, attack',       [68, 69, 70]),
]


def setup_mpl():
    plt.rcParams.update({
        'text.usetex': False,
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


def find_query(manifest, qid):
    for q in manifest['queries']:
        if q['query_id'] == qid:
            return q
    raise ValueError(f'qid {qid} not found')


def make_skeleton_artists(ax, pos, parents, color, axes_override,
                          lw=0.7, ms=6, root_color='#cccccc'):
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
            color=root_color, linewidth=0.4, alpha=0.45, zorder=1)
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


def render_triple(triple_slug, triple_label, qids, cond, manifest):
    """Render one video for one triple."""
    qmeta = [find_query(manifest, qid) for qid in qids]
    skel_a = qmeta[0]['skel_a']; skel_b = qmeta[0]['skel_b']
    n_a = len(cond[skel_a]['parents'])
    n_b = len(cond[skel_b]['parents'])
    src_parents = cond[skel_a]['parents']
    tgt_parents = cond[skel_b]['parents']

    # Source positions
    src_data = []
    for qm in qmeta:
        sm = np.load(MOTION_DIR / qm['src_fname'])
        src_data.append({
            'qid': qm['query_id'],
            'pos': get_positions(sm, n_a),
        })
    src_axes = best_view_aggregate([s['pos'] for s in src_data])

    # Pre-load all method outputs
    method_outputs = {}
    target_pos_for_aggregate = []
    for mi, (name, color, out_dir) in enumerate(METHODS):
        method_outputs[mi] = {}
        for qm in qmeta:
            p = output_path(out_dir, qm['query_id'])
            if p.exists():
                arr = np.load(p).astype(np.float32)
                pos = get_positions(arr, n_b)
                method_outputs[mi][qm['query_id']] = pos
                target_pos_for_aggregate.append(pos)
            else:
                method_outputs[mi][qm['query_id']] = None
    tgt_axes = best_view_aggregate(target_pos_for_aggregate) if target_pos_for_aggregate else (0, 1)

    n_methods = len(METHODS)
    n_rows = n_methods + 1   # 1 source row + 14 method rows = 15
    n_cols = 3

    # Portrait video: 1080 wide x 1920 tall (9:16)
    # Layout columns: [label_col, src/out cols x 3]
    fig_w, fig_h = 9.0, 16.0
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=120)

    gs = GridSpec(n_rows, n_cols + 1, figure=fig,
                  width_ratios=[1.4, 1.0, 1.0, 1.0],
                  hspace=0.16, wspace=0.05,
                  left=0.005, right=0.995, top=0.965, bottom=0.025)

    # Top-row title (the triple name)
    fig.text(0.5, 0.985, triple_label, ha='center', va='top',
             fontsize=14, fontweight='bold')

    # Source row label
    ax_s_lbl = fig.add_subplot(gs[0, 0])
    ax_s_lbl.text(0.95, 0.5, 'source', ha='right', va='center',
                  fontsize=10, style='italic',
                  transform=ax_s_lbl.transAxes)
    ax_s_lbl.set_xticks([]); ax_s_lbl.set_yticks([])
    for s in ax_s_lbl.spines.values():
        s.set_visible(False)

    src_artists = {}
    for ci, sd in enumerate(src_data):
        ax = fig.add_subplot(gs[0, 1 + ci])
        lines, scatter, hax, vax = make_skeleton_artists(
            ax, sd['pos'], src_parents, SOURCE_COLOR, src_axes,
            lw=0.6, ms=4)
        src_artists[ci] = (sd['pos'], lines, scatter, hax, vax)

    # Method rows
    out_artists = {}
    for mi, (name, color, _) in enumerate(METHODS):
        row = mi + 1
        # Row label (method name)
        ax_lbl = fig.add_subplot(gs[row, 0])
        ax_lbl.text(0.95, 0.5, name, ha='right', va='center',
                    fontsize=8.5, color=color, fontweight='bold',
                    transform=ax_lbl.transAxes)
        ax_lbl.set_xticks([]); ax_lbl.set_yticks([])
        for s in ax_lbl.spines.values():
            s.set_visible(False)

        for ci, qid in enumerate(qids):
            ax = fig.add_subplot(gs[row, 1 + ci])
            pos = method_outputs[mi][qid]
            if pos is not None:
                lines, scatter, hax, vax = make_skeleton_artists(
                    ax, pos, tgt_parents, color, tgt_axes,
                    lw=0.7, ms=5)
                out_artists[(mi, ci)] = (pos, lines, scatter, hax, vax)
            else:
                ax.text(0.5, 0.5, '-', ha='center', va='center',
                        fontsize=8, color='gray',
                        transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(False)
                out_artists[(mi, ci)] = None

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
    out_path = OUT_DIR / f'app_triple_{triple_slug}.mp4'
    writer = FFMpegWriter(fps=FPS, codec='libx264',
                          bitrate=4500,
                          extra_args=['-pix_fmt', 'yuv420p',
                                      '-preset', 'medium',
                                      '-crf', '20'])
    t0 = time.time()
    print(f'Rendering {total_frames} frames -> {out_path.name}')
    anim.save(str(out_path), writer=writer, dpi=120)
    plt.close(fig)
    dt = time.time() - t0
    sz_mb = out_path.stat().st_size / 1024 / 1024
    print(f'  done in {dt:.1f}s ({sz_mb:.2f} MB)')


def main():
    parse_args()
    setup_mpl()
    cond = np.load(COND_FILE, allow_pickle=True).item()
    manifest = json.loads(QUERY_SET.read_text())
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    t_total_start = time.time()
    for ti, (slug, label, qids) in enumerate(TRIPLES, start=1):
        print(f'\n[{ti}/{len(TRIPLES)}] Triple {slug}')
        # Prefix the slug with its index for natural ordering
        slug_indexed = f'{ti}_{slug}'
        render_triple(slug_indexed, label, qids, cond, manifest)
    dt_total = time.time() - t_total_start
    print(f'\nAll {len(TRIPLES)} triples rendered in {dt_total/60:.1f} min')


if __name__ == '__main__':
    main()
