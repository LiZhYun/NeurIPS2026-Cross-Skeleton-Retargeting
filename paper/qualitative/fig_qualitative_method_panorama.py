"""Draw one source clip next to what every method made of it.

The figure takes a single source clip and shows, side by side, the motion each of the
twelve trained or retrieval methods produced for the same target skeleton. Twelve
different answers to the same question is the point the paper makes: nothing in the
training data says which one is right.

Each cell shows three poses of the motion, evenly spaced in time and laid out left to
right, with a faint grey line tracing where the root joint travelled. The source strip
at the top shows six poses of the source clip. Colours group methods by family.

Run it after generating outputs for the query set:

    python -m paper.qualitative.fig_qualitative_method_panorama --outputs <folder>

The output folder is laid out as the generate steps write it: <method>/set49/query_XXXX.npy,
with the folder names of docs/methods.md (for example ace_i, moreflow_t, al_flow).
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parents[2]
# Defaults, all overridable on the command line.
OUTPUT_ROOT = ROOT / 'outputs'     # outputs/<method>/set49/query_XXXX.npy, as generate writes them
COND_FILE = ROOT / 'dataset/truebones/zoo/truebones_processed/cond.npy'
MOTION_DIR = ROOT / 'dataset/truebones/zoo/truebones_processed/motions'
QUERY_SET = ROOT / 'benchmark/sets/truebones_49.json'
OUT_DIR = ROOT / 'paper/output/figures'


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

# The source clip: query 76, a Bird attack retargeted to KingCobra, chosen for its
# large movement.
QUERY_ID = 76

# (label in the figure, colour, output folder). The two random baselines are left out
# so the figure shows the trained and retrieval methods only.
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

PHASES_OUT = [0.10, 0.50, 0.90]   # three poses per method cell
PHASES_SRC = [0.05, 0.20, 0.40, 0.60, 0.80, 0.95]  # six poses in the wide source strip
SOURCE_COLOR = '#222222'


def setup_mpl():
    plt.rcParams.update({
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{amsmath}\usepackage{amssymb}\usepackage[HTML]{xcolor}',
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman'],
        'font.size': 8,
        'axes.unicode_minus': False,
        'pdf.fonttype': 42,
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


def sample_phase_frames(T, phases):
    return [int(round(p * (T - 1))) for p in phases]


def draw_pose_strip(ax, pos, parents, color, frames, axes_override,
                stride_pad=1.40, lw=0.7, ms=3,
                show_root_trail=True, root_color='#cccccc'):
    """Draw several poses of one motion side by side, left to right.

    Every pose is drawn at the same opacity; the sense of time comes from the spacing
    and from the faint grey line tracing where the root joint travelled."""
    h_axis, v_axis = axes_override
    h_extent = float(pos[..., h_axis].max() - pos[..., h_axis].min())
    v_extent = float(pos[..., v_axis].max() - pos[..., v_axis].min())
    body_size = max(0.3, v_extent)
    per_frame_h = float(max(
        pos[f, :, h_axis].max() - pos[f, :, h_axis].min() for f in frames
    ))
    stride = stride_pad * max(body_size, h_extent, per_frame_h)
    alphas = np.full(len(frames), 0.88)
    for si, f in enumerate(frames):
        x_off = si * stride
        p = pos[f]
        if show_root_trail:
            root = pos[:, 0, :]
            ax.plot(root[:, h_axis] + x_off,
                    root[:, v_axis],
                    color=root_color, linewidth=0.5, alpha=0.55, zorder=1)
        for j in range(p.shape[0]):
            pp = parents[j]
            if pp >= 0 and pp < p.shape[0] and pp != j:
                ax.plot([p[j, h_axis] + x_off, p[pp, h_axis] + x_off],
                        [p[j, v_axis], p[pp, v_axis]],
                        color=color, linewidth=lw, alpha=alphas[si],
                        solid_capstyle='round', zorder=3)
        ax.scatter(p[:, h_axis] + x_off, p[:, v_axis],
                   c=color, s=ms, alpha=alphas[si], edgecolors='none',
                   zorder=4)
    x_min = pos[..., h_axis].min() - 0.1 * body_size
    x_max = x_min + (len(frames) - 1) * stride + h_extent + 0.2 * body_size
    y_min = pos[..., v_axis].min() - 0.1 * body_size
    y_max = pos[..., v_axis].max() + 0.1 * body_size
    return x_min, x_max, y_min, y_max


def style_strip_axis(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_aspect('equal', adjustable='datalim')


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

    # Pre-load all method outputs
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

    # Three bands from top to bottom: source strip, arrow, grid of methods.
    src_panel_h = 1.10
    arrow_h = 0.32
    method_panel_w, method_panel_h = 1.65, 1.10
    legend_h = 0.32   # bottom strip reserved for the family-colour legend
    fig_w = grid_cols * method_panel_w + 0.30
    fig_h = (src_panel_h + arrow_h + grid_rows * method_panel_h
             + 0.55 + legend_h)
    fig = plt.figure(figsize=(fig_w, fig_h))

    bottom_frac = legend_h / fig_h
    gs = GridSpec(3, 1, figure=fig,
                  height_ratios=[src_panel_h, arrow_h,
                                 grid_rows * method_panel_h],
                  hspace=0.05,
                  left=0.03, right=0.99, top=0.91, bottom=bottom_frac)

    # -------- Source panel (top) --------
    ax_src = fig.add_subplot(gs[0])
    src_frames = sample_phase_frames(src_pos.shape[0], PHASES_SRC)
    # The Bird skeleton has 61 joints, so small markers and thin lines keep the
    # joints from merging into a blob.
    x_min, x_max, y_min, y_max = draw_pose_strip(
        ax_src, src_pos, src_parents, SOURCE_COLOR, src_frames,
        axes_override=src_axes, lw=0.7, ms=1.8)
    ax_src.set_xlim(x_min, x_max); ax_src.set_ylim(y_min, y_max)
    style_strip_axis(ax_src)
    ax_src.set_title(
        rf'\textbf{{Source clip}}: {skel_a} attack',
        loc='left', fontsize=9.5, pad=4)

    # -------- Arrow row (middle) --------
    ax_arr = fig.add_subplot(gs[1])
    ax_arr.set_xticks([]); ax_arr.set_yticks([])
    for s in ax_arr.spines.values():
        s.set_visible(False)
    ax_arr.text(0.5, 0.55,
                r'$\Big\downarrow$~~same source clip $\to$ 12 trained methods, '
                r'12 visibly different KingCobra outputs',
                ha='center', va='center', fontsize=9.5,
                transform=ax_arr.transAxes)

    # -------- Method grid (bottom) --------
    inner = gs[2].subgridspec(grid_rows, grid_cols, wspace=0.10, hspace=0.20)
    for mi, (name, color, _) in enumerate(METHODS):
        r, c = mi // grid_cols, mi % grid_cols
        ax = fig.add_subplot(inner[r, c])
        pos = method_pos[mi]
        if pos is not None:
            frames = sample_phase_frames(pos.shape[0], PHASES_OUT)
            x_min, x_max, y_min, y_max = draw_pose_strip(
                ax, pos, tgt_parents, color, frames,
                axes_override=tgt_axes, lw=0.9, ms=4)
            ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
        else:
            ax.text(0.5, 0.5, 'missing', transform=ax.transAxes,
                    ha='center', va='center', fontsize=6, color='gray')
        style_strip_axis(ax)
        ax.set_title(rf'\textbf{{{name}}}', fontsize=8.5, color=color, pad=3)

    # Legend of family colours, and one line saying how the poses are laid out.
    from matplotlib.patches import Patch
    family_swatches = [
        ('AL-Flow',     '#2ca02c'),
        ('AnyTop',      '#d62728'),
        ('MoReFlow',    '#ff7f0e'),
        ('ACE',         '#1f77b4'),
        ('DPG-SB-v3',   '#9467bd'),
        ('Motion2Motion', '#8c564b'),
        ('ANCHOR',      '#D4A017'),
    ]
    legend_handles = [Patch(facecolor=c, edgecolor='none', label=n)
                      for n, c in family_swatches]
    leg_ax = fig.add_axes([0.0, 0.018, 1.0, 0.04])
    leg_ax.set_axis_off()
    leg_ax.legend(handles=legend_handles, loc='center', ncol=7,
                  frameon=False, fontsize=7.5,
                  handlelength=1.0, handleheight=0.9, columnspacing=1.4,
                  handletextpad=0.4)
    fig.text(0.5, 0.001,
             r'3 frames per output cell at $[0.10, 0.50, 0.90]$; source strip '
             r'shows 6 frames; thin grey curve traces the root-joint trajectory.',
             ha='center', va='bottom', fontsize=6.8, color='#444444')

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = OUT_DIR / 'main_qualitative_method_panorama.pdf'
    out_png = OUT_DIR / 'main_qualitative_method_panorama.png'
    fig.savefig(out_pdf, dpi=300, bbox_inches='tight')
    fig.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_pdf.name} + {out_png.name}')


if __name__ == '__main__':
    main()
