"""Show every method on several source groups at once.

The figure repeats the comparison of the main qualitative figures for all fourteen
methods and several groups of three source clips, so a reader can check that the
pattern is not particular to one skeleton or one action. Each block holds one group:
a row of three source clips on top, then one row per method with its three outputs.

The groups are written in TRIPLES below as query numbers of the benchmark set; the
script draws all of them, three blocks per figure file.

Run it after generating outputs for the query set:

    python -m paper.qualitative.fig_qualitative_full_grid --outputs <folder>

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
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

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

# (label in the figure, colour, output folder)
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

# Each group is three query numbers of the benchmark set that share a source
# skeleton, a target skeleton and an action.
TRIPLES = [
    ('SpiderG $\\to$ Tricera, attack',         [0, 1, 2]),
    ('SandMouse $\\to$ Ostrich, idle',         [10, 11, 12]),
    ('Dragon $\\to$ Giantbee, idle',           [15, 16, 17]),
    ('Horse $\\to$ Ant, idle',                 [96, 97, 98]),
    ('Coyote $\\to$ Ostrich, attack',          [117, 118, 119]),
    ('Rhino $\\to$ SpiderG, attack',           [141, 142, 143]),
    ('Hippopotamus $\\to$ Lion, attack',       [68, 69, 70]),
]

PHASES = [0.10, 0.50, 0.90]   # three poses per cell
SOURCE_COLOR = '#222222'


def setup_mpl():
    plt.rcParams.update({
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{amsmath}\usepackage{amssymb}',
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman'],
        'font.size': 7,
        'axes.unicode_minus': False,
        'pdf.fonttype': 42,
    })


def load_manifest():
    return json.loads(QUERY_SET.read_text())


def get_positions(motion, n_joints):
    return motion[:, :n_joints, 0:3].astype(np.float32).copy()


def best_view(pos):
    extents = np.ptp(pos.reshape(-1, 3), axis=0)
    if extents[1] >= max(extents[0], extents[2]) * 0.5:
        h = 0 if extents[0] > extents[2] else 2
        return h, 1
    else:
        order = np.argsort(extents)[::-1]
        return order[0], order[1]


def best_view_aggregate(pos_list):
    """Pick a single projection across multiple motions."""
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


def draw_pose_strip(ax, pos, parents, color, frames, stride_pad=1.40,
                lw=0.65, ms=2.2, axes_override=None,
                show_root_trail=True, root_color='#cccccc'):
    """Draw several poses of one motion side by side, left to right.

    Every pose is drawn at the same opacity; the sense of time comes from the spacing
    and from the faint grey line tracing where the root joint travelled."""
    if axes_override is not None:
        h_axis, v_axis = axes_override
    else:
        h_axis, v_axis = best_view(pos)
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
                    color=root_color, linewidth=0.3, alpha=0.45, zorder=1)
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


def find_query(manifest, qid):
    for q in manifest['queries']:
        if q['query_id'] == qid:
            return q
    raise ValueError(f'qid {qid} not found')


def main():
    """Write the grid as three figure files, so each one fits a printed page."""
    parse_args()
    setup_mpl()
    cond = np.load(COND_FILE, allow_pickle=True).item()
    manifest = load_manifest()

    splits = [
        (TRIPLES[0:3], 'app_qualitative_full_grid_A'),
        (TRIPLES[3:6], 'app_qualitative_full_grid_B'),
        (TRIPLES[6:7], 'app_qualitative_full_grid_C'),
    ]
    for triples_subset, out_name in splits:
        _render_split(triples_subset, out_name, cond, manifest)


def _render_split(triples_subset, out_name, cond, manifest):
    n_methods = len(METHODS)
    n_qpt = 3  # queries per triple

    # Up to three groups side by side, in landscape.
    n_tile_cols = max(1, len(triples_subset))
    n_tile_rows = 1
    rows_per_tile = n_methods + 1
    cols_per_tile = n_qpt

    # Sized so each file fits one landscape page with room for the caption.
    cell_w, cell_h = 0.55, 0.32
    label_w = 1.45
    title_h = 0.28

    tile_w = label_w + cols_per_tile * cell_w
    tile_h = title_h + rows_per_tile * cell_h

    fig_w = n_tile_cols * tile_w + 0.15
    fig_h = n_tile_rows * tile_h + 0.10

    fig = plt.figure(figsize=(fig_w, fig_h))
    outer = GridSpec(n_tile_rows, n_tile_cols, figure=fig,
                     wspace=0.04, hspace=0.10,
                     left=0.005, right=0.998, top=0.995, bottom=0.005)

    # Pre-load skeleton metadata for the assigned triples
    triple_data = []
    for triple_label, qids in triples_subset:
        qmeta = [find_query(manifest, qid) for qid in qids]
        skel_a = qmeta[0]['skel_a']; skel_b = qmeta[0]['skel_b']
        n_a = len(cond[skel_a]['parents'])
        n_b = len(cond[skel_b]['parents'])
        # Source positions
        src_data = []
        for qm in qmeta:
            sm = np.load(MOTION_DIR / qm['src_fname'])
            src_data.append({
                'qid': qm['query_id'],
                'pos': get_positions(sm, n_a),
            })
        # One viewing direction for all source clips of this group
        src_axes = best_view_aggregate([s['pos'] for s in src_data])
        # Pre-load all method outputs to compute the target projection once.
        tgt_pos_for_aggregate = []
        for method_name, color, out_dir in METHODS:
            for qm in qmeta:
                p = output_path(out_dir, qm['query_id'])
                if p.exists():
                    arr = np.load(p).astype(np.float32)
                    tgt_pos_for_aggregate.append(get_positions(arr, n_b))
        tgt_axes = best_view_aggregate(tgt_pos_for_aggregate) if tgt_pos_for_aggregate else (0, 1)

        triple_data.append({
            'label': triple_label,
            'qids': qids,
            'src_data': src_data,
            'src_parents': cond[skel_a]['parents'],
            'tgt_parents': cond[skel_b]['parents'],
            'n_b': n_b,
            'src_axes': src_axes,
            'tgt_axes': tgt_axes,
        })

    # Render each tile
    for ti, td in enumerate(triple_data):
        tr = ti // n_tile_cols
        tc = ti % n_tile_cols
        # Inner gridspec: rows_per_tile + 1 (title) × cols_per_tile + 1 (label)
        inner = GridSpecFromSubplotSpec(
            rows_per_tile + 1, cols_per_tile + 1,
            subplot_spec=outer[tr, tc],
            width_ratios=[label_w / cell_w] + [1.0] * cols_per_tile,
            height_ratios=[title_h / cell_h] + [1.0] * rows_per_tile,
            wspace=0.05, hspace=0.05)

        # Tile title (spans across the cell columns)
        ax_title = fig.add_subplot(inner[0, 1:])
        ax_title.text(0.5, 0.45, rf'\textbf{{{td["label"]}}}',
                      ha='center', va='center', fontsize=8.5,
                      transform=ax_title.transAxes)
        ax_title.set_xticks([]); ax_title.set_yticks([])
        for s in ax_title.spines.values():
            s.set_visible(False)

        # Source row (row 1 of inner gridspec)
        # Source row label
        ax_src_lbl = fig.add_subplot(inner[1, 0])
        ax_src_lbl.text(0.95, 0.5, r'\textit{source}',
                        ha='right', va='center', fontsize=7,
                        transform=ax_src_lbl.transAxes)
        ax_src_lbl.set_xticks([]); ax_src_lbl.set_yticks([])
        for s in ax_src_lbl.spines.values():
            s.set_visible(False)

        for ci, sd in enumerate(td['src_data']):
            ax = fig.add_subplot(inner[1, 1 + ci])
            frames = sample_phase_frames(sd['pos'].shape[0], PHASES)
            # Some source skeletons have many joints (Bird has 61), so small markers.
            x_min, x_max, y_min, y_max = draw_pose_strip(
                ax, sd['pos'], td['src_parents'],
                SOURCE_COLOR, frames, lw=0.45, ms=1.0,
                axes_override=td['src_axes'])
            ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
            style_strip_axis(ax)

        # Method rows (rows 2..rows_per_tile of inner)
        for mi, (method_name, color, out_dir) in enumerate(METHODS):
            row = 2 + mi
            # Method label
            ax_m_lbl = fig.add_subplot(inner[row, 0])
            ax_m_lbl.text(0.95, 0.5, method_name,
                          ha='right', va='center', fontsize=6.5,
                          color=color,
                          transform=ax_m_lbl.transAxes)
            ax_m_lbl.set_xticks([]); ax_m_lbl.set_yticks([])
            for s in ax_m_lbl.spines.values():
                s.set_visible(False)

            for ci, qid in enumerate(td['qids']):
                ax = fig.add_subplot(inner[row, 1 + ci])
                out_path = output_path(out_dir, qid)
                if out_path.exists():
                    m = np.load(out_path).astype(np.float32)
                    pos = get_positions(m, td['n_b'])
                    frames = sample_phase_frames(pos.shape[0], PHASES)
                    x_min, x_max, y_min, y_max = draw_pose_strip(
                        ax, pos, td['tgt_parents'], color, frames,
                        lw=0.65, ms=2.2,
                        axes_override=td['tgt_axes'])
                    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
                else:
                    ax.text(0.5, 0.5, '—', ha='center', va='center',
                            fontsize=6, color='gray',
                            transform=ax.transAxes)
                style_strip_axis(ax)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = OUT_DIR / f'{out_name}.pdf'
    out_png = OUT_DIR / f'{out_name}.png'
    fig.savefig(out_pdf, dpi=200, bbox_inches='tight')
    fig.savefig(out_png, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_pdf.name} + {out_png.name}')


if __name__ == '__main__':
    main()
