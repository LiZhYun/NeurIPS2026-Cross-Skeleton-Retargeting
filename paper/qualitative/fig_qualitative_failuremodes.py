"""Show four ways a method can lose what made a source clip distinct.

Each row takes three different source clips of the same skeleton and action, and puts
them next to the three motions one method produced from them. Reading a row left to
right shows whether the three outputs still differ the way the three sources do.

The four rows are one method per failure mode: the latent codes collapse, the decoder
throws the difference away, the output depends partly on the source, and the output keeps
a positive link to the source while barely varying (collapsed variation). The retrieval comparator is left out on purpose: it ignores the
source clip, so all three of its outputs would be the same motion.

Each cell shows three poses of the motion, evenly spaced in time and laid out left to
right, with a faint grey line tracing where the root joint travelled.

Run it after generating outputs for the query set:

    python -m paper.qualitative.fig_qualitative_failuremodes --outputs <folder>

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
from matplotlib.patches import FancyArrowPatch

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

# (failure mode, method label, colour, output folder), in the order the paper describes them.
ROWS = [
    ('Latent collapse',          r'AL-Flow',
     '#2ca02c', 'al_flow'),
    ('Decoder collapse',         r'AnyTop',
     '#d62728', 'anytop'),
    ('Source-conditional (partial)', r'MoReFlow-T',
     '#ff7f0e', 'moreflow_t'),
    ('Collapsed variation',      r'ACE-I',
     '#1f77b4', 'ace_i'),
]

QUERY_IDS = [76, 77, 79]  # three Bird attack clips retargeted to KingCobra
PHASES = [0.10, 0.50, 0.90]      # three poses per output cell
SRC_PHASES = [0.10, 0.50, 0.90]  # and per source cell

SOURCE_COLOR = '#222222'


def setup_mpl():
    plt.rcParams.update({
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{amsmath}\usepackage{amssymb}',
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman'],
        'font.size': 8,
        'axes.unicode_minus': False,
        'pdf.fonttype': 42,
    })


def load_query_meta(qid):
    m = json.loads(QUERY_SET.read_text())
    for q in m['queries']:
        if q['query_id'] == qid:
            return q
    raise ValueError(f'query_id {qid} not in manifest')


def get_positions(motion, n_joints):
    """[T, J, 13] -> [T, n_joints, 3] root-relative positions."""
    return motion[:, :n_joints, 0:3].astype(np.float32).copy()


def best_view(pos):
    """Pick a 2D projection (which two of XYZ) that maximizes pose extent."""
    extents = np.ptp(pos.reshape(-1, 3), axis=0)
    if extents[1] >= max(extents[0], extents[2]) * 0.5:
        h = 0 if extents[0] > extents[2] else 2
        return h, 1
    else:
        order = np.argsort(extents)[::-1]
        return order[0], order[1]


def best_view_aggregate(pos_list):
    """Pick a single projection across multiple motions of the same skeleton."""
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
                axes_override=None,
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
                    color=root_color, linewidth=0.4, alpha=0.50, zorder=1)
        # The Bird source has 61 joints and the KingCobra output 19: thin lines and
        # small markers for the source, thicker ones for the output.
        is_source = (color == SOURCE_COLOR)
        line_w = 0.55 if is_source else 0.95
        marker_s = 1.6 if is_source else 5.0
        for j in range(p.shape[0]):
            pp = parents[j]
            if pp >= 0 and pp < p.shape[0] and pp != j:
                ax.plot([p[j, h_axis] + x_off, p[pp, h_axis] + x_off],
                        [p[j, v_axis], p[pp, v_axis]],
                        color=color, linewidth=line_w, alpha=alphas[si],
                        solid_capstyle='round', zorder=3)
        ax.scatter(p[:, h_axis] + x_off, p[:, v_axis],
                   c=color, s=marker_s, alpha=alphas[si], edgecolors='none',
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

    # Pre-load source motions and metadata
    sources = []
    for qid in QUERY_IDS:
        q = load_query_meta(qid)
        src_motion = np.load(MOTION_DIR / q['src_fname'])
        n_a = len(cond[q['skel_a']]['parents'])
        src_pos = get_positions(src_motion, n_a)
        n_b = len(cond[q['skel_b']]['parents'])
        sources.append({
            'qid': qid,
            'src_pos': src_pos,
            'src_parents': cond[q['skel_a']]['parents'],
            'tgt_parents': cond[q['skel_b']]['parents'],
            'n_b': n_b,
            'src_fname': q['src_fname'],
        })

    # One viewing direction shared by the three source clips
    src_axes = best_view_aggregate([s['src_pos'] for s in sources])

    # Load every output first, so one viewing direction can be chosen for all of them.
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
    n_cols = 6  # 3 source + 3 output
    panel_w, panel_h = 1.55, 1.25
    # Extra width on the left holds the row labels.
    fig_w = n_cols * panel_w + 1.6
    fig_h = n_rows * panel_h + 0.6
    fig = plt.figure(figsize=(fig_w, fig_h))

    # Three source columns, a narrow column holding the arrow, three output columns
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(n_rows, 7, figure=fig,
                  width_ratios=[1.0, 1.0, 1.0, 0.18, 1.0, 1.0, 1.0],
                  wspace=0.05, hspace=0.12,
                  left=0.16, right=0.995, top=0.90, bottom=0.04)

    for ri, (mode_label, method_label, color, out_dir) in enumerate(ROWS):
        for ci_src, sd in enumerate(sources):
            ax = fig.add_subplot(gs[ri, ci_src])
            frames = sample_phase_frames(sd['src_pos'].shape[0], SRC_PHASES)
            x_min, x_max, y_min, y_max = draw_pose_strip(
                ax, sd['src_pos'], sd['src_parents'],
                SOURCE_COLOR, frames, axes_override=src_axes)
            ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
            style_strip_axis(ax)

        # Output cells
        for ci_out, sd in enumerate(sources):
            ax = fig.add_subplot(gs[ri, 4 + ci_out])
            pos = method_outputs[ri][sd['qid']]
            if pos is not None:
                frames = sample_phase_frames(pos.shape[0], PHASES)
                x_min, x_max, y_min, y_max = draw_pose_strip(
                    ax, pos, sd['tgt_parents'], color, frames,
                    axes_override=tgt_axes)
                ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
            else:
                ax.text(0.5, 0.5, 'missing', transform=ax.transAxes,
                        ha='center', va='center', fontsize=6, color='gray')
            style_strip_axis(ax)

        # Row label at the left edge, centred on the row
        y_row = 0.04 + (n_rows - 1 - ri + 0.5) / n_rows * (0.90 - 0.04)
        fig.text(0.005, y_row,
                 rf'\textbf{{{mode_label}}}' + '\n' + rf'{method_label}',
                 fontsize=8.5, color=color, va='center', ha='left',
                 rotation=0)

    # An arrow from source to output on every row
    for ri in range(n_rows):
        ax_g = fig.add_subplot(gs[ri, 3])
        ax_g.set_xticks([]); ax_g.set_yticks([])
        ax_g.set_xlim(0, 1); ax_g.set_ylim(0, 1)
        for s in ax_g.spines.values():
            s.set_visible(False)
        ax_g.axvline(0.5, color='#999999', linewidth=0.6, ymin=0.05, ymax=0.95)
        arr = FancyArrowPatch((0.10, 0.5), (0.90, 0.5),
                               arrowstyle='->', mutation_scale=12,
                               color='#444444', linewidth=0.8,
                               transform=ax_g.transAxes)
        ax_g.add_patch(arr)

    # Headers over the two halves, and a small label over every column so the three
    # source strips do not read as one long strip.
    left, right = 0.16, 0.995
    fig.text(left + (right - left) * (1.5 / 7.18), 0.945,
             r'\textbf{Source clip} (Bird, attack)',
             ha='center', va='bottom', fontsize=8.5)
    fig.text(left + (right - left) * (5.5 / 7.18), 0.945,
             r'\textbf{Output} on KingCobra target skeleton',
             ha='center', va='bottom', fontsize=8.5)
    ratios = np.array([1.0, 1.0, 1.0, 0.18, 1.0, 1.0, 1.0], dtype=float)
    edges = np.concatenate([[0.0], np.cumsum(ratios)])
    centers = (edges[:-1] + ratios / 2.0) / ratios.sum()
    for idx, label in zip([0, 1, 2], [r'source 1', r'source 2', r'source 3']):
        fig.text(left + (right - left) * centers[idx], 0.912,
                 label, ha='center', va='bottom', fontsize=6.8, color='#555555')
    for idx, label in zip([4, 5, 6], [r'output 1', r'output 2', r'output 3']):
        fig.text(left + (right - left) * centers[idx], 0.912,
                 label, ha='center', va='bottom', fontsize=6.8, color='#555555')

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = OUT_DIR / 'app_qualitative_failure_modes.pdf'
    out_png = OUT_DIR / 'app_qualitative_failure_modes.png'
    fig.savefig(out_pdf, dpi=300, bbox_inches='tight')
    fig.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_pdf.name} + {out_png.name}')


if __name__ == '__main__':
    main()
