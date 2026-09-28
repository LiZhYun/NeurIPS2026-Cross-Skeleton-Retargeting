"""Rebuild every figure of the paper that plots measured numbers.

Each point, bar and interval is read from results/, so a reader can follow any mark in
a figure back to the measurement it came from.

    python -m paper.make_figures                  # writes paper/output/figures/
    python -m paper.make_figures --only main_sif_1891 app_effective_rank

Most figures typeset their text with LaTeX, so a TeX installation has to be on the path.
Each figure is written twice: a PDF at 300 dots per inch for the paper and a PNG at 200
for quick viewing.

The figures that draw skeletons need the motions each method produced, which this
repository does not ship; their scripts are in paper/qualitative/.
"""
import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
PDF_DPI, PNG_DPI = 300, 200

# One colour per method family, shared by every figure.
COL = {
    'ANCHOR': '#D4A017', 'random': '#7f7f7f',
    'random-same-cluster': '#7f7f7f', 'random-same-exact-action': '#7f7f7f',
    'ACE': '#1f77b4', 'ACE-T': '#1f77b4', 'ACE-I': '#6baed6',
    'ACE-no-source-latent': '#74c0fc', 'ACE-no-adversarial-loss': '#9ecae1',
    'MoReFlow': '#ff7f0e', 'MoReFlow-T': '#ff7f0e', 'MoReFlow-I': '#ffbb78',
    'AL-Flow': '#2ca02c', 'AL-Flow-Src': '#98df8a', 'AL-Flow-Src-G': '#5fb15f',
    'AnyTop': '#d62728', 'DPG': '#9467bd', 'DPG-SB-v3': '#9467bd',
    'M2M': '#8c564b', 'Motion2Motion-Direct': '#8c564b', 'Motion2Motion-BVH': '#c49c94',
    'oracle': '#000000',
}
FAMILY = {
    'ACE-T': 'ACE', 'ACE-I': 'ACE-I', 'MoReFlow-T': 'MoReFlow', 'MoReFlow-I': 'MoReFlow-I',
    'AL-Flow': 'AL-Flow', 'AL-Flow-Src': 'AL-Flow-Src', 'AL-Flow-Src-G': 'AL-Flow-Src-G',
    'AnyTop': 'AnyTop', 'DPG-SB-v3': 'DPG-SB-v3',
    'Motion2Motion-Direct': 'Motion2Motion-Direct', 'Motion2Motion-BVH': 'Motion2Motion-BVH',
    'ANCHOR': 'ANCHOR', 'random-same-cluster': 'random', 'random-same-exact-action': 'random',
}
RC = {
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsmath}\usepackage{amssymb}\usepackage{bm}',
    'font.family': 'serif', 'font.serif': ['Computer Modern Roman'], 'font.size': 8,
    'axes.labelsize': 8, 'axes.titlesize': 8.5, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 7, 'axes.unicode_minus': False, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 200,
}
# The main SIF figure uses matplotlib's own Computer Modern fonts instead of LaTeX.
RC_NO_TEX = dict(RC, **{'text.usetex': False, 'font.serif': ['cmr10'],
                        'mathtext.fontset': 'cm', 'axes.formatter.use_mathtext': True})


def tex_safe(s):
    """Escape LaTeX specials in strings that come from result-file keys."""
    if not isinstance(s, str) or '\\' in s:
        return s
    return s.replace('&', r'\&').replace('%', r'\%').replace('#', r'\#').replace('_', r'\_')


class Figures:
    def __init__(self, results, out):
        self.results, self.out = Path(results), Path(out)
        self.out.mkdir(parents=True, exist_ok=True)

    def read(self, rel):
        return json.loads((self.results / rel).read_text())

    def save(self, fig, name):
        fig.savefig(self.out / f'{name}.pdf', bbox_inches='tight', dpi=PDF_DPI)
        fig.savefig(self.out / f'{name}.png', bbox_inches='tight', dpi=PNG_DPI)
        plt.close(fig)
        path = (self.out / f'{name}.pdf').resolve()
        print('wrote', path.relative_to(ROOT) if ROOT in path.parents else path)

    # ---------------------------------------------------------------- helpers

    def sif_intersection(self):
        """SIF and output variation of the 14 methods on the 37-triple intersection."""
        return self.read('truebones/sif_37_intersection.json')['methods']

    # ------------------------------------------------------------ main figures

    def main_sif_1891(self):
        """SIF of the 14 methods on the 1,891-triple evaluation, raw and length-controlled."""
        plt.rcParams.update(RC_NO_TEX)
        d = self.read('truebones/sif_1891.json')
        rows = {}
        for mode in ('raw', 'length_controlled'):
            for r in d['scoring'][mode]:
                if r['method'] == 'ACE-no-source-latent':   # reported in the appendix only
                    continue
                rows.setdefault(r['method'], {})[mode] = r
        names = sorted(rows, key=lambda m: rows[m]['length_controlled']['sif'])

        fig, ax = plt.subplots(figsize=(6.75, 3.3))
        ax.axvspan(-0.10, 0.10, color='#E5E7EB', alpha=0.85, zorder=0, lw=0)
        ax.axvline(0, color='#888888', lw=0.7, ls='--', alpha=0.7, zorder=1)
        for i, m in enumerate(names):
            for mode, dy, marker, face in (('raw', 0.17, 'o', COL[m]),
                                           ('length_controlled', -0.17, 's', 'white')):
                r = rows[m][mode]
                lo, hi = r['ci95']
                ax.errorbar(r['sif'], i + dy, xerr=[[r['sif'] - lo], [hi - r['sif']]],
                            fmt=marker, ms=4.2, mfc=face, mec=COL[m], mew=1.0, ecolor=COL[m],
                            elinewidth=0.8, capsize=1.8, zorder=3)
            hi_all = max(rows[m]['raw']['ci95'][1], rows[m]['length_controlled']['ci95'][1])
            ax.text(hi_all + 0.02, i, f"$n={rows[m]['raw']['n_triples']:,}$".replace(',', '{,}'),
                    ha='left', va='center', fontsize=6, color='#333333')
        ax.set_yticks(np.arange(len(names)))
        ax.set_yticklabels(names)
        ax.set_ylim(-0.7, len(names) - 0.3)
        ax.set_xlim(-0.20, 0.72)
        ax.set_xlabel('Source-Instance Fidelity')
        ax.text(0.0, len(names) - 0.45, 'source-blind floor', ha='center', va='bottom',
                fontsize=6, color='#555555')
        h1 = ax.errorbar([], [], xerr=[], fmt='o', ms=4.2, mfc='#444444', mec='#444444',
                         ecolor='#444444', label='raw')
        h2 = ax.errorbar([], [], xerr=[], fmt='s', ms=4.2, mfc='white', mec='#444444',
                         ecolor='#444444', label='length-controlled')
        ax.legend(handles=[h1, h2], loc='lower right', frameon=False)
        ax.set_title('Source-Instance Fidelity on 1,891 Truebones triples, 14 methods', pad=4)
        ax.grid(True, axis='x', alpha=0.3, lw=0.4)
        ax.set_axisbelow(True)
        self.save(fig, 'main_sif_1891')

    def main_label_only_audit(self):
        """ANCHOR against the two label-only baselines, overall and on held-out queries."""
        plt.rcParams.update(RC)
        d = self.read('action_auc/label_only_audit.json')
        methods = ['ANCHOR', 'random-same-cluster', 'random-same-exact-action']
        overall, held = [], []
        for m in methods:
            row = next(r for r in d['rows'] if r['method'] == m)
            overall.append(row['overall_cluster_auc'])
            held.append(row['held_out_cluster_auc'])

        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        x = np.arange(len(methods))
        width = 0.35
        colors = [COL['ANCHOR'], COL['random'], COL['random']]
        ax.bar(x - width / 2, overall, width, color=colors, edgecolor='black',
               linewidth=0.6, label='Overall')
        ax.bar(x + width / 2, held, width, color=colors, edgecolor='black',
               linewidth=0.6, alpha=0.55, hatch='//', label='Held-out')
        ax.axhline(0.5, color='#444444', linewidth=0.6, linestyle=':', alpha=0.6, zorder=0)
        ax.text(2.45, 0.525, 'Random AUC = 0.5', fontsize=6, color='#444444',
                ha='right', va='bottom')
        for i in range(len(methods)):
            ax.text(i - width / 2, overall[i] + 0.012, f'{overall[i]:.3f}',
                    ha='center', va='bottom', fontsize=6.0, linespacing=1.0)
            ax.text(i + width / 2, held[i] + 0.012, f'{held[i]:.3f}',
                    ha='center', va='bottom', fontsize=6.0, linespacing=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(['ANCHOR', 'Random-same-\ncluster', 'Random-same-\nexact-action'],
                           rotation=0, fontsize=7)
        ax.set_ylabel('Cluster-tier AUC')
        ax.set_ylim(0.45, 1.0)
        ax.legend(loc='upper left', fontsize=6.5, framealpha=0.9)
        ax.set_title('Action-level audit', loc='center', pad=4)
        ax.grid(True, axis='y', alpha=0.25, linewidth=0.4)
        ax.set_axisbelow(True)
        self.save(fig, 'main_label_only_audit')

    LSIF_LABEL = {'ACE-I': 'ACE-I (3 seeds)', 'ACE-T': 'ACE-T (3 seeds)',
                  'ACE-no-source-latent': r'ACE trained with $z_{\text{src}}=0$',
                  'MoReFlow-T': 'MoReFlow-T',
                  'ACE-no-adversarial-loss': 'ACE without adversarial loss (3 seeds)',
                  'AL-Flow-Src': 'AL-Flow-Src', 'MoReFlow-I': 'MoReFlow-I',
                  'AL-Flow-Src-G': 'AL-Flow-Src-G', 'AnyTop': 'AnyTop', 'AL-Flow': 'AL-Flow'}
    # label position relative to each point: (dx, dy, horizontal and vertical alignment)
    LSIF_OFFSET = {'ACE-I': (-0.010, -0.014, 'right', 'top'),
                   'ACE-T': (0.012, -0.004, 'left', 'top'),
                   'ACE-no-source-latent': (-0.008, 0.016, 'right', 'bottom'),
                   'MoReFlow-T': (0.012, 0.0, 'left', 'center'),
                   'ACE-no-adversarial-loss': (0.012, 0.0, 'left', 'center'),
                   'AL-Flow-Src': (0.012, 0.0, 'left', 'center'),
                   'MoReFlow-I': (-0.012, 0.0, 'right', 'center'),
                   'AL-Flow-Src-G': (-0.012, 0.0, 'right', 'center'),
                   'AnyTop': (-0.012, 0.012, 'right', 'bottom'),
                   'AL-Flow': (0.012, 0.0, 'left', 'center')}

    def app_latent_sif(self):
        """SIF on the decoded motion against SIF on the latent codes, one point per method."""
        plt.rcParams.update(RC)
        d = self.read('latents/latent_sif.json')
        fig, ax = plt.subplots(figsize=(5.8, 3.6))
        ax.set_xlim(-0.30, 0.56)
        ax.set_ylim(-0.12, 0.52)
        ax.axhline(0, color='#bbbbbb', linewidth=0.5, linestyle='--', alpha=0.7, zorder=0)
        ax.axvline(0, color='#bbbbbb', linewidth=0.5, linestyle='--', alpha=0.7, zorder=0)
        by_key = {r['key']: r for r in d['rows']}
        for key, label in self.LSIF_LABEL.items():
            r = by_key[key]
            sif = statistics.mean(d['runs'][k]['sif'] for k in r['runs'])
            lsif = statistics.mean(d['runs'][k]['latent_sif'] for k in r['runs'])
            ax.scatter(lsif, sif, s=52, color=COL[key], edgecolor='black', linewidth=0.7, zorder=3)
            dx, dy, ha, va = self.LSIF_OFFSET[key]
            ax.annotate(tex_safe(label), (lsif, sif), xytext=(lsif + dx, sif + dy), fontsize=6.0,
                        ha=ha, va=va, color='#1a1a1a', zorder=4)
        ax.set_xlabel(r'L-SIF $\rho$ (on the latent codes)')
        ax.set_ylabel(r'SIF $\rho$ (on the decoded motion)')
        ax.set_title('SIF against latent SIF', loc='left', pad=4)
        ax.grid(True, alpha=0.25, linewidth=0.4)
        ax.set_axisbelow(True)
        self.save(fig, 'app_latent_sif')

    # -------------------------------------------------------- appendix figures

    ROTATION_RUNS = [('ACE-T-seed42', 'ACE-T, seed 42', 'ACE'),
               ('ACE-T-seed43', 'ACE-T, seed 43', 'ACE'),
               ('ACE-T-seed44', 'ACE-T, seed 44', 'ACE'),
               ('ACE-I-seed42', 'ACE-I, seed 42', 'ACE-I'),
               ('ACE-I-seed43', 'ACE-I, seed 43', 'ACE-I'),
               ('ACE-I-seed44', 'ACE-I, seed 44', 'ACE-I'),
               ('ACE-no-source-latent-seed42',
                r'ACE trained with $z_{\text{src}}=0$', 'ACE-no-source-latent'),
               ('ACE-no-adversarial-loss-seed42',
                'ACE without adversarial loss, seed 42', 'ACE-no-adversarial-loss'),
               ('ACE-no-adversarial-loss-seed43',
                'ACE without adversarial loss, seed 43', 'ACE-no-adversarial-loss'),
               ('ACE-no-adversarial-loss-seed44',
                'ACE without adversarial loss, seed 44', 'ACE-no-adversarial-loss'),
               ('MoReFlow-T', 'MoReFlow-T', 'MoReFlow'),
               ('MoReFlow-I', 'MoReFlow-I', 'MoReFlow-I'),
               ('AL-Flow', 'AL-Flow', 'AL-Flow'),
               ('AL-Flow-Src', 'AL-Flow-Src', 'AL-Flow-Src'),
               ('AL-Flow-Src-G', 'AL-Flow-Src-G', 'AL-Flow-Src-G')]

    def app_rotation_test(self):
        """Latent rotations against matched-magnitude noise, for every trained run."""
        plt.rcParams.update(RC)
        labels, ratios, colors = [], [], []
        for run, label, family in self.ROTATION_RUNS:
            labels.append(label)
            ratios.append(self.read(f'latents/perturbation/{run}.json')['aggregate']['ratio_mean'])
            colors.append(COL[family])
        fig, ax = plt.subplots(figsize=(6.75, 3.2))
        x = np.arange(len(labels))
        ax.bar(x, ratios, color=colors, edgecolor='black', linewidth=0.4, width=0.7)
        ax.axhline(1.0, color=COL['random'], linestyle='--', linewidth=0.8,
                   label='noise floor (ratio = 1)')
        for i, r in enumerate(ratios):
            ax.text(i, r + 0.02 * max(ratios), f'{r:.1f}', ha='center', fontsize=6, color='#222222')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=7)
        ax.set_ylabel('change under rotation / change under noise')
        ax.set_ylim(0, 1.12 * max(ratios))
        ax.legend(loc='upper left', fontsize=7)
        ax.set_title('Latent rotations versus noise of the same size',
                     loc='left', pad=4)
        ax.grid(True, axis='y', alpha=0.3, linewidth=0.4)
        ax.set_axisbelow(True)
        self.save(fig, 'app_rotation_test')

    def app_cross_seed_alignment(self):
        """How much of the cross-seed latent difference a single rotation explains."""
        plt.rcParams.update(RC)
        groups = [('ACE-T', 'cross_seed_alignment_ace_t.json', COL['ACE']),
                  ('ACE-I', 'cross_seed_alignment_ace_i.json', COL['ACE-I']),
                  ('ACE without adversarial loss',
                   'cross_seed_alignment_no_adversarial_loss.json', COL['ACE-no-adversarial-loss'])]
        rows = []
        for label, fname, color in groups:
            a = self.read(f'latents/{fname}')['seed_pair_alignment']
            for lo, hi in (('42', '43'), ('42', '44'), ('43', '44')):
                x = a[f'seed{lo}_vs_seed{hi}']
                rows.append((label, f'seeds {lo} and {hi}',
                             x['fraction_explained_by_rotation'],
                             x['row_cosine_after_rotation'], color))
        fig, ax = plt.subplots(figsize=(3.4, 2.5))
        y = np.arange(len(rows))
        colors = [r[4] for r in rows]
        ax.scatter([r[2] for r in rows], y, color=colors, s=60, edgecolor='black',
                   linewidth=0.6, label='share explained by rotation')
        ax.scatter([r[3] for r in rows], y, color=colors, s=40, edgecolor='black',
                   linewidth=0.6, marker='D', alpha=0.7, label='cosine after rotation')
        ax.axvline(1.0, color='#444444', linestyle=':', linewidth=0.5, alpha=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels([tex_safe(f'{r[0]}: {r[1]}') for r in rows], fontsize=6.5)
        ax.set_xlim(0.3, 1.05)
        ax.set_xlabel('agreement between the two seeds')
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=6.5)
        ax.set_title('Cross-seed Procrustes alignment', loc='left', pad=4)
        ax.grid(True, axis='x', alpha=0.3, linewidth=0.4)
        ax.set_axisbelow(True)
        self.save(fig, 'app_cross_seed_alignment')

    RANK_COL = {'ACE-no-adversarial-loss-seed42': COL['ACE-no-adversarial-loss'],
                'ACE-T-seed42': COL['ACE'], 'ACE-I-seed42': COL['ACE-I'],
                'AL-Flow': COL['AL-Flow'], 'MoReFlow-I': COL['MoReFlow-I'],
                'MoReFlow-T': COL['MoReFlow'], 'AnyTop encoder z': COL['AnyTop']}
    RANK_LABEL = {'ACE-no-adversarial-loss-seed42': 'ACE without adversarial loss (seed 42)',
                  'ACE-T-seed42': 'ACE-T (seed 42)', 'ACE-I-seed42': 'ACE-I (seed 42)',
                  'AL-Flow': 'AL-Flow', 'MoReFlow-I': 'MoReFlow-I',
                  'MoReFlow-T': 'MoReFlow-T', 'AnyTop encoder z': r'AnyTop encoder $z$'}

    def app_effective_rank(self):
        """Effective rank of each model's latent distribution."""
        plt.rcParams.update(RC)
        rows = self.read('latents/effective_rank.json')['rows']
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        for i, r in enumerate(rows):
            col = self.RANK_COL[r['model']]
            ax.hlines(i, 0.8, r['effective_rank'], color=col, linewidth=2)
            ax.scatter(r['effective_rank'], i, s=60, color=col, edgecolor='black',
                       linewidth=0.6, zorder=3)
            ax.text(r['effective_rank'] * 1.25, i, f"{r['effective_rank']:.2f}", va='center',
                    fontsize=6.5, color='#222222')
        ax.axvline(256, color='#444444', linestyle=':', linewidth=0.6,
                   label=r'Latent size $d=256$')
        ax.set_yticks(np.arange(len(rows)))
        ax.set_yticklabels([self.RANK_LABEL[r['model']] for r in rows], fontsize=6.5)
        ax.set_xlabel('effective rank')
        ax.set_xscale('log')
        ax.set_xlim(0.8, 320)
        ax.legend(loc='lower right', fontsize=6.5)
        ax.set_title('Effective rank of trained latent distribution', loc='left', pad=4)
        ax.grid(True, axis='x', alpha=0.3, linewidth=0.4, which='both')
        ax.set_axisbelow(True)
        self.save(fig, 'app_effective_rank')

    def app_rotation_test_vs_sif(self):
        """Rotation sensitivity of the decoder against SIF, run by run."""
        plt.rcParams.update(RC)
        inter = self.sif_intersection()
        pairs = [('ACE-T-seed42', 'ACE-T', 'ACE'), ('ACE-T-seed43', 'ACE-T', 'ACE'),
                 ('ACE-T-seed44', 'ACE-T', 'ACE'), ('ACE-I-seed42', 'ACE-I', 'ACE-I'),
                 ('ACE-I-seed43', 'ACE-I', 'ACE-I'), ('ACE-I-seed44', 'ACE-I', 'ACE-I'),
                 ('ACE-no-source-latent-seed42', 'ACE-no-source-latent', 'ACE-no-source-latent'),
                 ('ACE-no-adversarial-loss-seed42', 'ACE-no-adversarial-loss', 'ACE-no-adversarial-loss'),
                 ('ACE-no-adversarial-loss-seed43', 'ACE-no-adversarial-loss', 'ACE-no-adversarial-loss'),
                 ('ACE-no-adversarial-loss-seed44', 'ACE-no-adversarial-loss', 'ACE-no-adversarial-loss'),
                 ('MoReFlow-T', 'MoReFlow-T', 'MoReFlow'),
                 ('MoReFlow-I', 'MoReFlow-I', 'MoReFlow-I'),
                 ('AL-Flow', 'AL-Flow', 'AL-Flow'),
                 ('AL-Flow-Src', 'AL-Flow-Src', 'AL-Flow-Src'),
                 ('AL-Flow-Src-G', 'AL-Flow-Src-G', 'AL-Flow-Src-G')]
        fig, ax = plt.subplots(figsize=(3.6, 2.6))
        for run, method, family in pairs:
            ratio = self.read(f'latents/perturbation/{run}.json')['aggregate']['ratio_mean']
            ax.scatter(ratio, inter[method]['sif'], color=COL.get(family, '#444444'), s=42,
                       edgecolor='black', linewidth=0.5, alpha=0.85)
        ax.axhline(0, color='#888888', linewidth=0.5, linestyle='--', alpha=0.7)
        ax.axvline(1, color='#888888', linewidth=0.5, linestyle='--', alpha=0.7)
        # the upright line marks a ratio of one, where a rotation does no more than noise
        ax.text(1.05, -0.09, 'noise floor', fontsize=6, color='#666666')
        ax.set_xlabel('latent rotation / matched noise ratio')
        ax.set_ylabel(r'SIF $\rho$')
        ax.set_xscale('log')
        ax.set_xlim(0.8, 45)
        ax.set_ylim(-0.10, 0.60)
        handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=COL[f],
                          markeredgecolor='black', markersize=6, label=tex_safe(lab))
                   for f, lab in (('ACE', 'ACE-T'), ('ACE-I', 'ACE-I'),
                                  ('ACE-no-source-latent', r'ACE trained with $z_{\text{src}}=0$'),
                                  ('ACE-no-adversarial-loss', 'ACE without adversarial loss'),
                                  ('MoReFlow', 'MoReFlow-T'), ('MoReFlow-I', 'MoReFlow-I'),
                                  ('AL-Flow', 'AL-Flow'))]
        ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1.02, 0.5),
                  fontsize=5.5, ncol=1, framealpha=0.92)
        ax.set_title('Latent rotation test and SIF',
                     loc='left', pad=4)
        ax.grid(True, alpha=0.25, linewidth=0.4)
        ax.set_axisbelow(True)
        self.save(fig, 'app_rotation_test_vs_sif')

    def app_resampling_intervals(self):
        """The SIF interval of every method under four resampling schemes."""
        plt.rcParams.update(RC)
        d = self.read('truebones/sif_clustered_intervals.json')['methods']
        methods = sorted(d.items(),
                         key=lambda x: -round(x[1]['independent_triples']['sif'], 3))
        schemes = ['independent_triples', 'whole_source_skeletons',
                   'whole_target_skeletons', 'whole_actions']
        scheme_labels = ['each triple alone', 'by source skeleton', 'by target skeleton',
                         'by action']
        offset = {s: 0.18 - i * 0.12 for i, s in enumerate(schemes)}
        marker = dict(zip(schemes, ['o', 's', 'D', '^']))
        alpha = dict(zip(schemes, [1.0, 0.8, 0.7, 0.6]))
        fig, ax = plt.subplots(figsize=(6.75, 3.5))
        for i, (method, ds) in enumerate(methods):
            c = COL[FAMILY.get(method, 'random')]
            for s in schemes:
                lo, mid, hi = ds[s]['ci95_low'], ds[s]['sif'], ds[s]['ci95_high']
                ax.errorbar(mid, i + offset[s], xerr=[[mid - lo], [hi - mid]], ecolor=c,
                            elinewidth=0.6, capsize=2, alpha=alpha[s], fmt='none')
                ax.scatter(mid, i + offset[s], marker=marker[s], s=22, color=c,
                           edgecolor='black', linewidth=0.4, alpha=alpha[s], zorder=3)
        ax.axvline(0, color='#888888', linestyle='--', linewidth=0.5, alpha=0.7, zorder=0)
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels([m for m, _ in methods], fontsize=6.5)
        ax.set_xlabel(r'SIF $\rho$ (95\% CI under each resampling scheme)')
        ax.set_xlim(-0.78, 0.75)
        handles = [Line2D([0], [0], marker=marker[s], color='w', markerfacecolor='#444444',
                          markeredgecolor='black', markersize=5, label=scheme_labels[i])
                   for i, s in enumerate(schemes)]
        ax.legend(handles=handles, loc='upper right', fontsize=6.5, framealpha=0.9, ncol=2)
        ax.set_title('SIF intervals under four resampling schemes', loc='left', pad=4)
        ax.grid(True, axis='x', alpha=0.3, linewidth=0.4)
        ax.set_axisbelow(True)
        self.save(fig, 'app_resampling_intervals')

    def app_triples_per_method(self):
        """How many triples each method could be scored on."""
        plt.rcParams.update(RC)
        d = self.read('truebones/sif_triple_support.json')['per_method_own_triples']
        # most triples first; methods that cover the same number keep the fidelity ranking
        methods = sorted(d.items(), key=lambda x: (-x[1]['n_triples'], -x[1]['sif']))
        names = [m for m, _ in methods]
        ns = [v['n_triples'] for _, v in methods]
        fig, ax = plt.subplots(figsize=(6.75, 2.9))
        x = np.arange(len(names))
        ax.bar(x, ns, color=[COL[FAMILY[m]] for m in names], edgecolor='black',
               linewidth=0.4, width=0.7)
        ax.axhline(49, color='#444444', linestyle='--', linewidth=0.6, alpha=0.7,
                   label=r'$n=49$ data ceiling')
        for i, n in enumerate(ns):
            ax.text(i, n + 0.6, str(n), ha='center', fontsize=6.5)
        ax.set_xticks(x)
        ax.set_xticklabels([tex_safe(n) for n in names], rotation=35, ha='right', fontsize=7)
        ax.set_ylabel('triples evaluated')
        ax.set_ylim(30, 55)
        ax.legend(loc='upper right', fontsize=6.5)
        ax.set_title('Triples each method could be scored on', loc='left', pad=4)
        ax.grid(True, axis='y', alpha=0.3, linewidth=0.4)
        ax.set_axisbelow(True)
        self.save(fig, 'app_triples_per_method')

    SIF_R_OFFSET = {
        'ACE-I': (0.85, 0.020, 'right', 'bottom'), 'ACE-T': (1.10, 0.020, 'left', 'bottom'),
        'MoReFlow-T': (1.10, 0.020, 'left', 'bottom'),
        'MoReFlow-I': (1.10, -0.022, 'left', 'top'),
        'random-same-exact-action': (0.85, 0.020, 'right', 'bottom'),
        'random-same-cluster': (1.10, -0.022, 'left', 'top'),
        'AL-Flow-Src': (0.85, 0.020, 'right', 'bottom'),
        'Motion2Motion-Direct': (1.10, 0.020, 'left', 'bottom'),
        'ANCHOR': (0.85, -0.022, 'right', 'top'), 'AnyTop': (1.10, 0.020, 'left', 'bottom'),
        'AL-Flow-Src-G': (1.18, -0.022, 'left', 'top'),
        'AL-Flow': (0.85, -0.022, 'right', 'top'),
        'DPG-SB-v3': (0.85, 0.020, 'right', 'bottom'),
        'Motion2Motion-BVH': (1.10, -0.022, 'left', 'top')}

    def app_sif_vs_variation(self):
        """SIF against how much the outputs of a triple differ from one another."""
        plt.rcParams.update(RC)
        inter = self.sif_intersection()
        fig, ax = plt.subplots(figsize=(5.4, 3.0))
        for m in sorted(FAMILY, key=lambda m: -inter[m]['sif']):
            rho, r = inter[m]['sif'], inter[m]['variation']
            ax.scatter(r, rho, color=COL[FAMILY[m]], s=52, edgecolor='black', linewidth=0.5,
                       zorder=3)
            xf, yo, ha, va = self.SIF_R_OFFSET[m]
            ax.annotate(tex_safe(m), (r, rho), xytext=(r * xf, rho + yo), fontsize=5.5,
                        color='#222222', ha=ha, va=va, zorder=4)
        ax.axhline(0, color='#888888', linestyle='--', linewidth=0.5, alpha=0.7, zorder=0)
        ax.set_xscale('log')
        ax.set_xlabel(r'output variation $R$ (log scale)')
        ax.set_ylabel(r'SIF $\rho$')
        ax.set_xlim(0.04, 30)
        ax.set_ylim(-0.18, 0.70)
        ax.set_title(r'SIF $\rho$ against output variation $R$', loc='left', pad=4)
        ax.grid(True, alpha=0.3, linewidth=0.4)
        ax.set_axisbelow(True)
        self.save(fig, 'app_sif_vs_variation')

    def app_synthetic_calibration(self):
        """SIF on the synthetic environment where the true retargeting map is known."""
        plt.rcParams.update(RC)
        s = self.read('synthetic/sif_calibration.json')
        rows = [('oracle', 'Oracle\n(out = $T^*(x_a)$)', COL['oracle']),
                ('random clip from cell', 'Random clip\nfrom cell', COL['random']),
                ('random Gaussian noise', 'Random Gaussian\nnoise', COL['random'])]
        res = [s['outputs'][k] for k, _, _ in rows]
        rhos = [r['sif'] for r in res]
        fig, ax = plt.subplots(figsize=(4.5, 2.4))
        x = np.arange(len(rows))
        ax.bar(x, rhos, color=[c for _, _, c in rows], edgecolor='black', linewidth=0.5,
               width=0.55)
        ax.errorbar(x, rhos, yerr=[[r['sif'] - r['ci95'][0] for r in res],
                                   [r['ci95'][1] - r['sif'] for r in res]],
                    ecolor='black', elinewidth=0.6, capsize=3, fmt='none')
        for i, rho in enumerate(rhos):
            ax.text(i, rho + 0.04 if rho > 0.1 else 0.08, f'{rho:+.3f}', ha='center',
                    fontsize=6.5)
        ax.axhline(0, color='#888888', linewidth=0.5, linestyle='--', alpha=0.7)
        ax.axhline(1, color='#444444', linewidth=0.5, linestyle=':', alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([lab for _, lab, _ in rows], fontsize=7)
        ax.set_ylabel(r'SIF (95\% CI)')
        ax.set_ylim(-0.20, 1.15)
        ax.set_title(rf"Synthetic 2$\times$2 SIF calibration ({res[0]['n_groups']} groups)",
                     loc='left', pad=4)
        ax.grid(True, axis='y', alpha=0.25, linewidth=0.4)
        ax.set_axisbelow(True)
        self.save(fig, 'app_synthetic_calibration')

    # Rows of the per-cluster heatmap, with the labels the manuscript prints.
    PER_CLUSTER_ROWS = [('ANCHOR', 'ANCHOR'),
                        ('ANCHOR with predicted cluster', 'ANCHOR with predicted cluster'),
                        ('Q-Retrieval (Q-feature only)', 'Q-Retrieval (Q-feature only)'),
                        ('MoReFlow-T', 'MoReFlow-T'), ('ACE-T', 'ACE-T'),
                        ('MoReFlow-I', 'MoReFlow-I'), ('AL-Flow-Src-G', 'AL-Flow-Src-G'),
                        ('ACE-I', 'ACE-I'), ('DPG-SB-v3', 'DPG-SB-v3'), ('AnyTop', 'AnyTop'),
                        ('Motion2Motion-BVH', 'Motion2Motion-BVH')]
    CLUSTERS = ['locomotion', 'combat', 'idle', 'death', 'jump']

    def app_action_auc_by_group(self):
        """Held-out action-level AUC of every method, split by action group."""
        plt.rcParams.update(RC)
        auc = self.read('action_auc/fold_auc.json')['methods']
        data = np.array([[float(np.mean([
            auc[m][f]['procrustes']['per_cluster_test_test_cluster_tier'][c]['auc_ci'][1]
            for f in ('42', '43')])) for c in self.CLUSTERS] for m, _ in self.PER_CLUSTER_ROWS])
        fig, ax = plt.subplots(figsize=(5.8, 3.0))
        im = ax.imshow(data, cmap='viridis', aspect='auto', vmin=0.30, vmax=0.95)
        ax.set_xticks(range(len(self.CLUSTERS)))
        ax.set_xticklabels(self.CLUSTERS, fontsize=7)
        ax.set_yticks(range(len(self.PER_CLUSTER_ROWS)))
        ax.set_yticklabels([lab for _, lab in self.PER_CLUSTER_ROWS], fontsize=7)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center',
                        color='white' if data[i, j] < 0.65 else 'black', fontsize=6.5)
        cbar = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
        cbar.set_label('cluster-tier AUC', fontsize=7)
        cbar.ax.tick_params(labelsize=6.5)
        ax.set_title('Held-out action-level AUC by action group', loc='left', pad=4)
        self.save(fig, 'app_action_auc_by_group')

    def app_anchor_enumeration(self):
        """ANCHOR on the full enumeration, split by how many skeletons are held out."""
        plt.rcParams.update(RC)
        d = self.read('action_auc/enumeration.json')['subsets']
        rows = [('All', 'all', COL['ANCHOR']),
                ('In-distribution', 'in_distribution', COL['oracle']),
                ('Mixed', 'mixed', COL['MoReFlow']),
                ('Held-out', 'held_out', COL['AnyTop'])]
        aucs = [d[k]['auc_ci'][1] for _, k, _ in rows]
        los = [d[k]['auc_ci'][1] - d[k]['auc_ci'][0] for _, k, _ in rows]
        his = [d[k]['auc_ci'][2] - d[k]['auc_ci'][1] for _, k, _ in rows]
        fig, ax = plt.subplots(figsize=(3.6, 2.4))
        x = np.arange(len(rows))
        ax.bar(x, aucs, color=[c for _, _, c in rows], edgecolor='black', linewidth=0.5,
               width=0.6)
        ax.errorbar(x, aucs, yerr=[los, his], ecolor='black', elinewidth=0.6, capsize=3,
                    fmt='none')
        for i, (auc_, key) in enumerate(zip(aucs, [k for _, k, _ in rows])):
            ax.text(i, auc_ + 0.01, f"{auc_:.3f}\n($n={d[key]['n']:,}$)", ha='center',
                    va='bottom', fontsize=6)
        ax.axhline(0.5, color='#888888', linestyle=':', linewidth=0.5, alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([lab for lab, _, _ in rows], fontsize=7)
        ax.set_ylabel('cluster-tier AUC')
        ax.set_ylim(0.45, 1.0)
        ax.set_title('ANCHOR enumeration AUC by split', loc='left', pad=4)
        ax.grid(True, axis='y', alpha=0.3, linewidth=0.4)
        ax.set_axisbelow(True)
        self.save(fig, 'app_anchor_enumeration')

    MASTER_AUC_ROWS = [('Action oracle', 'oracle'), ('Self-positive reference', 'oracle'),
                       ('ANCHOR', 'ANCHOR'), ('Cluster-Classifier Retrieval', 'ANCHOR'),
                       ('Q-Retrieval (Q-feature only)', 'ANCHOR'),
                       ('MoReFlow-T', 'MoReFlow'), ('MoReFlow-I', 'MoReFlow-I'),
                       ('ACE-T', 'ACE'), ('ACE-I', 'ACE-I'),
                       ('Motion2Motion-Direct', 'Motion2Motion-Direct'),
                       ('Motion2Motion-BVH', 'Motion2Motion-BVH'),
                       ('AnyTop', 'AnyTop'), ('DPG-SB-v3', 'DPG-SB-v3'),
                       ('Random target skeleton (null)', 'random')]

    def app_action_auc_band(self):
        """Held-out action-level AUC of every method family, on one axis."""
        plt.rcParams.update(RC)
        auc = self.read('action_auc/fold_auc.json')['methods']
        rows = [(m, float(np.mean([auc[m][f]['procrustes']['cluster_tier_test_test_auc_ci'][1]
                                   for f in ('42', '43')])), fam)
                for m, fam in self.MASTER_AUC_ROWS]
        rows.sort(key=lambda r: r[1])
        fig, ax = plt.subplots(figsize=(6.75, 2.7))
        ax.axvline(0.5, color='#888888', linestyle=':', linewidth=0.5, alpha=0.6)
        for i, (_, a, fam) in enumerate(rows):
            ax.scatter(a, i, color=COL[fam], s=70, edgecolor='black', linewidth=0.5)
            ax.text(a + 0.012, i, f'{a:.3f}', va='center', fontsize=6, color='#222222')
        ax.set_yticks(np.arange(len(rows)))
        ax.set_yticklabels([r[0] for r in rows], fontsize=6.5)
        ax.set_xlabel('cluster-tier held-out AUC (Procrustes, mean of folds 42 + 43)')
        ax.set_xlim(0.3, 1.0)
        ax.set_title('Action-level AUC across methods (Procrustes)', loc='left', pad=4)
        ax.grid(True, axis='x', alpha=0.3, linewidth=0.4)
        ax.set_axisbelow(True)
        self.save(fig, 'app_action_auc_band')

    FIGURES = ['main_sif_1891', 'main_label_only_audit', 'app_latent_sif',
               'app_rotation_test', 'app_cross_seed_alignment', 'app_effective_rank',
               'app_rotation_test_vs_sif', 'app_resampling_intervals',
               'app_triples_per_method', 'app_sif_vs_variation',
               'app_synthetic_calibration', 'app_action_auc_by_group',
               'app_anchor_enumeration', 'app_action_auc_band']

    def all(self, only=None):
        for name in self.FIGURES:
            if only and name not in only:
                continue
            getattr(self, name)()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--results', default=str(ROOT / 'results'),
                    help='folder holding the result files')
    ap.add_argument('--out', default=str(ROOT / 'paper' / 'output' / 'figures'),
                    help='where to write the figures')
    ap.add_argument('--only', nargs='*',
                    help='names of the figures to rebuild; all of them by default')
    a = ap.parse_args()
    Figures(a.results, a.out).all(a.only)


if __name__ == '__main__':
    main()
