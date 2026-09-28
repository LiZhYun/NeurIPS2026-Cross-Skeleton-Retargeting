"""Rebuild every table of the paper from the result files.

Each number printed in the paper is read from results/ and formatted here, so a reader
can follow any figure in a table back to the measurement it came from.

    python -m paper.make_tables                 # writes paper/output/tables/
    python -m paper.make_tables --out somewhere

Each table is written as a bare `tabular`, with no surrounding `table` environment, so
the paper can place it in its own float. Two tables describe the methods and the theory
rather than measure anything; they are kept as written text in paper/static/ and copied
across, so the output folder holds a complete set. Each file is named as the paper's LaTeX
source includes it.
"""
import argparse
import json
import shutil
import statistics
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).resolve().parent / 'static'

# The fourteen evaluated methods. Tables rank methods by score, and this order
# decides who comes first when two methods score the same.
METHODS = ['AnyTop', 'ACE-T', 'ACE-I', 'AL-Flow', 'AL-Flow-Src', 'AL-Flow-Src-G',
           'MoReFlow-T', 'MoReFlow-I', 'Motion2Motion-BVH', 'ANCHOR',
           'Motion2Motion-Direct', 'random-same-cluster', 'random-same-exact-action',
           'DPG-SB-v3']
RUN_LABEL = {'ACE-no-adversarial-loss-seed42': 'ACE without adversarial loss (seed 42)',
             'ACE-T-seed42': 'ACE-T (seed 42)', 'ACE-I-seed42': 'ACE-I (seed 42)',
             'AL-Flow': 'AL-Flow', 'MoReFlow-I': 'MoReFlow-I', 'MoReFlow-T': 'MoReFlow-T',
             'AnyTop encoder z': 'AnyTop encoder $z$'}
ROBOT_CASES = [('true_retarget', 'True retargeted counterpart'),
               ('random_clip', 'Random same-group clip'),
               ('unpaired_objective', 'Unpaired objective'),
               ('averaging_objective', 'Averaging objective'),
               ('true_pair_model', 'Model trained on true pairs')]
ROBOTS = [('unitree_g1', 'Unitree G1'), ('booster_t1_29dof', 'Booster T1'),
          ('fourier_n1', 'Fourier N1'), ('stanford_toddy', 'Stanford Toddy'),
          ('engineai_pm01', 'EngineAI PM01'), ('pal_talos', 'PAL Talos')]


# ---------------------------------------------------------------- formatting

def num(x, d=3):
    """Signed value in math mode, with a plain zero when the value rounds to zero."""
    if abs(x) < 0.5 * 10 ** (-d):
        return f'{0:.{d}f}'
    return f'{x:+.{d}f}'.replace('-', '$-$').replace('+', '$+$')


def fmt(v, prec=3):
    return f'{v:.{prec}f}'


def interval(lo, mean, hi, prec=3):
    return f'{mean:.{prec}f}\\,[{lo:.{prec}f},\\,{hi:.{prec}f}]'


def pval(p):
    return '$<$0.001' if p < 0.001 else f'{p:.3f}'


def ci_num(x):
    return num(x, 3) if abs(x) < 0.005 else num(x, 2)   # keep the sign of bounds near zero


def reading(p_raw, p_len, r_raw, r_len):
    if p_raw >= 0.05 and p_len >= 0.05:
        return 'at floor'
    if abs(r_raw) < 0.10 and abs(r_len) < 0.10:
        return 'near floor'
    return 'above floor'


def tabular(spec, header, lines, footnote=None, small=None):
    body = f'\\begin{{tabular}}{{{spec}}}\n\\toprule\n{header}\n\\midrule\n'
    body += '\n'.join(lines) + '\n'
    if footnote:
        body += f'\\midrule\n{footnote}\n'
    body += '\\bottomrule\n\\end{tabular}\n'
    if small:
        body = '{\\%s\n' % small + body + '}\n'
    return body


def shown(path):
    """A path relative to the repository root when it lies inside it."""
    path = Path(path).resolve()
    return path.relative_to(ROOT) if ROOT in path.parents else path


def stacked(top, bottom):
    """A table header cell set on two lines."""
    return f'\\begin{{tabular}}[b]{{@{{}}r@{{}}}}{top}\\\\{bottom}\\end{{tabular}}'


class Tables:
    def __init__(self, results, out):
        self.results, self.out = Path(results), Path(out)
        self.out.mkdir(parents=True, exist_ok=True)

    def read(self, rel):
        return json.loads((self.results / rel).read_text())

    def write(self, name, body):
        (self.out / f'{name}.tex').write_text(body)
        print('wrote', shown(self.out / f'{name}.tex'))

    # ------------------------------------------------------- dataset and roster

    def dataset_statistics(self):
        s = self.read('truebones/dataset_statistics.json')
        rows = [
            ('Skeletons', f"{s['skeletons']}"),
            ('Total clips', f"{s['clips']}"),
            ('Coarse clusters', f"{s['action_groups']}"),
            ('Exact actions', f"{s['exact_actions']}"),
            ('Skeleton-action cells',
             f"{s['skeleton_action_cells']:,} ($70 \\times 90$)".replace(',', '{,}')),
            ('Occupied cells', f"{s['occupied_cells']} ({s['occupied_cells_pct']:.1f}\\%)"),
            ('Median clips per occupied cell',
             f"{s['median_clips_per_occupied_cell']:.0f}"),
            ('Source groups with at least three clips',
             f"{s['source_groups_with_three_clips']['groups']} "
             f"({s['source_groups_with_three_clips']['source_clips']} source clips)"),
            ('SIF triples, every eligible target (main evaluation)',
             f"{s['main_evaluation']['triples']:,} ({s['main_evaluation']['queries']:,} queries)".replace(',', '{,}')),
            ('SIF triples, one target per group (original evaluation)',
             f"{s['original_evaluation']['triples']} ({s['original_evaluation']['queries']} queries)"),
            ('Original cross-method intersection',
             f"{s['intersection']['triples']} triples "
             f"({s['intersection']['queries']} queries)"),
        ]
        self.write('tab_dataset_statistics',
                   tabular('lr', 'Quantity & Value \\\\',
                           [f'{q} & {v} \\\\' for q, v in rows]))

    # ---------------------------------------------------------- Truebones SIF

    def sif_main_1891(self):
        d = self.read('truebones/sif_1891.json')
        rows = {}
        for mode in ('raw', 'length_controlled'):
            for r in d['scoring'][mode]:
                if r['method'] in METHODS:
                    rows.setdefault(r['method'], {})[mode] = r
        order = sorted(rows, key=lambda m: -rows[m]['length_controlled']['sif'])
        lines = []
        for m in order:
            lines.append(self._sif_1891_row(m, rows[m], 'reading'))
        extra = {mode: next(r for r in d['scoring'][mode]
                            if r['method'] == 'ACE-no-source-latent')
                 for mode in ('raw', 'length_controlled')}
        lines.append('\\midrule')
        lines.append(self._sif_1891_row(
            'ACE trained with $z_{\\text{src}}=0$ (real $z_{\\text{src}}$ at generation)',
            extra, 'ablation'))
        self.write('tab_sif_main_1891', tabular(
            'lrlrlrrl',
            'Method & $n$ & Raw SIF [95\\% CI] & $p$ & Length-controlled SIF [95\\% CI] '
            '& $p$ & Variation & Reading \\\\', lines))

    @staticmethod
    def _sif_1891_row(label, r, last):
        o, f = r['raw'], r['length_controlled']
        if last == 'reading':
            last = reading(o['p_value'], f['p_value'], o['sif'], f['sif'])
        return (f"{label} & {o['n_triples']:,} & {num(o['sif'])} "
                f"[{ci_num(o['ci95'][0])}, {ci_num(o['ci95'][1])}] "
                f"& {pval(o['p_value'])} & {num(f['sif'])} "
                f"[{ci_num(f['ci95'][0])}, {ci_num(f['ci95'][1])}] "
                f"& {pval(f['p_value'])} & {o['variation']:.2f} & {last} \\\\"
                ).replace(',', '{,}', 1)

    def sif_original_49(self):
        c = self.read('truebones/sif_49.json')['scoring']
        v = self.read('truebones/sif_49_variation.json')['methods']
        get = {mode: {r['method']: r for r in c[mode]['truebones_49']}
               for mode in ('raw', 'length_controlled')}
        order = sorted(METHODS, key=lambda m: -get['length_controlled'][m]['sif'])
        lines = [f"{m} & {get['raw'][m]['n_triples']} & "
                 f"{num(get['raw'][m]['sif'])} & "
                 f"{pval(get['raw'][m]['p_value'])} & "
                 f"{num(get['length_controlled'][m]['sif'])} & "
                 f"{pval(get['length_controlled'][m]['p_value'])} & "
                 f"{v[m]['raw']['variation']:.2f} \\\\" for m in order]
        self.write('tab_sif_original_49', tabular(
            'lrrrrrr',
            'Method & $n$ & Raw SIF & $p$ & Length-controlled SIF & $p$ & Variation \\\\',
            lines))

    def clustered_ci(self):
        d = self.read('truebones/sif_clustered_intervals.json')['methods']
        lines = []
        # ranked by the interval midpoint as printed, so methods that tie at three
        # decimals keep the order of the result file
        for method, ds in sorted(d.items(),
                                 key=lambda x: -round(x[1]['independent_triples']['sif'], 3)):
            cells = []
            for k in ('independent_triples', 'whole_source_skeletons',
                      'whole_target_skeletons', 'whole_actions'):
                cell = interval(ds[k]['ci95_low'], ds[k]['sif'], ds[k]['ci95_high'])
                if ds[k]['ci95_low'] > 0:
                    cell = f'\\textbf{{{cell}}}'
                cells.append(cell)
            lines.append(f'{method} & ' + ' & '.join(cells) + ' \\\\')
        note = ('\\multicolumn{5}{p{0.96\\linewidth}}{\\footnotesize Bold lower CI excludes '
                'zero. Bootstrap $N=10{,}000$, seed $=42$. ACE-I is the only method whose '
                'lower CI excludes zero under all four resampling schemes; ACE-T and '
                'random-same-exact-action survive resampling by source skeleton and by '
                'target skeleton, but resampling by action drops their lower CI to or below '
                'zero because the 37-triple intersection is concentrated on a few actions.} \\\\')
        self.write('tab_resampling_intervals', tabular(
            'lrrrr',
            'Method & each triple alone & by source skeleton & by target skeleton & '
            'by action \\\\', lines, note, small='scriptsize'))

    def paired_ace_moreflow(self):
        rows = self.read('truebones/sif_paired_ace_moreflow.json')['comparisons']
        label = {'truebones_1891': '1{,}891 triples', 'truebones_49': '49 triples'}
        mlabel = {'raw': 'raw', 'length_controlled': 'length-controlled'}
        lines = []
        for s in ('truebones_1891', 'truebones_49'):
            for mode in ('raw', 'length_controlled'):
                cells = []
                for a in ('ACE-T', 'ACE-I'):
                    r = next(x for x in rows if x['evaluation_set'] == s
                             and x['scoring'] == mode and x['method'] == a)
                    cells.append(f"{num(r['difference'])} [{num(r['ci95'][0], 2)}, "
                                 f"{num(r['ci95'][1], 2)}], $p$={pval(r['p_value'])}")
                lines.append(f'{label[s]} & {mlabel[mode]} & {cells[0]} & {cells[1]} \\\\')
        self.write('tab_paired_ace_moreflow', tabular(
            'llll',
            'Evaluation & Scoring & ACE-T minus MoReFlow-T & ACE-I minus MoReFlow-I \\\\',
            lines))

    def sif_stability(self):
        h = self.read('truebones/sif_stability.json')['methods']
        ks = ['25', '100', '500', '1000']
        lines = [f"{m} & {h[m]['single_triple_spread']['3']['mean_sd']:.2f} & "
                 + ' & '.join(f"{h[m]['average_over_k_triples'][k]['sd']:.3f}" for k in ks) + ' \\\\'
                 for m in ('AnyTop', 'ACE-T', 'ACE-I', 'MoReFlow-T', 'random-same-exact-action')]
        header = (' & One triple & \\multicolumn{4}{c}{Average over $k$ triples} \\\\\n'
                  '\\cmidrule(lr){3-6}\n'
                  'Method & (three clips) & $k=25$ & $k=100$ & $k=500$ & $k=1{,}000$ \\\\')
        self.write('tab_sif_stability', tabular('lrrrrr', header, lines))

    def noise_control(self):
        g = self.read('truebones/sif_noise_control.json')['methods']
        lines = []
        for m in ('AnyTop', 'AL-Flow', 'AL-Flow-Src', 'AL-Flow-Src-G', 'DPG-SB-v3',
                  'MoReFlow-T', 'MoReFlow-I'):
            r = g[m]
            seeds = [r[k]['sif'] for k in ('seed42', 'seed43', 'seed44')]
            shared = num(r['shared_noise']['sif']) if 'shared_noise' in r else 'deterministic'
            lines.append(f"{m} & {' & '.join(num(s) for s in seeds)} & {num(np.mean(seeds))} "
                         f"$\\pm$ {np.std(seeds, ddof=1):.3f} & {shared} \\\\")
        header = (' & \\multicolumn{3}{c}{Independent noise, three seeds} & & \\\\\n'
                  '\\cmidrule(lr){2-4}\n'
                  'Method & seed 1 & seed 2 & seed 3 & Mean $\\pm$ sd & Shared noise \\\\')
        self.write('tab_noise_control', tabular('lrrrrr', header, lines))

    def pair_count_distribution(self):
        d = self.read('truebones/sif_triple_support.json')['source_clips_per_triple']
        # alphabetical by the short name each method carried while it was scored, which
        # is the order the paper prints
        order = ['ACE-I', 'ACE-T', 'AL-Flow', 'AL-Flow-Src', 'AL-Flow-Src-G', 'ANCHOR',
                 'AnyTop', 'DPG-SB-v3', 'Motion2Motion-BVH', 'Motion2Motion-Direct',
                 'MoReFlow-I', 'MoReFlow-T', 'random-same-exact-action', 'random-same-cluster']
        lines = [f"{m} & {d[m]['fewest']} & {d[m]['median']} & {d[m]['most']} & "
                 f"{d[m]['n_triples']} \\\\" for m in order]
        note = ('\\multicolumn{5}{p{0.94\\linewidth}}{\\footnotesize Min and median are '
                'uniformly 3 because the SIF benchmark requires $\\geq 3$ source clips per '
                'triple by design; max reflects the few triples in Truebones with 4 or 5 '
                'source clips. Variation across methods comes from the few queries that '
                'some baselines fail to score.} \\\\')
        self.write('tab_pair_count_distribution', tabular(
            'lrrrr',
            'Method & Min pairs & Median pairs & Max pairs & $n$ triples \\\\', lines, note))

    def truebones_four_measures(self):
        c = {r['method']: r
             for r in self.read('truebones/sif_49.json')['scoring']['raw']['truebones_49']}
        v = self.read('truebones/sif_49_variation.json')['methods']
        auc = {r['method']: r
               for r in self.read('truebones/action_auc_49.json')['sets']['truebones_49']}
        val = {r['method']: r
               for r in self.read('truebones/realism_49.json')['sets']['truebones_49']}
        order = sorted(METHODS, key=lambda m: -c[m]['sif'])
        lines = []
        for m in order:
            vr = val.get(m)
            real = (f"{vr['realistic_pct']:.1f}\\%"
                    if vr and vr.get('realistic_pct') is not None
                    and vr.get('n_outputs', 0) > vr.get('n_not_comparable', 0) else '--')
            lines.append(f"{m} & {num(c[m]['sif'])} & {auc[m]['auc_mean']:.3f} & "
                         f"{v[m]['raw']['variation']:.2f} & {real} \\\\")
        self.write('tab_truebones_four_measures', tabular(
            'lrrrr', 'Method & Raw SIF & Action AUC & Variation & Realistic \\\\', lines))

    # -------------------------------------------------------------- ACE controls

    def ace_source_removal(self):
        d = self.read('truebones/ace_source_removal.json')['cells']
        lines = []
        for source in ('real', 'removed', 'shuffled'):
            for length in ('copied', 'fixed'):
                cells = []
                for model in ('ACE-T', 'ACE-no-source-latent'):
                    c = d[f'{source}/{model}/{length}']
                    cells.append(f"{num(c['sif'])} & {pval(c['p_value'])} & "
                                 f"{c['variation']:.3f}")
                lines.append(f'{source} & {length} & {cells[0]} & {cells[1]} \\\\')
        header = (' & & \\multicolumn{3}{c}{ACE-T} & \\multicolumn{3}{c}'
                  '{Model trained with $z_{\\text{src}}=0$} \\\\\n'
                  '\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\n'
                  'Source at generation & Output length & SIF & $p$ & Variation & SIF & $p$ '
                  '& Variation \\\\')
        self.write('tab_ace_source_removal', tabular('llrrrrrr', header, lines))

    def ace_adv_matched(self):
        raw = self.read('truebones/ace_adversarial_matched_raw.json')
        fx = self.read('truebones/ace_adversarial_fixed_length.json')['evaluations']

        def cell(models, key):
            per_seed = models[key]['per_seed']
            seeds = [per_seed[str(s)] for s in (42, 43, 44)]
            sifs = [x['sif'] for x in seeds]
            spread = [x['variation'] for x in seeds]
            return f"{num(float(np.mean(sifs)))} & {min(spread):.3f}--{max(spread):.3f}"

        rows = [('raw, length copied', '49', raw['models'], raw['difference']),
                ('length fixed at generation', '49', fx['truebones_49']['models'],
                 fx['truebones_49']['difference']),
                ('length fixed at generation', '1{,}891', fx['truebones_1891']['models'],
                 fx['truebones_1891']['difference'])]
        lines = []
        for scoring, n, models, diff in rows:
            lines.append(f"{scoring} & {n} & {cell(models, 'ACE-I')} & "
                         f"{cell(models, 'ACE-no-adversarial-loss')} "
                         f"& {num(diff['mean_difference'])} [{num(diff['ci95'][0], 2)}, "
                         f"{num(diff['ci95'][1], 2)}] & {pval(diff['p_value'])} \\\\")
        header = (' & & \\multicolumn{2}{c}{With adversarial loss} & \\multicolumn{2}{c}'
                  '{Without} & & \\\\\n'
                  '\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\n'
                  'Scoring & Triples & SIF & Variation & SIF & Variation & '
                  'Difference [95\\% CI] & $p$ \\\\')
        self.write('tab_ace_adversarial_loss', tabular('lrrrrrlr', header, lines))

    def ace_stretch(self):
        d = self.read('truebones/ace_stretch_check.json')['methods']
        lines = [f"{m} & {d[m]['variation_raw']:.3f} & {d[m]['variation_stretched']:.3f} & "
                 f"{d[m]['identical_copies_stretched']:.3f} \\\\"
                 for m in ('ACE-T', 'ACE-I', 'MoReFlow-T', 'MoReFlow-I')]
        header = (' & \\multicolumn{2}{c}{Actual outputs} & Identical copies, \\\\\n'
                  '\\cmidrule(lr){2-3}\n'
                  'Method & Raw & Stretched to 64 frames & stretched to 64 frames \\\\')
        self.write('tab_ace_stretch', tabular('lrrr', header, lines))

    def ace_leakage_nn(self):
        d = self.read('truebones/ace_leakage_nearest_neighbour.json')['methods']
        lines = []
        for m in ('ACE-I', 'ACE-T'):
            x = d[m]
            n_pool = x['n_queries'] - x['n_closer_to_source']
            far = x['mean_distance_to_source']
            near = x['mean_distance_to_nearest_target_clip']
            lines.append(f"{m} & {x['n_queries']} & {n_pool} "
                         f"({100.0 - x['pct_closer_to_source']:.1f}\\%) & "
                         f"{far:.3f} & {near:.3f} & {far / near:.2f} \\\\")
        note = ('\\multicolumn{6}{p{0.94\\linewidth}}{\\footnotesize Distance computed in a '
                'four-dimensional kinematic feature space (centre-of-mass displacement, '
                'centre-of-mass variance, mean velocity, foot-contact density). 76--79\\% of '
                'ACE outputs are closer to a target-pool training clip than to the '
                "corresponding source clip, weakening source-memorisation as an explanation "
                "of ACE's SIF lift.} \\\\")
        self.write('tab_ace_nearest_clip', tabular(
            'lrlrrr',
            'Method & Queries & ' + stacked('Closer to the', 'target pool') + ' & '
            + stacked('Mean distance', 'to source') + ' & '
            + stacked('Mean distance', 'to target pool') + ' & Ratio \\\\', lines, note))

    # ------------------------------------------------------------------ latents

    def latent_rotation_ratios(self):
        groups = [
            ('ACE-T (3 seeds)', [('ACE-T-seed42', '42'), ('ACE-T-seed43', '43'),
                                 ('ACE-T-seed44', '44')]),
            ('ACE-I (3 seeds)', [('ACE-I-seed42', '42'), ('ACE-I-seed43', '43'),
                                 ('ACE-I-seed44', '44')]),
            ('ACE trained with $z_{\\text{src}}=0$', [('ACE-no-source-latent-seed42', '42')]),
            ('ACE without adversarial loss (3 seeds)',
             [('ACE-no-adversarial-loss-seed42', '42'),
              ('ACE-no-adversarial-loss-seed43', '43'),
              ('ACE-no-adversarial-loss-seed44', '44')]),
            ('MoReFlow-T', [('MoReFlow-T', '--')]),
            ('MoReFlow-I', [('MoReFlow-I', '--')]),
            ('AL-Flow', [('AL-Flow', '--')]),
            ('AL-Flow-Src', [('AL-Flow-Src', '--')]),
            ('AL-Flow-Src-G', [('AL-Flow-Src-G', '--')]),
        ]
        lines = []
        for gi, (label, members) in enumerate(groups):
            if gi > 0:
                lines.append('\\midrule')
            for mi, (run, seed) in enumerate(members):
                a = self.read(f'latents/perturbation/{run}.json')['aggregate']
                ratio = a['ratio_mean']
                if ratio > 5.0:
                    ratio_str, verdict = f'\\textbf{{{ratio:.2f}}}', 'non-degenerate (strong)'
                elif ratio > 1.5:
                    ratio_str, verdict = f'{ratio:.2f}', 'non-degenerate'
                else:
                    ratio_str, verdict = f'\\emph{{{ratio:.2f}}}', '\\emph{at noise floor}'
                lines.append(f"{label if mi == 0 else ''} & {seed} & "
                             f"{a['rel_div_rotation_mean']:.3f} & {a['rel_div_noise_mean']:.3f} "
                             f"& {ratio_str} & {verdict} \\\\")
        note = ('\\multicolumn{6}{p{0.96\\linewidth}}{\\footnotesize Each row reports the mean '
                'over the 130 queries of the 37-triple intersection, with 20 random $O(d)$ '
                'rotations and 20 matched-magnitude noise samples per query; the ratio is '
                'the mean of the per-query rotation-to-noise ratios. ACE without the '
                'adversarial loss is statistically indistinguishable from the noise floor '
                '(Wilcoxon $p>0.4$ for all three seeds); every other method exceeds the '
                'noise floor at $p<10^{-12}$.} \\\\')
        self.write('tab_rotation_test', tabular(
            'llrrrl',
            'Method & Seed & Change under rotation & Change under noise & Ratio & '
            'Verdict \\\\',
            lines, note))

    def cross_seed_alignment(self):
        groups = [('ACE-T', 'cross_seed_alignment_ace_t.json'),
                  ('ACE-I', 'cross_seed_alignment_ace_i.json'),
                  ('ACE without adversarial loss',
                   'cross_seed_alignment_no_adversarial_loss.json')]
        pairs = [('42', '43'), ('42', '44'), ('43', '44')]
        lines = []
        for gi, (label, fname) in enumerate(groups):
            if gi > 0:
                lines.append('\\midrule')
            a = self.read(f'latents/{fname}')['seed_pair_alignment']
            for lo, hi in pairs:
                x = a[f'seed{lo}_vs_seed{hi}']
                lines.append(f"{label} & {lo} vs {hi} & "
                             f"{x['fraction_explained_by_rotation']:.3f} & "
                             f"{x['row_cosine_after_rotation']:.3f} \\\\")
        note = ('\\multicolumn{4}{p{0.78\\linewidth}}{\\footnotesize \\emph{Share explained by '
                'rotation} is the fraction of cross-seed latent variance accounted for by the '
                'best $O(d)$ rotation (orthogonal Procrustes). \\emph{Cosine after rotation} is the '
                'mean per-row cosine similarity after that rotation. Both are computed on '
                'per-target-skeleton latents across the 130 queries of the 37-triple set, '
                'aggregated over 21 target skeletons.} \\\\')
        self.write('tab_cross_seed_alignment', tabular(
            'llrr', 'Group & Seed pair & Share explained by rotation & Cosine after rotation \\\\',
            lines, note))

    def effective_rank(self):
        rows = self.read('latents/effective_rank.json')['rows']
        best = max(rows, key=lambda r: r['effective_rank'])
        lines = []
        for r in rows:
            rank = f"{r['effective_rank']:.2f}"
            if r is best:
                rank = f'\\textbf{{{rank}}}'
            lines.append(f"{RUN_LABEL[r['model']]} & {r['shape'][0]:,}".replace(',', '{,}') +
                         f" $\\times$ {r['shape'][1]} & {r['spectral_flatness']:.4f} & "
                         f"{rank} \\\\")
        note = ('\\multicolumn{4}{p{0.85\\linewidth}}{\\footnotesize Effective rank computed '
                'on the per-method latent distribution over the 130 queries of the 37-triple '
                "intersection. AnyTop's encoder hidden state is an order of magnitude richer "
                'than the compressed latents shared by the MoReFlow-based models.} \\\\')
        self.write('tab_effective_rank', tabular(
            'lrrr',
            'Method & Sample shape & Spectral flatness & Effective rank $/$ 256 \\\\',
            lines, note, small='small'))

    def lsif_failure_modes(self):
        """SIF on the decoded motion next to SIF on the latent codes, one row per method."""
        from scipy.stats import pearsonr
        d = self.read('latents/latent_sif.json')
        runs = d['runs']
        rows = []
        for r in d['rows']:
            sif = statistics.mean(runs[k]['sif'] for k in r['runs'])
            lsif = statistics.mean(runs[k]['latent_sif'] for k in r['runs'])
            rows.append((sif, lsif, r['label']))
        rows.sort(key=lambda x: -x[0])
        lines = [f"{label} & {fmt(sif)} & {fmt(lsif)} \\\\" for sif, lsif, label in rows]
        names = sorted(runs)
        r, p = pearsonr([runs[k]['sif'] for k in names], [runs[k]['latent_sif'] for k in names])
        above = [k for k in names if runs[k]['latent_sif_ci95'][0] > 0]
        below = [k for k in names if runs[k]['latent_sif_ci95'][1] < 0]

        def join(ks):
            return ks[0] if len(ks) == 1 else ', '.join(ks[:-1]) + ' and ' + ks[-1]

        note = ('\\multicolumn{3}{p{0.94\\linewidth}}{\\footnotesize Cross-method Pearson '
                f"correlation between SIF and L-SIF: $r = {r:.3f}$, $p={p:.2f}$ across "
                f"$n={len(names)}$ method runs. The 95\\% interval of L-SIF lies above zero for "
                f"{join(above)} and below zero for {join(below)}; it includes zero for every other "
                'run.} \\\\')
        self.write('tab_latent_sif', tabular(
            'lrr', 'Method & SIF $\\rho$ & L-SIF $\\rho$ \\\\', lines, note))

    # ---------------------------------------------------------------- synthetic

    def synthetic_sif_calibration(self):
        s = self.read('synthetic/sif_calibration.json')
        labels = {'oracle': 'Oracle (output $= T^{*}(x_a)$)',
                  'random clip from cell': 'Random clip from cell',
                  'random Gaussian noise': 'Random Gaussian noise'}

        def var(x):
            if x < 1000:
                return f'{x:.2f}'
            e = int(np.floor(np.log10(x)))
            return f'${x / 10 ** e:.1f}{{\\times}}10^{{{e}}}$'

        lines = []
        for key, lab in labels.items():
            r = s['outputs'][key]
            cell = f"{num(r['sif'])} [{num(r['ci95'][0])}, {num(r['ci95'][1])}]"
            if key == 'oracle':
                cell = f'\\textbf{{{cell}}}'
            lines.append(f"{lab} & {cell} & {pval(r['p'])} & {var(r['variation'])} \\\\")
        note = (f"\\multicolumn{{4}}{{p{{0.85\\linewidth}}}}{{\\footnotesize Synthetic "
                f"2$\\times$2 paired-dense setting, {s['outputs']['oracle']['n_groups']} "
                f"groups of three to six source clips. Another "
                f"{s['n_groups_left_out_identical_sources']} groups are left out because their "
                "source clips are identical once rotation and scale are removed, so SIF is "
                "undefined there. The oracle saturates SIF; both source-blind references sit "
                "at zero. Noise has no motion structure, so its variation is very large.} \\\\")
        self.write('tab_synthetic_sif_calibration', tabular(
            'lrrr', 'Output & SIF (95\\% CI) & $p$ & Variation \\\\', lines, note))

    def conditional_mean_ladder(self):
        seeds = [self.read(f'synthetic/conditional_mean_ladder_seed{s}.json')
                 for s in (42, 43, 44)]
        ms = [2, 4, 8, 16, 32, 50]   # random pairing needs at least two clips per cell
        lines = []
        for m in ms:
            ratios, errors = [], []
            for d in seeds:
                for r in d['rows']:
                    if r['pairs_per_cell'] == m:
                        ratios.append(r['variance_ratio'])
                        errors.append(r['mean_squared_error'] /
                                      max(r['true_map_variance_per_cell'], 1e-9))
            lines.append(f'{m} & {statistics.mean(ratios):.3f} $\\pm$ '
                         f'{statistics.stdev(ratios):.3f} & {statistics.mean(errors):.2f} '
                         f'$\\pm$ {statistics.stdev(errors):.2f} & {len(ratios)} \\\\')
        note = ('\\multicolumn{4}{p{0.94\\linewidth}}{\\footnotesize Each source clip is trained '
                'against the true target of another clip from the same cell, drawn afresh at '
                'every step. Across all $M$ and 3 seeds, the predicted variance stays at about '
                '2\\% of the oracle within-cell variance, the cell-mean compression '
                'Proposition~\\ref{prop:cm} predicts.} \\\\')
        self.write('tab_conditional_mean_ladder', tabular(
            'rrrr',
            '$M$ pairs / cell & $\\sigma^2_{\\text{pred}} / \\sigma^2_{\\text{oracle}}$ & '
            'MSE $/\\sigma^2_{\\text{oracle}}$ & $n_{\\text{seeds}}$ \\\\', lines, note))

    # -------------------------------------------------------------- action AUC

    def label_only(self):
        d = self.read('action_auc/label_only_audit.json')
        anchor = next(r for r in d['rows'] if r['method'] == 'ANCHOR')
        exact = next(r for r in d['rows'] if r['method'] == 'random-same-exact-action')
        cluster = next(r for r in d['rows'] if r['method'] == 'random-same-cluster')
        lines = []
        for r in d['rows']:
            if r['method'] == 'ANCHOR':
                cells = (fmt(r['overall_cluster_auc']), fmt(r['held_out_cluster_auc']))
            else:
                cells = (f"{fmt(r['overall_cluster_auc'])}\\,"
                         f"({r['overall_cluster_auc'] - anchor['overall_cluster_auc']:+.3f})",
                         f"{fmt(r['held_out_cluster_auc'])}\\,"
                         f"({r['held_out_cluster_auc'] - anchor['held_out_cluster_auc']:+.3f})")
            lines.append(f"{r['method']} & {r['overall_n']} & {r['held_out_n']} & "
                         f"{cells[0]} & {cells[1]} \\\\")
        splits = ('overall', 'held_out')

        def gap(r, split):
            # the difference of the two printed values, as the table shows it
            return round(r[f'{split}_cluster_auc'] - anchor[f'{split}_cluster_auc'], 3)

        note = ('\\multicolumn{5}{p{0.94\\linewidth}}{\\footnotesize Parenthetical $\\Delta$ '
                'values are method AUC minus ANCHOR. random-same-cluster matches ANCHOR '
                f'within $\\pm {max(abs(gap(cluster, k)) for k in splits):.3f}$ on both splits. '
                'random-same-exact-action exceeds ANCHOR '
                f"by ${gap(exact, 'overall'):+.3f}$ overall and "
                f"${gap(exact, 'held_out'):+.3f}$ held-out, but only on the "
                f"${exact['overall_n']}/{exact['overall_total']}$ and "
                f"${exact['held_out_n']}/{exact['held_out_total']}$ "
                'coverage where exact-action positives exist; the coverage '
                'qualifier travels with the gap.} \\\\')
        self.write('tab_label_only', tabular(
            'lrrll',
            'Method & Overall $n / 600$ & Held-out $n / 200$ & Overall AUC & '
            'Held-out AUC \\\\', lines, note))

    def enumeration(self):
        d = self.read('action_auc/enumeration.json')['subsets']
        rows = [('All', 'all', False),
                ('In-distribution (same train pair)', 'in_distribution', False),
                ('Mixed (one held-out skeleton)', 'mixed', True),
                ('Held-out (both cross-skeleton)', 'held_out', False)]
        lines = []
        for label, key, bold in rows:
            lo, mid, hi = d[key]['auc_ci']
            cell = f'{mid:.3f} [{lo:.3f}, {hi:.3f}]'
            if bold:
                cell = f'\\textbf{{{cell}}}'
            n = f"{d[key]['n']:,}".replace(',', '{,}')
            lines.append(f'{label} & {n} & {cell} \\\\')
        note = ('\\multicolumn{3}{p{0.96\\linewidth}}{\\footnotesize ANCHOR on the '
                '30{,}497-pair full enumeration. Cluster-eligibility filter removes triples '
                'whose target skeleton has no in-cluster positives. The mixed subset is the '
                'held-out case.} \\\\')
        self.write('tab_anchor_enumeration', tabular(
            'lrr',
            'Subset & $n$ triples (cluster-eligible) & Cluster-tier AUC (95\\% CI) \\\\',
            lines, note))

    def anchor_pred(self):
        d = self.read('action_auc/anchor_cluster_classifier.json')
        lines = [f"{r['domain']} & {100 * r['n_correct'] / r['n_total']:.1f}\\% & "
                 f"{r['n_correct']} $/$ {r['n_total']} \\\\" for r in d['rows']]
        note = ('\\multicolumn{3}{p{0.93\\linewidth}}{\\footnotesize Replacing the '
                'ground-truth cluster label with the predicted cluster pulls Procrustes '
                'performance below random on exact-tier and on a hand-curated query subset. '
                'The in-domain row is the accuracy on the clips the classifier was trained '
                'on.} \\\\')
        self.write('tab_anchor_cluster_classifier', tabular(
            'lrr', 'Domain & Cluster classifier accuracy & $n$ correct $/$ total \\\\',
            lines, note))

    def _fold_mean(self, auc, method, distance, tier):
        return float(np.mean([auc[method][f][distance][f'cluster_tier_{tier}_auc_ci'][1]
                              for f in ('42', '43')]))

    def master_auc(self):
        auc = self.read('action_auc/fold_auc.json')['methods']
        groups = [['Action oracle', 'Self-positive reference'],
                  ['ANCHOR', 'Cluster-Classifier Retrieval', 'Q-Retrieval (Q-feature only)'],
                  ['MoReFlow-T', 'MoReFlow-I', 'ACE-T', 'ACE-I'],
                  ['Random target skeleton (null)']]
        blocks = ['\n'.join(
            f'{m} & ' + ' & '.join(f"{self._fold_mean(auc, m, d, 'test_test'):.3f}"
                                   for d in ('procrustes', 'zscore_dtw', 'q_component'))
            + ' \\\\' for m in g) for g in groups]
        body = ('\\begin{tabular}{lrrr}\n\\toprule\n'
                '\\multicolumn{4}{c}{Cluster-tier held-out AUC (average of folds 42 and 43)} \\\\\n'
                'Method & Procrustes & Z-DTW & Q-comp \\\\\n\\midrule\n'
                + '\n\\midrule\n'.join(blocks) + '\n\\midrule\n'
                '\\multicolumn{4}{l}{\\footnotesize Motion2Motion-BVH has Procrustes only '
                'and is reported separately.} \\\\\n\\bottomrule\n\\end{tabular}\n')
        self.write('tab_action_auc_three_distances', body)

    def gen_variants_summary(self):
        auc = self.read('action_auc/fold_auc.json')['methods']
        rows = [('AnyTop', 'self-supervised source-conditioned diffusion'),
                ('Precursor action-conditioned generator', 'action-conditioned generation'),
                ('Precursor latent-bridge generator', 'latent bridge objective'),
                ('DPG-SB-v3', 'latent bridge objective; no decoded-motion penalties'),
                ('AL-Flow', 'cluster + exact-action labels'),
                ('AL-Flow-Src', 'AL-Flow + source motion + skeleton identity'),
                ('AL-Flow-Src-G', 'AL-Flow + source motion + skeleton graph')]
        lines = [f"{m} & {channels} & {self._fold_mean(auc, m, 'procrustes', 'overall'):.3f} \\\\"
                 for m, channels in rows]
        n_q = {m: auc[m]['42']['procrustes']['cluster_tier_n']
               for m in ('AnyTop', 'AL-Flow', 'AL-Flow-Src')}
        body = ('{\\small\n\\begin{tabular}{p{0.22\\linewidth}p{0.50\\linewidth}r}\n\\toprule\n'
                'Variant & Conditioning channels & Cluster AUC \\\\\n\\midrule\n'
                + '\n'.join(lines) + '\n\\midrule\n'
                '\\multicolumn{3}{p{0.94\\linewidth}}{\\footnotesize The two precursor rows '
                'are reported only for the action-level AUC analysis and are not part of the '
                '14-method SIF roster. '
                f"AL-Flow and AL-Flow-Src produce outputs for {n_q['AL-Flow']} and "
                f"{n_q['AL-Flow-Src']} of the {n_q['AnyTop']} queries in each fold, and "
                "their AUC is computed on those queries.} \\\\\n"
                '\\bottomrule\n\\end{tabular}\n}\n')
        self.write('tab_generated_variants_auc', body)

    # ------------------------------------------------------------------- robots

    def robot_g1_full(self):
        raw = {r['method']: r for r in self.read('robots/g1_four_measures_raw.json')['rows']}
        dur = {r['method']: r for r in
               self.read('robots/g1_four_measures_length_controlled.json')['rows']}
        fair = self.read('robots/g1_random_clip_auc.json')['own_clip_excluded']
        # G1 groups are single actions, so the random-clip row names the action
        labels = dict(ROBOT_CASES, random_clip='Random same-action clip')
        lines = []
        for key, _ in ROBOT_CASES:
            auc = fair if key == 'random_clip' else raw[key]['action_auc_mean']
            lines.append(f"{labels[key]} & {num(raw[key]['sif'])} & "
                         f"{num(dur[key]['sif'])} & {auc:.3f} & "
                         f"{raw[key]['variation']:.3f} & "
                         f"{raw[key]['realistic_pct']:.1f}\\% \\\\")
        self.write('tab_robot_g1_full', tabular(
            'lrrrrr',
            ' & Raw SIF & Length-controlled SIF & Action AUC & Variation & Realistic \\\\',
            lines))

    def robot_adversarial(self):
        raw = {r['method']: r for r in self.read('robots/g1_four_measures_raw.json')['rows']}
        second_setup = self.read('robots/g1_adversarial_second_setup.json')['runs']
        planned = raw['adversarial_objective']
        lines = [f"Planned run & {num(planned['sif'])} & "
                 f"{planned['action_auc_mean']:.3f} & diverged & "
                 f"{planned['realistic_pct']:.1f}\\% \\\\"]
        for i, r in enumerate(second_setup, 1):
            lines.append(f"Re-run, seed {i} & {num(r['sif'])} & {r['action_auc']:.3f} & "
                         f"{r['variation']:.3f} & {r['realistic_pct']:.1f}\\% \\\\")
        self.write('tab_robot_adversarial', tabular(
            'lrrrr', ' & Raw SIF & Action AUC & Variation & Realistic \\\\', lines))

    def _six_robots(self):
        d = self.read('robots/six_robots_sif.json')
        return ({k: {x['method']: x for x in v} for k, v in d['results'].items()},
                d['prediction_checks'])

    def robot_predictions(self):
        res, checks = self._six_robots()
        count = dict(true_retarget=0, floor=0, floor_dense=0, averaging=0, unpaired=0, both=0)
        missed_first, missed_third = [], []
        for k, g in checks.items():
            r = res[k]
            if r['true_retarget']['sif'] >= 0.80 and \
                    g['true_retarget']['sif_interval']['ci95'][0] > 0:
                count['true_retarget'] += 1
            else:
                missed_first.append(r['true_retarget']['sif'])
            ci90 = g['random_clip']['sif_interval']['ci90']
            if ci90[0] >= -0.20 and ci90[1] <= 0.20:
                count['floor'] += 1
                count['floor_dense'] += '|dense|' in k
            for case, key in (('averaging_objective', 'averaging'),
                              ('unpaired_objective', 'unpaired')):
                q = g[case]['variation_vs_true_retarget']
                if q['ratio'] <= 0.35 and q['ci95'][1] < 0.50:
                    count[key] += 1
                elif key == 'unpaired':
                    missed_third.append(q['ratio'])
            q = checks[k]['true_pair_model']['variation_vs_true_retarget']
            interval = checks[k]['true_pair_model']['sif_interval']
            if r['true_pair_model']['sif'] >= 0.60 and interval['ci95'][0] > 0 and \
                    q['ratio'] >= 0.70 and q['ci95'][0] > 0.50:
                count['both'] += 1
        sif_true_pair = [res[k]['true_pair_model']['sif'] for k in res]
        lines = [
            f"The true retarget is recognized & SIF $\\geq 0.80$, interval above 0 & "
            f"{count['true_retarget']} of 24; every interval above 0; misses "
            f"{min(missed_first):.3f}--{max(missed_first):.3f} \\\\",
            f"A random same-group clip sits at the floor & 90\\% interval within $\\pm 0.20$ "
            f"& {count['floor']} of 24; {count['floor_dense']} of 12 in the denser setting \\\\",
            f"The averaging objective loses variation & variation $Q \\leq 0.35$, upper "
            f"bound $< 0.50$ & {count['averaging']} of 24 \\\\",
            f"The unpaired objective loses variation & variation $Q \\leq 0.35$, upper "
            f"bound $< 0.50$ & {count['unpaired']} of 24; missed with variation "
            f"$Q = {max(missed_third):.2f}$ \\\\",
            f"The true-pair model recovers both & SIF $\\geq 0.60$ and variation "
            f"$Q \\geq 0.70$ & "
            f"{count['both']} of 24; SIF {min(sif_true_pair):.2f}--"
            f"{max(sif_true_pair):.2f} \\\\",
        ]
        self.write('tab_robot_predictions', tabular(
            'p{0.28\\linewidth}p{0.26\\linewidth}p{0.34\\linewidth}',
            'Prediction & Threshold & Result \\\\', lines))

    def robot_per_robot(self):
        res, checks = self._six_robots()
        lines = []
        for rk, rname in ROBOTS:
            for setting, slabel in (('sparse', 'three-clip'), ('dense', 'denser')):
                for mode, mlabel in (('raw', 'raw'),
                                     ('length_controlled', 'length-controlled')):
                    k = f'{rk}|{setting}|{mode}'
                    r, g = res[k], checks[k]
                    lines.append(
                        f"{rname} & {slabel} & {mlabel} & "
                        f"{num(r['true_retarget']['sif'])} & "
                        f"{num(r['random_clip']['sif'])} & "
                        f"{g['unpaired_objective']['variation_vs_true_retarget']['ratio']:.3f} & "
                        f"{g['averaging_objective']['variation_vs_true_retarget']['ratio']:.3f} & "
                        f"{num(r['true_pair_model']['sif'])} & "
                        f"{g['true_pair_model']['variation_vs_true_retarget']['ratio']:.2f} \\\\")
                lines.append('\\addlinespace')
        header = (' & & & True retarget & Random clip & Unpaired & Averaging & '
                  '\\multicolumn{2}{c}{True-pair model} \\\\\n'
                  '\\cmidrule(lr){8-9}\n'
                  'Robot & Setting & Scoring & SIF & SIF & Variation ($Q$) & '
                  'Variation ($Q$) & SIF & Variation ($Q$) \\\\')
        self.write('tab_robot_per_robot', tabular('lllrrrrrr', header, lines[:-1]))

    def six_robots_four_measures(self):
        fm = {(r['robot'], r['setting']): r
              for r in self.read('robots/six_robots_four_measures.json')['results']}
        fair = self.read('robots/six_robots_random_clip_auc.json')['own_clip_excluded']
        res, checks = self._six_robots()

        def agg(values, form):
            return f'{form(float(np.mean(values)))} ({form(min(values))}--{form(max(values))})'

        lines = []
        for setting, slabel in (('sparse', 'Three clips per group'), ('dense', 'Denser setting')):
            lines.append(f'\\multicolumn{{5}}{{l}}{{\\emph{{{slabel}}}}} \\\\')
            for key, label in ROBOT_CASES:
                sif = [res[f'{rk}|{setting}|raw'][key]['sif'] for rk, _ in ROBOTS]
                if key == 'random_clip':
                    auc = [fair[f'{rk}|{setting}'] for rk, _ in ROBOTS]
                else:
                    auc = [fm[(rk, setting)]['rows'][key]['action_auc_mean'] for rk, _ in ROBOTS]
                realism = [fm[(rk, setting)]['rows'][key]['realistic_pct']
                           for rk, _ in ROBOTS]
                q = [1.0 if key == 'true_retarget'
                     else checks[f'{rk}|{setting}|raw'][key]['variation_vs_true_retarget']['ratio']
                     for rk, _ in ROBOTS]
                lines.append(
                    f"{label} & {agg(sif, lambda x: f'{x:+.2f}'.replace('-', '$-$'))} & "
                    f"{agg(auc, lambda x: f'{x:.2f}')} & {agg(q, lambda x: f'{x:.2f}')} & "
                    f"{agg(realism, lambda x: f'{x:.0f}')}\\% \\\\")
        self.write('tab_six_robots_four_measures', tabular(
            'lllll',
            ' & Raw SIF & Action AUC & Variation ($Q$) & Realistic \\\\', lines))

    # ------------------------------------------------------------------- driver

    def all(self):
        self.dataset_statistics()
        self.sif_main_1891(); self.sif_original_49(); self.clustered_ci()
        self.paired_ace_moreflow(); self.sif_stability(); self.noise_control()
        self.pair_count_distribution(); self.truebones_four_measures()
        self.ace_source_removal(); self.ace_adv_matched(); self.ace_stretch()
        self.ace_leakage_nn()
        self.latent_rotation_ratios(); self.cross_seed_alignment(); self.effective_rank()
        self.lsif_failure_modes()
        self.synthetic_sif_calibration(); self.conditional_mean_ladder()
        self.label_only(); self.enumeration(); self.anchor_pred()
        self.master_auc(); self.gen_variants_summary()
        self.robot_g1_full(); self.robot_adversarial(); self.robot_predictions()
        self.robot_per_robot(); self.six_robots_four_measures()
        for f in sorted(STATIC.glob('*.tex')):
            shutil.copyfile(f, self.out / f.name)
            print('copied', shown(self.out / f.name))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--results', default=str(ROOT / 'results'),
                    help='folder holding the result files')
    ap.add_argument('--out', default=str(ROOT / 'paper' / 'output' / 'tables'),
                    help='where to write the tables')
    a = ap.parse_args()
    Tables(a.results, a.out).all()


if __name__ == '__main__':
    main()
