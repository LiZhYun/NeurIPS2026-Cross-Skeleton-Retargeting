# Rebuilding the tables and figures of the paper

Every table and plot in the paper is built from a file in `results/`, except one short
main-text table whose numbers are copied from `tab_robot_g1_full` (see below). The two
scripts in this folder turn those files into the paper's tables and plots. This way you can
trace any number in the paper back to the measurement behind it.

Run both from the repository root:

```bash
python -m paper.make_tables
python -m paper.make_figures
```

`make_tables` writes one LaTeX file per table to `paper/output/tables/`. Each is a bare
`tabular` with no surrounding `table` environment, so the paper can place it in its own
float. It takes a few seconds and needs nothing beyond the main environment.

`make_figures` writes each plot twice to `paper/output/figures/`: a PDF at 300 dots per
inch for the paper and a PNG at 200 for quick viewing. Its labels are typeset with LaTeX,
so a TeX installation (with `latex`, `dvipng` and `kpsewhich`) has to be on your path. The
one exception is `main_sif_1891`, which draws its text without LaTeX. To rebuild only some
figures, name them:

```bash
python -m paper.make_figures --only main_sif_1891 app_action_auc_by_group
```

Both scripts take `--results` to read the result files from another folder and `--out` to
write somewhere else. Files are named as the paper's LaTeX source includes them.

### Which setup gives the paper's figures exactly

The figures in the paper were drawn with matplotlib 3.10 on Python 3.10, with TeX Live 2023
providing LaTeX. With that setup, listed in `requirements-figures.txt` in this folder,
the PNGs that `make_figures` writes are identical, pixel for pixel, to the ones in the paper,
except for small differences in how the text of `main_sif_1891` is smoothed. Any Python 3.10
works, for example the robot environment's Python (`robots/environment.yml`):

```bash
python3.10 -m venv figures-env
figures-env/bin/pip install -r paper/requirements-figures.txt
figures-env/bin/python -m paper.make_figures
```

The main environment has the older matplotlib 3.1.3. `make_figures` also runs there and
draws the same figures from the same numbers, at the same size, and they look the same,
but they are not identical pixel for pixel to the paper's, because the older matplotlib
draws some lines and text slightly differently. The tables do not depend on matplotlib.

## Which file feeds which table and figure

A triple is one source skeleton, one target skeleton and one action, with the three to five
source clips that share them. The main Truebones evaluation has 1,891 triples; the original
evaluation has 49; 37 of those are covered by every method.

### Main text

| Table or figure | What it shows | Read from `results/` |
|---|---|---|
| `main_sif_1891` | SIF of the fourteen methods on the 1,891 triples, raw and length-controlled | `truebones/sif_1891.json` |
| `tab_label_only` | Retrieval by action label alone can match ANCHOR's action-level AUC | `action_auc/label_only_audit.json` |
| `main_label_only_audit` | The same comparison as a plot | `action_auc/label_only_audit.json` |

The short human-to-G1 table in the main text is written directly in the paper. Its numbers
are the same as in `tab_robot_g1_full` below.

### Appendix: the Truebones evaluation

| Table or figure | What it shows | Read from `results/` |
|---|---|---|
| `tab_dataset_statistics` | How many animals, clips, actions and triples the evaluation has | `truebones/dataset_statistics.json` |
| `tab_sif_main_1891` | SIF on the 1,891 triples, with intervals and p-values | `truebones/sif_1891.json` |
| `tab_sif_original_49` | SIF on the 49 original triples | `truebones/sif_49.json`, `truebones/sif_49_variation.json` |
| `tab_resampling_intervals`, `app_resampling_intervals` | SIF intervals when whole source skeletons, target skeletons or actions are resampled | `truebones/sif_clustered_intervals.json` |
| `tab_paired_ace_moreflow` | Paired SIF differences between ACE and MoReFlow on the same triples | `truebones/sif_paired_ace_moreflow.json` |
| `tab_sif_stability` | How much SIF moves from one triple to the next, and how stable the average is | `truebones/sif_stability.json` |
| `tab_noise_control` | SIF when the generation noise is shared within a triple | `truebones/sif_noise_control.json` |
| `tab_pair_count_distribution`, `app_triples_per_method` | How many triples each method could be scored on | `truebones/sif_triple_support.json` |
| `app_sif_vs_variation` | SIF against how much a method's outputs differ from one another | `truebones/sif_37_intersection.json` |
| `tab_truebones_four_measures` | SIF, action-level AUC, variation and realism on the 49 triples | `truebones/sif_49.json`, `truebones/sif_49_variation.json`, `truebones/action_auc_49.json`, `truebones/realism_49.json` |
| `tab_ace_source_removal` | ACE when its source input is removed or shuffled | `truebones/ace_source_removal.json` |
| `tab_ace_adversarial_loss` | ACE-I with and without its adversarial loss | `truebones/ace_adversarial_matched_raw.json`, `truebones/ace_adversarial_fixed_length.json` |
| `tab_ace_stretch` | Output variation after stretching every output to the same length | `truebones/ace_stretch_check.json` |
| `tab_ace_nearest_clip` | Whether ACE outputs lie closer to real target clips than to their source | `truebones/ace_leakage_nearest_neighbour.json` |

### Appendix: inside the models

| Table or figure | What it shows | Read from `results/` |
|---|---|---|
| `tab_rotation_test`, `app_rotation_test` | Whether turning a model's internal code changes its output more than noise of the same size | `latents/perturbation/*.json` |
| `app_rotation_test_vs_sif` | That test against SIF, run by run | `latents/perturbation/*.json`, `truebones/sif_37_intersection.json` |
| `tab_cross_seed_alignment`, `app_cross_seed_alignment` | How much of the difference between two training seeds a single rotation explains | `latents/cross_seed_alignment_*.json` |
| `tab_effective_rank`, `app_effective_rank` | How many directions each model's internal codes really use | `latents/effective_rank.json` |
| `tab_latent_sif`, `app_latent_sif` | SIF on the output motion next to the same score on the internal codes | `latents/latent_sif.json` |

### Appendix: the synthetic setting

| Table or figure | What it shows | Read from `results/` |
|---|---|---|
| `tab_synthetic_sif_calibration`, `app_synthetic_calibration` | SIF where the true mapping is known | `synthetic/sif_calibration.json` |
| `tab_conditional_mean_ladder` | A model trained on squared error keeps little of the variation that belongs to the source, however many true pairs it gets | `synthetic/conditional_mean_ladder_seed{42,43,44}.json` |

### Appendix: action-level AUC

| Table or figure | What it shows | Read from `results/` |
|---|---|---|
| `tab_anchor_cluster_classifier` | How well ANCHOR predicts the action group | `action_auc/anchor_cluster_classifier.json` |
| `app_action_auc_by_group` | Held-out action-level AUC of every method, by action group | `action_auc/fold_auc.json` |
| `tab_anchor_enumeration`, `app_anchor_enumeration` | ANCHOR on every eligible triple, split by how many skeletons are held out | `action_auc/enumeration.json` |
| `tab_generated_variants_auc` | Action-level AUC of the generative methods and their label-matched comparators | `action_auc/fold_auc.json` |
| `tab_action_auc_three_distances`, `app_action_auc_band` | Held-out action-level AUC under three ways of comparing motions | `action_auc/fold_auc.json` |

### Appendix: the robot studies

| Table | What it shows | Read from `results/` |
|---|---|---|
| `tab_robot_g1_full` | Human to G1: SIF, action-level AUC, variation and realism | `robots/g1_four_measures_raw.json`, `robots/g1_four_measures_length_controlled.json`, `robots/g1_random_clip_auc.json` |
| `tab_robot_adversarial` | Human to G1 with an adversarial objective and no pairs | `robots/g1_four_measures_raw.json`, `robots/g1_adversarial_second_setup.json` |
| `tab_robot_predictions` | The six-robot predictions, written down before scoring, and how they came out | `robots/six_robots_sif.json` |
| `tab_robot_per_robot` | The six-robot results for every robot | `robots/six_robots_sif.json` |
| `tab_six_robots_four_measures` | The six-robot study on all four measures | `robots/six_robots_sif.json`, `robots/six_robots_four_measures.json`, `robots/six_robots_random_clip_auc.json` |

The robot pictures in the appendix are drawn by `robots/render/`; see
[docs/robots.md](../docs/robots.md).

### Tables written by hand

`tab_method_roster` lists the evaluated methods and `tab_theory_classes` relates each family
of methods to the formal results. They hold no measured numbers, so they are kept as text in
`paper/static/`, and `make_tables` copies them into its output folder.

## Figures drawn from the methods' motions

The figures that show skeletons in motion, and the videos that go with them, need the
motions each method produced and the processed Truebones data
([docs/data.md](../docs/data.md)). Neither is shipped. Their scripts are in
`paper/qualitative/`.

These scripts read the answers to the 49 original triples (the 49 set), in the folders the
methods' `generate` steps write them to ([docs/methods.md](../docs/methods.md)): one
`query_XXXX.npy` file per query in `outputs/<method>/set49/`. The method folders are named
as in that guide:

| Method | Folder | Method | Folder |
|---|---|---|---|
| AnyTop | `anytop` | DPG-SB-v3 | `dpg_sb` |
| ACE-T | `ace_t` | Motion2Motion-Direct | `motion2motion_direct` |
| ACE-I | `ace_i` | Motion2Motion-BVH | `motion2motion_bvh` |
| AL-Flow | `al_flow` | ANCHOR | `anchor` |
| AL-Flow-Src | `al_flow_src` | random-same-cluster | `random_same_cluster` |
| AL-Flow-Src-G | `al_flow_src_g` | random-same-exact-action | `random_same_exact_action` |
| MoReFlow-T | `moreflow_t` | | |
| MoReFlow-I | `moreflow_i` | | |

The scripts read `outputs/` by default; if your answers are elsewhere, pass the folder
that holds the method folders with `--outputs`:

```bash
python -m paper.qualitative.fig_qualitative_method_panorama
python -m paper.qualitative.fig_qualitative_failuremodes
python -m paper.qualitative.fig_qualitative_full_grid --outputs my_outputs
```

Given the motions behind the paper, the panorama and the failure-mode figure come out
pixel for pixel as printed when drawn with the figure setup above.

| Script | Writes | Methods it needs |
|---|---|---|
| `fig_qualitative_method_panorama` | `main_qualitative_method_panorama`: one Bird attack clip and what twelve methods made of it on the KingCobra | all but the two random baselines |
| `fig_qualitative_failuremodes` | `app_qualitative_failure_modes`: three Bird attack clips and one method per way of losing their differences | AL-Flow, AnyTop, MoReFlow-T, ACE-I |
| `fig_qualitative_full_grid` | `app_qualitative_full_grid_A`, `_B`, `_C`: every method on seven groups of three source clips | all fourteen |

They write to `paper/output/figures/` and, like `make_figures`, need LaTeX. The full grid
takes a few minutes. The processed data is read from its default place,
`dataset/truebones/zoo/truebones_processed/`; `--motions` and `--cond` point elsewhere.

The three matching videos are drawn by `video_qualitative_method_panorama`,
`video_qualitative_failuremodes` and `video_qualitative_full_grid`, which take the same
options and write MP4 files to `paper/output/videos/`. They do not need LaTeX, but they need
`ffmpeg` on your path. The full-grid videos, one per group of source clips, take about a
quarter of an hour together on a CPU, so on a shared machine run them as a batch job. For
example:

```bash
python -m paper.qualitative.video_qualitative_method_panorama
```

The animal pictures in the first two figures of the paper are rendered in Blender, which
is not part of the environment. `render_truebones_mesh_assets.py` draws the animal meshes
with a coloured skeleton on top, and `render_anytop_style_motion_assets.py` draws single
poses as coloured skeletons (it calls `composite_anytop_floor.py` to add the floor). Both
need the original Truebones download under `datasets/truebones/zoo/Truebone_Z-OO/`, and the
second also reads the method folders from `outputs/`, laid out as above. The first lines
of each script show how to start it from Blender. Their images go to
`paper/output/motion_mesh_assets/`.

## Collecting action-level AUC

`results/action_auc/fold_auc.json` gathers the action-level AUC of every method into one
file. You only need to rebuild it if you run that evaluation again yourself
([docs/benchmark.md](../docs/benchmark.md)). Give `benchmark.action_auc` one output folder
per method and fold, laid out as `<root>/<method>/fold_42/`, and then collect them:

```bash
python -m paper.collect_action_auc --root auc_runs --out outputs/fold_auc.json
```

The method's name in the collected file is its folder name. Without `--out`, the script
replaces the file shipped in `results/action_auc/`.
