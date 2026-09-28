# Results

Every number the paper prints comes from a file in this folder. The scripts in `paper/`
read these files and write the tables and figures; nothing is typed into a script by hand.

Each file starts with a `description` field that says in plain words what it measures and
how to read its fields. Two scores come up throughout. Source-Instance Fidelity (`sif`) asks
whether a retargeted motion keeps what made its source clip different from other clips of
the same action. It is a correlation, so it runs from about −1 to 1, and a method that
ignores the source clip scores near zero. The action-level AUC (area under the ROC curve)
asks whether an output is closer to real target clips of the right action than to clips of
other actions. It runs from 0 to 1, and 0.5 means no better than chance.

A triple is one source skeleton, one target skeleton and one action. On the Truebones
evaluation it holds three to five source clips and the motions a method produced for them.

The tables and figures below are named as `paper/make_tables.py` and `paper/make_figures.py`
write them. [paper/README.md](../paper/README.md) says what each one shows.

## `truebones/`: the animal dataset

| File | Read by |
|---|---|
| `sif_1891.json` | `tab_sif_main_1891`, `main_sif_1891` |
| `sif_49.json` | `tab_sif_original_49`, `tab_truebones_four_measures` |
| `sif_49_variation.json` | `tab_sif_original_49`, `tab_truebones_four_measures` |
| `sif_37_intersection.json` | `app_rotation_test_vs_sif`, `app_sif_vs_variation` |
| `sif_clustered_intervals.json` | `tab_resampling_intervals`, `app_resampling_intervals` |
| `sif_triple_support.json` | `tab_pair_count_distribution`, `app_triples_per_method` |
| `sif_stability.json` | `tab_sif_stability` |
| `sif_noise_control.json` | `tab_noise_control` |
| `sif_paired_ace_moreflow.json` | `tab_paired_ace_moreflow` |
| `ace_source_removal.json` | `tab_ace_source_removal` |
| `ace_adversarial_matched_raw.json`, `ace_adversarial_fixed_length.json` | `tab_ace_adversarial_loss` |
| `ace_stretch_check.json` | `tab_ace_stretch` |
| `ace_leakage_nearest_neighbour.json` | `tab_ace_nearest_clip` |
| `ace_seed_spread.json` | no script; it holds the per-seed scores behind the two three-seed ACE rows of `tab_latent_sif` |
| `action_auc_49.json`, `realism_49.json` | `tab_truebones_four_measures` |
| `dataset_statistics.json` | `tab_dataset_statistics`; counted from the clip list in `benchmark/clip_index.json` and the query sets in `benchmark/sets/` |

## `robots/`: human motion capture retargeted to humanoid robots

| File | Read by |
|---|---|
| `g1_four_measures_raw.json` | `tab_robot_g1_full`, `tab_robot_adversarial` |
| `g1_four_measures_length_controlled.json` | `tab_robot_g1_full` |
| `g1_random_clip_auc.json` | `tab_robot_g1_full` |
| `g1_adversarial_second_setup.json` | `tab_robot_adversarial` |
| `six_robots_sif.json` | `tab_robot_predictions`, `tab_robot_per_robot`, `tab_six_robots_four_measures` |
| `six_robots_four_measures.json`, `six_robots_random_clip_auc.json` | `tab_six_robots_four_measures` |

## `synthetic/`: a made-up world where the right answer is known

| File | Read by |
|---|---|
| `sif_calibration.json` | `tab_synthetic_sif_calibration`, `app_synthetic_calibration` |
| `conditional_mean_ladder_seed{42,43,44}.json` | `tab_conditional_mean_ladder` |

## `latents/`: what happens inside the models

| File | Read by |
|---|---|
| `perturbation/<run>.json` | `tab_rotation_test`, `app_rotation_test`, `app_rotation_test_vs_sif` |
| `cross_seed_alignment_*.json` | `tab_cross_seed_alignment`, `app_cross_seed_alignment` |
| `effective_rank.json` | `tab_effective_rank`, `app_effective_rank` |
| `latent_sif.json` | `tab_latent_sif`, `app_latent_sif` |

## `action_auc/`: can the action still be recognised

| File | Read by |
|---|---|
| `fold_auc.json` | `tab_action_auc_three_distances`, `tab_generated_variants_auc`, `app_action_auc_band`, `app_action_auc_by_group` |
| `label_only_audit.json` | `tab_label_only`, `main_label_only_audit` |
| `enumeration.json` | `tab_anchor_enumeration`, `app_anchor_enumeration` |
| `anchor_cluster_classifier.json` | `tab_anchor_cluster_classifier` |

## Where the latent-space comparison gets its numbers

`latents/latent_sif.json` holds two scores for each of the sixteen trained runs, on the 37
triples every method answered. The SIF of each run is computed with the scoring in
`benchmark/score.py` from the motions that run produced. The latent score (L-SIF) is the same
correlation taken on the run's latent codes. Rows that cover three training seeds average all
three. The file holds only these scores; the table script computes the correlation between
them across the sixteen runs.
