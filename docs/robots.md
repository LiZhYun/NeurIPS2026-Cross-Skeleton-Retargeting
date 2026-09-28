# The robot studies

The animal results in the paper have one weakness: in the Truebones Zoo nobody knows which
target motion truly belongs to which source clip, so the data can show that methods sit at
the source-blind floor (the score of a method that ignores its source) but not what a
correct method would score. The two robot studies supply that missing piece. Each human clip
has a known robot counterpart, so we can check that Source-Instance Fidelity (SIF)
recognises a true retarget, and we can compare the two kinds of training objective the
paper's theory is about with a model trained on the true pairs, all on the same data.

- **Human to Unitree G1.** Human motion capture from BONES-SEED, whose every human take
  comes with a Unitree G1 version of the same take.
- **LAFAN1 to six robots.** The LAFAN1 recordings retargeted to six humanoid robots of
  different sizes and builds with General Motion Retargeting, a public retargeting tool.

This guide walks through both, from the downloads to the paper's tables. The code is in
[`robots/`](../robots); [robots/README.md](../robots/README.md) says which measures were
fixed before each study and which were added later.

## Words used below

- **Action group**: the clips scored together. In the G1 study, the recordings of one
  action (such as one kind of crouch); in the six-robot study, the performances of one
  numbered routine (such as dance2) by different people.
- **SIF**: within each action group, the correlation between how different the human
  clips are from one another and how different the robot motions are. Near one when the
  robot motions keep the differences between their sources, near zero when they ignore
  them.
- **Variation**: how far apart the robot motions of an action group are, relative to how
  far apart their human sources are. It falls to zero when a method returns nearly the
  same motion whatever it is given. In the six-robot study the paper reports it as Q, a
  model's variation divided by that of the true retarget.
- **Action-level AUC**: whether the action is still recognisable in a robot motion: the
  chance that real robot clips of the same action come out closer to it than clips of
  other actions. One half means the motion carries no information about its action.
- **Realism**: the share of robot motions whose poses lie as close to real robot poses as
  95 per cent of held-out real clips do.
- **Setting** (six-robot study only): **sparse** keeps at most three performers and one
  clip each per action group, about as much data as the animal set has; **dense** keeps
  every performer and two clips each.
- **Raw and length-controlled**: two ways of comparing motions of unequal duration: over
  the frames they share, or after stretching both to the same length.

The rows of every robot table are the same kinds of thing:

| row | what it is |
| --- | --- |
| true retarget | the clip's own robot counterpart |
| random clip | the robot counterpart of a different clip of the same action group, drawn at random |
| unpaired objective | two autoencoders, one per body, sharing one space of codes and never shown a pair (an autoencoder squeezes a motion into a short code and rebuilds it from that code) |
| averaging objective | a model trained to match a random robot clip of the same action group; its best answer is the group's average motion |
| model trained on true pairs | the same model trained to match the true counterpart |
| adversarial objective (G1 only) | a model trained like ACE, one of the evaluated methods: against a critic, a second network that tells real robot motion from produced motion, and against a ten-number, position-based subset of ACE's summary of the source motion |

The paper's tables print some of these names differently. The true retarget is printed as
"True retargeted counterpart"; the random clip as "Random same-action clip" (G1) or
"Random same-group clip" (six robots). The sparse setting is printed as "Three clips per
group" or "three-clip", and the dense setting as "Denser setting" or "denser"; the code
and its folders keep the names `sparse` and `dense`. Raw and length-controlled scoring
appear as "Raw SIF" and "Length-controlled SIF", or in a "Scoring" column. In the table of
the adversarial objective, "Planned run" is the model of step 4 of the G1 study, and
"Re-run, seed 1" to "seed 3" are the three seeds (42, 43 and 44 in the code) of its second
training setup, step 6.

## Before you start

### Two environments

Training and scoring use the repository's main environment (see the
[main README](../README.md)). Preparing the data and drawing the figure need a newer Python
and MuJoCo, a library that simulates and draws robots, and use their own:

```bash
conda env create -f robots/environment.yml
conda activate motion-robots
```

Every command below is run from the repository root and says which environment it needs.

### Code fetched from other projects

```bash
bash robots/human_to_g1/fetch_g1_model.sh   # the Unitree G1 description, from HoloSoma
bash robots/fetch_gmr.sh                    # General Motion Retargeting and its six robots
```

A robot description is the set of files that says what a robot is made of: its links,
their lengths and shapes, and how its joints connect and how far they turn. HoloSoma is an
open-source humanoid robot project from Amazon; the G1 study uses its description of the
Unitree G1. Both scripts download a fixed commit into `external/`, which git ignores.

### Data

This repository contains no motion data from either motion capture set, and no file of
poses or joint positions made from them. Both studies keep everything they read and write
under `data/robots/`, which git ignores.

**BONES-SEED** (G1 study). The dataset's page
(<https://huggingface.co/datasets/bones-studio/seed>) asks you to accept its licence before
downloading. Download these three files:

| file | what it is | size |
| --- | --- | --- |
| `metadata/seed_metadata_v004.parquet` | the metadata table, one row per recording | 4.5 MB |
| `soma_uniform.tar.gz` | the human recordings as BVH files (a common text format for skeleton motion), on one shared skeleton | 45 GB |
| `g1.tar.gz` | the matching Unitree G1 recordings as CSV tables | 23.5 GB |

Use version 4 of the metadata table: the clip selection is drawn from it, and a different
version can give a different selection. The archives do not need unpacking; the extraction
step reads the 710 files it needs straight out of them. Training data includes Motion Data
by Bones Studio (https://bones.studio/). Use of the underlying dataset is subject to the
BONES Motion Capture Dataset License Agreement.

**LAFAN1** (six-robot study). The 77 motion files are in `lafan1/lafan1.zip` in
<https://github.com/ubisoft/ubisoft-laforge-animation-dataset>, stored with Git Large File
Storage (Git LFS). Unpack it into the folder this guide uses:

```bash
unzip lafan1.zip -d data/robots/lafan1_to_six_robots/lafan1_bvh
```

Each file is then named like `dance1_subject1.bvh`. LAFAN1 is licensed CC BY-NC-ND 4.0.

## Study 1: human motion capture to the Unitree G1

Each step is `python -m robots.human_to_g1.<step>`, and all of them use the data folder
`data/robots/human_to_g1/` unless given `--data_dir`.

| # | step | environment | GPU | time |
| --- | --- | --- | --- | --- |
| 1 | `select_clips` | robot | no | under a minute |
| 2 | `extract_clips` | either | no | about half an hour |
| 3 | `convert_to_tensors` | robot | no | about two minutes |
| 4 | `train`, `train_adversarial` | main | yes | about three minutes for the three models of `train`, under two minutes for `train_adversarial` |
| 5 | `score_four_measures`, `random_clip_auc` | main | no | about twenty minutes per length setting |
| 6 | `train_adversarial_second_setup` | main | yes | about five minutes per seed |

**1. Choose the clips** (robot environment):

```bash
python -m robots.human_to_g1.select_clips \
    --metadata <download folder>/metadata/seed_metadata_v004.parquet
```

This draws ninety actions, at most thirty per cent of them locomotion, with four
recordings each (three when only three suitable ones exist), and writes the list to
`data/robots/human_to_g1/clips.json`. The draw is fixed by a seed, so it gives exactly the
paper's selection: 90 actions and 355 clips. The list is not shipped because it copies
parts of the metadata, which the dataset's licence treats as confidential.

**2. Pull the clips out of the archives** (either environment):

```bash
python -m robots.human_to_g1.extract_clips \
    --human_archive <download folder>/soma_uniform.tar.gz \
    --robot_archive <download folder>/g1.tar.gz
```

This writes 355 human and 355 robot files under `data/robots/human_to_g1/extracted/`.
`--check` reports how many are present without extracting anything.

**3. Convert them to joint positions** (robot environment, after `fetch_g1_model.sh`):

```bash
python -m robots.human_to_g1.convert_to_tensors
```

For every clip this writes the positions of 29 human joints and 30 robot body points, at
30 frames per second, to `tensors/human/` and `tensors/g1/`, and a report of simple checks
(bones keep their length, feet stay near the ground, both sides have the same number of
frames) to `conversion_report.json`. It stops with an error if no clip can be converted.

**4. Train the models** (main environment, GPU):

```bash
python -m robots.human_to_g1.train --model all
python -m robots.human_to_g1.train_adversarial
```

Each model writes one robot motion per human clip to `generated/<model>/` and saves itself
to `checkpoints/<model>.pt`. On our GPU, `train --model all` took 173 seconds for its three
models (about a minute and a half for the unpaired objective and half a minute for each of
the other two). No trained models are distributed with this repository. On our GPU,
retraining with these settings gives the paper's motions bit for bit (our GPU is an NVIDIA
H200, and we checked this again from a fresh copy of this repository); on other hardware
small differences are possible. A saved model can
write its motions again without training:

```bash
python -m robots.human_to_g1.generate \
    --checkpoint data/robots/human_to_g1/checkpoints/true_pair_model.pt \
    --out_dir data/robots/human_to_g1/generated_again/true_pair_model
```

**5. Score** (main environment, processor only):

```bash
python -m robots.human_to_g1.score_four_measures --length raw \
    --out output/robots/g1_four_measures_raw.json
python -m robots.human_to_g1.score_four_measures --length length_controlled \
    --out output/robots/g1_four_measures_length_controlled.json
python -m robots.human_to_g1.random_clip_auc --out output/robots/g1_random_clip_auc.json
```

The scores go to `output/robots/`, so the shipped files in `results/robots/` stay as they
are for comparison (see the end of this guide).

Nearly all the time goes into the action measure, which compares every robot motion with
about a hundred real clips. `python -m robots.human_to_g1.score --length raw --out
<file.json>` gives SIF and variation alone in a few minutes.

**6. The adversarial model's second training setup** (main environment, GPU):

```bash
python -m robots.human_to_g1.train_adversarial_second_setup \
    --out output/robots/g1_adversarial_second_setup.json
```

Training of the adversarial model in step 4 does not converge; its output grows without
bound. This step trains it again with ACE's own optimiser settings (a smaller learning
rate, a penalty on the critic's gradients, the critic held fixed while the model learns),
with three seeds, and scores each run on the same four measures. The objective, data,
network and scoring are unchanged.

## Study 2: LAFAN1 to six robots

Each step is `python -m robots.lafan1_to_six_robots.<step>`, and all of them use the data
folder `data/robots/lafan1_to_six_robots/`. The first three steps run in the robot
environment with the retargeting tool on the import path:

```bash
conda activate motion-robots
export PYTHONPATH=external/GMR
```

| # | step | environment | GPU | time |
| --- | --- | --- | --- | --- |
| 1 | `choose_windows` | robot | no | seconds |
| 2 | `retarget`, `retarget_retry` | robot | no | 4 to 12 minutes per robot with 16 processes |
| 3 | `build_tensors` | robot | no | about two minutes |
| 4 | `train` | main | yes | a few minutes per robot and setting |
| 5 | `score`, `score_four_measures`, `random_clip_auc` | main | no | about half an hour in all |

**1. Choose the windows**:

```bash
python -m robots.lafan1_to_six_robots.choose_windows
```

It checks that all 77 recordings have 22 joints and 30 frames per second, and picks two
ten-second windows from each, starting a quarter and about two thirds of the way in. The
result is `takes.json`.

**2. Retarget every recording to every robot**:

```bash
for R in unitree_g1 booster_t1_29dof fourier_n1 stanford_toddy engineai_pm01 pal_talos; do
  python -m robots.lafan1_to_six_robots.retarget --robot $R \
      --takes data/robots/lafan1_to_six_robots/takes.json \
      --bvh_dir data/robots/lafan1_to_six_robots/lafan1_bvh \
      --out_dir data/robots/lafan1_to_six_robots/retargeted --processes 16
done
python -m robots.lafan1_to_six_robots.retarget_retry
```

Retargeting takes about forty seconds of processor time per recording for the four
lighter robots and about two minutes for the EngineAI PM01 and the PAL TALOS. The
retargeting tool finds each frame's joint angles with a solver, a numerical optimiser that
searches for the angles putting the robot's body closest to the person's. The solver
occasionally crashes on a recording, and a process can also be stopped when the machine
runs out of memory (16 processes can need more than 32 GB). A run that loses a process
this way does not finish on its own: once no new file has appeared in the robot's folder
under `retargeted/` for several minutes, stop it and go on to the next robot. The retry
step redoes, one process per recording, anything a crash left missing.

**3. Build the joint positions and the clip list**:

```bash
python -m robots.lafan1_to_six_robots.build_tensors
```

This writes one file per clip and skeleton to `tensors/<human or robot>/`, the clip list
`clips.json` (29 action groups; 72 clips in the sparse setting and 154 in the dense one),
and `conversion_report.json`.

**4. Train** (main environment, GPU):

```bash
python -m robots.lafan1_to_six_robots.train
```

This trains the unpaired, averaging and true-pair models for every robot and setting and
writes their motions to `generated/<robot>/<setting>/<model>/`. `--robot` and `--setting`
train one at a time.

The retargeting solver's answers can differ from ours in the seventh significant digit on
another machine, and training can magnify such a difference. When we repeated the whole
study from a fresh copy of this repository, the joint positions of three robots (the Unitree G1,
the Booster T1 in the dense setting, and the Fourier N1) differed from ours by at most 2.4e-7, and their trained
models' numbers moved: for example, the averaging model's SIF for the Unitree G1 in the sparse
setting went from 0.059 to 0.028, the unpaired model's median realism in the dense setting
from 61% to 48% (its range over the robots is 1% to 100% either way), and the true-pair
model's SIF range from 0.77 to 0.99 to 0.78 to 0.99. The other three robots gave our
numbers exactly. Every prediction in the paper's table of six-robot predictions has the
same outcome as before, and the true-pair model still passes all 24 of its checks.

**5. Score** (main environment, processor only):

```bash
python -m robots.lafan1_to_six_robots.score --out output/robots/six_robots_sif.json
python -m robots.lafan1_to_six_robots.score_four_measures \
    --out output/robots/six_robots_four_measures.json
python -m robots.lafan1_to_six_robots.random_clip_auc \
    --out output/robots/six_robots_random_clip_auc.json
```

`score` first checks that every model produced a motion of the right shape for every clip,
and stops if not.

## The robot figure

The paper's robot figure (one dance on six robots, and two performers on the Unitree G1
with every model's output) is drawn from the six-robot study's data and trained models:

```bash
conda activate motion-robots
python -m robots.render.run_panels
```

It needs a GPU and takes about three minutes; `--no_video` draws the stills only,
in about a minute. The figure, the separate stills and videos, and a record of what they
show go to `paper/output/robot_figure/`. The models write body positions, not joint
angles, so each of their frames is fitted back onto the robot within its joint limits;
the fit error is printed on the figure.

## From the result files to the paper

`python -m paper.make_tables` rebuilds every appendix robot table from `results/robots/`:

| paper table | result files |
| --- | --- |
| Human-to-G1 study, all four measures (appendix) | `g1_four_measures_raw.json`, `g1_four_measures_length_controlled.json`, `g1_random_clip_auc.json` |
| An adversarial objective in the style of ACE (appendix) | the adversarial row of `g1_four_measures_raw.json`, and `g1_adversarial_second_setup.json` |
| Six-robot predictions (appendix) | `six_robots_sif.json` |
| Six-robot results for every robot, setting and scoring (appendix) | `six_robots_sif.json` |
| Six-robot study, four measures (appendix) | `six_robots_sif.json`, `six_robots_four_measures.json`, `six_robots_random_clip_auc.json` |

The G1 table in the main text shows the raw-comparison numbers of
`g1_four_measures_raw.json`, with the random clip's action-level AUC taken from
`g1_random_clip_auc.json`.

How the numbers are read:

- In the G1 files, each entry of `rows` is one table row: `sif`, `variation` (the median
  over action groups), `action_auc_mean` and `realistic_pct`. The random clip's action AUC
  comes from `own_clip_excluded` in `g1_random_clip_auc.json`, because a random clip left
  among its own references would find an exact copy of itself.
- In `six_robots_sif.json`, `results` holds every row for each `robot|setting|scoring`.
  `prediction_checks` holds, for the same keys, each model's `sif_interval` (the mean with
  95% and 90% intervals) and its `variation_vs_true_retarget`, whose `ratio` is the paper's
  Q. The predictions table counts, over the 24 combinations of robot, setting and scoring,
  how often each prediction's threshold is met. `pooled` averages each model over the six
  robots; the paper does not report it.
- In `six_robots_four_measures.json`, each entry is one robot and setting, with the action
  AUC and realism of every row; the six-robot four-measure table reports their mean and
  range over the robots.

To check that your own run matches the paper, compare each file you wrote to
`output/robots/` with the file of the same name in `results/robots/`, for example with
`diff output/robots/six_robots_sif.json results/robots/six_robots_sif.json`. For the G1
study, scores computed on another computer can differ from ours from about the sixth
significant digit on; with the retrained models from our GPU, every number the paper prints
came out identical. For the six-robot study, the trained models'
numbers can move more, as described under step 4 of that study.
