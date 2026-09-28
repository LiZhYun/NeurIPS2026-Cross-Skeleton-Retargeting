# Code, models and data by others

This repository builds on work by other people. This page lists each outside source, who
made it, its licence, and how it reaches you. A source reaches you in one of three ways:

- included: the files are in this repository;
- fetched: a script in this repository downloads it at a fixed version into `external/`,
  which git ignores, or the environment file installs it;
- downloaded by you: you get it from its owner yourself, usually after accepting its terms.

Our own code is released under the MIT licence ([LICENSE](LICENSE)). The licences below
apply to the outside parts, not to our code.

## Summary

| Source | Made by | Licence | How it reaches you |
|---|---|---|---|
| AnyTop | Gat et al. | MIT | included, in `core/anytop/` and `core/truebones/` |
| Diffusion code inside AnyTop | OpenAI (guided-diffusion, baselines) | MIT | included, as part of AnyTop |
| Text conditioning code inside AnyTop | Meta (AudioCraft) | MIT | included, as part of AnyTop |
| Rotation code inside AnyTop | Meta (PyTorch3D), through ACTOR by Petrovich et al. | BSD (PyTorch3D), MIT (ACTOR) | included, as part of AnyTop |
| Motion library for BVH files | Sigal Raab, after Daniel Holden's code; fork by inbar-2344 | not stated | fetched by `environment.yml` |
| T5 text model | Google | Apache-2.0 | fetched from Hugging Face on first use |
| Motion2Motion | Chen et al. | none given | fetched by `methods/motion2motion/fetch_official.sh` |
| General Motion Retargeting (GMR) | Yanjie Ze et al. | MIT; each robot model has its own | fetched by `robots/fetch_gmr.sh` |
| Unitree G1 model from HoloSoma | Amazon FAR | Apache-2.0 for HoloSoma; the G1 folder has no licence file of its own | fetched by `robots/human_to_g1/fetch_g1_model.sh` |
| MuJoCo | Google DeepMind | Apache-2.0 | fetched by `robots/environment.yml` |
| mink | Kevin Zakka et al. | Apache-2.0 | fetched by `robots/environment.yml` |
| Truebones Zoo | Truebones Studios | Truebones terms of use | downloaded by you |
| BONES-SEED | Bones Studio | BONES Motion Capture Dataset License Agreement | downloaded by you |
| LAFAN1 | Ubisoft La Forge | CC BY-NC-ND 4.0 | downloaded by you |

The sections below give the details.

## Code included in this repository

### AnyTop

AnyTop is a diffusion model that generates animal motion for skeletons of any shape, by
Inbar Gat, Sigal Raab, Guy Tevet, Yuval Reshef, Amit H. Bermano and Daniel Cohen-Or
(<https://github.com/Anytop2025/Anytop>). Several of the methods we evaluate build on it,
and its data preparation turns the Truebones Zoo into the form every method reads.

We copied its code from commit `9e0085c` into `core/anytop/`, and its Truebones data
preparation into `core/truebones/`. It is released under the MIT licence, and its licence
text is kept in `core/anytop/LICENSE`. That text names Guy Tevet as copyright holder, because AnyTop's code builds on his Motion Diffusion Model
(<https://github.com/GuyTevet/motion-diffusion-model>, also MIT). We include only
the parts the paper needs; AnyTop's own evaluation, editing and visualisation code is left
out.

Every copied file has its import lines changed so that it loads as part of the `core`
package. Most files are otherwise exactly as in AnyTop. These are the ones we changed:

- `core/anytop/utils/parser_util.py`: the choice of skeletons now also offers `train` and
  `held_out`, the 60 training and 10 held-out skeletons of the paper. Two options,
  `--wandb_project` and `--wandb_entity`, name the Weights & Biases project a run reports to.
- `core/anytop/utils/ml_platforms.py`: a training run can send plots to Weights & Biases. A
  new run with the name of an earlier one now starts its own record, instead of stopping
  with an error or writing into the earlier record. When no Weights & Biases account is
  named, your default account is used.
- `core/truebones/param_utils.py`: the folder of the raw download is set to
  `datasets/truebones/zoo/Truebone_Z-OO`; AnyTop pointed to `dataset/`, where the processed
  files go. The list of the 10 held-out skeletons and the `train` and `held_out` choices
  are added. The existing choices are unchanged.
- `core/truebones/motion_process.py`: the data preparation shows a progress bar while it
  works through the animals. What it computes is unchanged.
- `core/truebones/create_dataset.py` and `core/truebones/process_new_skeleton.py`: a note at
  the top says how to run them. AnyTop has two copies of `process_new_skeleton.py`; ours
  follows the one in AnyTop's `utils/` folder.

`core/anytop/data_loaders/truebones/data/dataset.py` and `core/anytop/utils/model_util.py`
are unchanged apart from imports. What our source-conditioned version of AnyTop adds lives
in `methods/anytop_source/`, which imports AnyTop's classes and extends them rather than
editing them.

### Code carried inside AnyTop

AnyTop itself contains code from other projects. It is included here as part of AnyTop,
with the credits AnyTop gives it:

- The diffusion process in `core/anytop/diffusion/` and parts of
  `core/anytop/sample/generate.py` are based on OpenAI's guided-diffusion
  (<https://github.com/openai/guided-diffusion>), and `core/anytop/diffusion/logger.py` is
  taken from OpenAI's baselines (<https://github.com/openai/baselines>). Both are released
  under the MIT licence. Comments in `gaussian_diffusion.py` also point to Jonathan Ho's
  original diffusion code for some of its formulas.
- `core/anytop/model/conditioners.py`, which turns joint names into text features with T5,
  carries a Meta Platforms copyright notice. Its code closely follows the conditioning
  module of Meta's AudioCraft (<https://github.com/facebookresearch/audiocraft>), whose
  code is released under the MIT licence.
- `core/anytop/utils/rotation_conversions.py` converts between ways of writing rotations. It
  comes from PyTorch3D by Meta (BSD licence), by way of ACTOR by Petrovich et al.
  (<https://github.com/Mathux/ACTOR>, MIT licence). The PyTorch3D licence text is kept next
  to it in `core/anytop/utils/PYTORCH3D_LICENSE`.

## Code and models fetched by scripts

### The motion library for BVH files

The Truebones preparation code and the Motion2Motion runner read and write BVH files (a
common text format for skeleton animation) with a small Python library by Sigal Raab
(<https://github.com/sigal-raab/Motion>), which its README describes as based on Daniel
Holden's code for "A Deep Learning Framework for Character Motion Synthesis and Editing".
AnyTop installs a fork of it (<https://github.com/inbar-2344/Motion>), and so do we:
`environment.yml` installs that fork at commit `ac23625`.

We could not confirm a licence. Neither repository has a licence file, and the installed
package does not state one. It is not included in this repository.

### T5 text model

AnyTop describes each joint by its name, and turns the names into numbers with Google's T5
model (`t5-base`), through the Hugging Face `transformers` library. The model weights are
downloaded from Hugging Face the first time a method needs them. The model is released under
Apache-2.0, as is the `transformers` library.

### Motion2Motion

Motion2Motion by Chen et al. (arXiv:2508.13139) is one of the evaluated methods. The
authors' program (<https://github.com/LinghaoChan/Motion2Motion_codes>) has no licence, so
we do not copy it. `methods/motion2motion/fetch_official.sh` downloads it at commit
`751d114` into `external/motion2motion/`. It then changes one line on your copy, so that a
query naming a joint the skeleton lacks is recorded as unanswered instead of stopping the
run to wait for typed input.

The direct variant in `methods/motion2motion/core.py` is our own implementation of the
published method, working on the dataset's own motion format. It does not contain the
authors' code.

### General Motion Retargeting (GMR)

The study with six robots uses GMR by Yanjie Ze and colleagues
(<https://github.com/YanjieZe/GMR>) to transfer LAFAN1 human motion to each robot. GMR is
released under the MIT licence. `robots/fetch_gmr.sh` downloads it at commit `bb1bbe4`
into `external/GMR/`.

GMR ships a model of each robot, and each robot model keeps its own licence, which you find
in its folder under `external/GMR/assets/`. For the six robots of the study:

| Robot | Made by | Licence of the robot model |
|---|---|---|
| Unitree G1 | Unitree Robotics | BSD 3-Clause |
| Booster T1 | Booster Robotics | Apache-2.0 |
| Fourier N1 | Fourier | LGPL-3.0 |
| Stanford Toddy | Haochen Shi | MIT |
| EngineAI PM01 | EngineAI | BSD 3-Clause |
| PAL Talos | PAL Robotics | Apache-2.0 |

### The Unitree G1 model from HoloSoma

The human-to-G1 study needs a model of the Unitree G1 robot. BONES-SEED stores each robot
recording as the robot's joint angles, and the model turns those angles into the position
of every body part, which is what the study scores. We use the 29-joint G1 model from
Amazon FAR's HoloSoma (<https://github.com/amazon-far/holosoma>), the model the paper's
numbers were computed with. `robots/human_to_g1/fetch_g1_model.sh` downloads only that
model, at commit `80f1221`, into `external/holosoma/`. HoloSoma is released under
Apache-2.0. The G1 folder has no separate licence file.

This is a different file from the G1 model that comes with GMR, so the two studies' G1
numbers cannot be compared point by point.

### MuJoCo and mink

The robot studies use MuJoCo by Google DeepMind (<https://github.com/google-deepmind/mujoco>)
to compute and draw robot poses, and mink by Kevin Zakka and colleagues
(<https://github.com/kevinzakka/mink>) to solve robot poses when drawing. Both are released
under Apache-2.0 and are installed by `robots/environment.yml`.

### Other Python packages

The two environment files install ordinary Python packages, such as PyTorch, NumPy, SciPy
and matplotlib, each under its own licence. None of their code is copied here.

## Data you download yourself

No motion data is included, and nothing made from it. [docs/data.md](docs/data.md) explains
how to get each dataset.

### Truebones Zoo

The animal motions come from the Truebones Zoo by Truebones Studios
(<https://truebones.gumroad.com/l/skZMC>). Its terms of use, which come with the download,
allow uses such as personal projects, demonstrations and education, but do not allow the
files to be resold, passed on, or used in video games.

### BONES-SEED

The human motions of the G1 study and their G1 versions come from BONES-SEED by Bones Studio
(<https://bones.studio/>), available after accepting its licence. It is licensed under the
BONES Motion Capture Dataset License Agreement, which does not allow the data, or models
whose output could stand in for it, to be passed on. Work that uses it must carry this
credit:

> Training data includes Motion Data by Bones Studio (https://bones.studio/). Use of the
> underlying dataset is subject to the BONES Motion Capture Dataset License Agreement.

### LAFAN1

The human motions of the six-robot study come from LAFAN1 by Ubisoft La Forge
(<https://github.com/ubisoft/ubisoft-laforge-animation-dataset>). It is licensed under
Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 (CC BY-NC-ND 4.0), which does
not allow sharing anything made from it.
