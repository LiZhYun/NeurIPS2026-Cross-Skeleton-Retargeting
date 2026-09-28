# One motion capture set to six humanoid robots

The human-to-G1 study asks the paper's question of one robot. This one asks it of six, so
the answer cannot be a quirk of a single machine. The robots are the Unitree G1, the
Booster T1, the Fourier N1, the Stanford Toddy, the EngineAI PM01 and the PAL TALOS; they
differ in height, in how many joints they have and in how their limbs are arranged.

The human recordings are LAFAN1, in which a handful of people each perform a set of
numbered routines: dance1, fight1, aiming2 and so on. A routine performed by several
different people is what makes the paper's question answerable, because those performances
are exactly the instances whose differences should survive retargeting. Each routine is one
action group, the set of clips scored together. The true robot counterpart of each
performance is computed by General Motion Retargeting, which solves, frame by frame, for
the joint angles that put the robot's body where the person's was.

Two settings are reported side by side. In the **sparse** setting an action group keeps at
most three performers and one window each, which is about as much data per group as the
animal set has, and is the setting the paper's argument is about. In the **dense** setting
an action group keeps every performer and both windows.

The full walk-through, with what every output means, is in
[docs/robots.md](../../docs/robots.md).

## Getting the recordings

LAFAN1 is released under Creative Commons Attribution-NonCommercial-NoDerivatives 4.0,
which forbids redistributing the recordings or adapted versions of them. This repository
contains no motion data and no file of poses or joint positions made from them. Download
the 77 motion files, which are BVH files (a common text format for skeleton motion), from

    https://github.com/ubisoft/ubisoft-laforge-animation-dataset

and unpack them into `data/robots/lafan1_to_six_robots/lafan1_bvh/`, so that each file is
named like `dance1_subject1.bvh`.

## Steps

Run everything from the repository root. "Main" is the repository's main environment
(`environment.yml` at the root); "robot" is `robots/environment.yml`, which holds MuJoCo,
a library that simulates and draws robots. The retargeting steps need the retargeting
tool, fetched once with `bash robots/fetch_gmr.sh`, on the import path
(`export PYTHONPATH=external/GMR`).

| step | environment | GPU | time |
| --- | --- | --- | --- |
| `choose_windows` | robot | no | seconds |
| `retarget --robot <robot> ...` for each of the six robots | robot | no | 4 to 12 minutes per robot with 16 processes |
| `retarget_retry` | robot | no | only if something is missing |
| `build_tensors` | robot | no | about two minutes |
| `train` | main | yes | a few minutes per robot and setting |
| `score --out output/robots/six_robots_sif.json` | main | no | about ten minutes |
| `score_four_measures --out output/robots/six_robots_four_measures.json` | main | no | about ten minutes with 12 processes |
| `random_clip_auc --out output/robots/six_robots_random_clip_auc.json` | main | no | about ten minutes |

Each step is `python -m robots.lafan1_to_six_robots.<step>`. The retargeting loop over the
six robots is:

```
for R in unitree_g1 booster_t1_29dof fourier_n1 stanford_toddy engineai_pm01 pal_talos; do
  python -m robots.lafan1_to_six_robots.retarget --robot $R \
      --takes data/robots/lafan1_to_six_robots/takes.json \
      --bvh_dir data/robots/lafan1_to_six_robots/lafan1_bvh \
      --out_dir data/robots/lafan1_to_six_robots/retargeted --processes 16
done
```

The retry step exists because the solver (the numerical optimiser that finds each frame's
joint angles) occasionally crashes on a recording; running many at once can then lose a
process's whole share of the work silently. It redoes whatever is missing, one process per
recording, so a crash is visible and costs only that recording.

The three models are the same as in the human-to-G1 study, and so are their training
settings. Only the number of joints changes from robot to robot. The adversarial model is
not repeated here. No trained models are distributed; `train` saves each one under
`data/robots/lafan1_to_six_robots/checkpoints/`.

## Notes

Before it scores anything, `score` checks that every model produced a motion for every
clip, of the right shape and with no missing values, and stops if not. Otherwise a model
with a few missing motions would quietly be judged on an easier subset than the others.

An action group needs at least three clips to be scored, because two clips give a single
distance and no correlation. An action group with more than six clips is cut to six, so
every possible way of handing the outputs to the sources can be tried in the shuffle test. That test asks whether a model's SIF is higher than it would be if each
group's outputs were handed to their sources in a random order, which is the score of a
model that ignores its source.

Every row is scored over the same first 256 frames, which is the length the models
produce, so no row is judged over more of a clip than another.
