# The robot studies

The paper's main results are about animals, where no one knows which target motion truly
belongs to which source clip. These two studies ask the same question of humanoid robots,
where the answer can be checked: for every human clip there is a known robot counterpart.
Several people perform the same action; if a model keeps what made each performance
different, its robot motions differ from one another the way the human recordings do.
That is what Source-Instance Fidelity (SIF) measures, computed here by the same `sif`
package as for the animals.

* `human_to_g1/`: human motion capture from BONES-SEED to one Unitree G1 humanoid. The
  collection ships a G1 version of every human take, so the counterpart comes with the data.
* `lafan1_to_six_robots/`: the LAFAN1 recordings to six humanoid robots of different sizes
  and builds, with the counterparts computed by a retargeting tool.
* `render/`: the pictures and videos behind the paper's robot figure.

**The step-by-step guide is [docs/robots.md](../docs/robots.md)**: what to download, which
environment and hardware each step needs, how long it takes, and which paper table each
result file feeds.

## Two environments

Training and scoring run in the repository's main environment. Selecting and converting
the BONES-SEED clips, retargeting LAFAN1 and drawing the figure need a newer Python and
MuJoCo, a library that simulates and draws robots, and run in `robots/environment.yml`
(`conda env create -f robots/environment.yml`, then `conda activate motion-robots`).
Pulling the BONES-SEED clips out of their archives runs in either environment. Only
training and drawing need a GPU.

## What was fixed before the studies, and what was added later

For the human-to-G1 study, SIF (with motions compared raw and length-controlled) and
variation (how far apart a model's robot motions for one action are, relative to how far
apart their human sources are), and the unpaired, averaging and true-pair models, were
fixed before any SIF number on this data was seen. Action-level AUC, realism and the
adversarial model were added later, before any of their numbers were seen. The adversarial
model's training did not converge; a second version with different training settings,
fixed before looking at its results, was run with three seeds, and both are reported.

For the six-robot study, SIF and variation, with five predictions and their thresholds,
were fixed before any trained model was scored; only the true retarget and random clip
rows on the Unitree G1 had been seen, in a first trial run. Action-level AUC and realism were
computed afterwards. The predictions and their outcomes are listed in the paper's appendix
on the robot studies (the table of six-robot predictions), which
`python -m paper.make_tables` rebuilds from `results/robots/six_robots_sif.json`.

## Data and licences

No trained models are included, and no motion data and no file of poses or joint
positions made from the recordings; the steps rebuild everything from the original
downloads.

* **BONES-SEED** (human-to-G1 study), from Bones Studio under the BONES Motion Capture
  Dataset License Agreement. Training data includes Motion Data by Bones Studio
  (https://bones.studio/). Use of the underlying dataset is subject to the BONES Motion
  Capture Dataset License Agreement.
* **LAFAN1** (six-robot study), from Ubisoft La Forge under CC BY-NC-ND 4.0.

Two pieces of other people's code are fetched by script at a fixed commit, not copied. A
robot description is the set of files that says what a robot is made of: its links, their
lengths and shapes, and how its joints connect and how far they turn.

* The Unitree G1 description of the human-to-G1 study comes from HoloSoma, an open-source
  humanoid robot project from Amazon, and is fetched by
  `robots/human_to_g1/fetch_g1_model.sh`. HoloSoma is released under Apache-2.0, but its
  G1 folder carries no licence of its own, which is why it is fetched rather than copied.
* General Motion Retargeting (MIT) is fetched by `robots/fetch_gmr.sh`. Each robot
  description inside it carries its own licence: BSD 3-Clause for the Unitree G1 and the
  EngineAI PM01, Apache-2.0 for the Booster T1 and the PAL TALOS, MIT for the Stanford
  Toddy, and LGPL 3.0 for the Fourier N1.

The two studies use two different descriptions of the Unitree G1: the HoloSoma one, which
matches the rig (the robot model, with its exact joints and body sizes) the BONES-SEED G1
recordings were made on, and the retargeting tool's own.
The robot is the same machine, but the two studies' G1 numbers are not directly
comparable point by point.
