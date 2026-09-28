# Human motion capture to one Unitree G1

The BONES-SEED collection ships, for every human recording, a Unitree G1 version of the same
take. That gives something the animal data cannot: for every human clip there is a robot
clip that is its counterpart. The study uses that to ask whether a model trained to turn
human motion into robot motion keeps what made each performance different, and to check
that Source-Instance Fidelity (SIF) recognises a genuine counterpart when it sees one.

The collection is much larger than the animal data and mostly walking, so using it whole
would answer an easier question. The study instead draws a small, deliberately varied
sample: ninety actions with three or four recordings each, at most thirty per cent of them
locomotion. The recordings of one action form an action group, the set of clips scored
together. That is roughly the shape of the animal data, which is the setting the paper's
argument is about.

The full walk-through, with the files to download and what every output means, is in
[docs/robots.md](../../docs/robots.md).

## Rows of the tables

| row | what it is |
| --- | --- |
| true retarget | the clip's own robot counterpart, the G1 version of the same take. It shows SIF recognises a real correspondence. |
| random clip | the robot counterpart of a different clip of the same action, drawn at random. It shows what knowing only the action name is worth. |
| unpaired objective | one autoencoder per skeleton (a network that squeezes a motion into a short code and rebuilds it), sharing one space of codes; neither ever sees the other skeleton, so nothing in training says which human clip a robot motion belongs to. |
| averaging objective | asked to match a robot clip drawn at random from the same action rather than the true counterpart. The best answer to that request is the action's average motion. |
| model trained on true pairs | the same model asked to match the true counterpart. It shows what the model can do when the correspondence is given rather than guessed. |
| adversarial objective | a model trained like ACE, one of the methods the paper evaluates: against a critic that tells real robot motion from produced motion, and against a summary of the source motion. |

## Steps

Run everything from the repository root. "Main" is the repository's main environment
(`environment.yml` at the root); "robot" is `robots/environment.yml`.

| step | environment | GPU | time |
| --- | --- | --- | --- |
| `bash robots/human_to_g1/fetch_g1_model.sh` | either | no | a minute |
| `select_clips --metadata <seed_metadata_v004.parquet>` | robot | no | under a minute |
| `extract_clips --human_archive <soma_uniform.tar.gz> --robot_archive <g1.tar.gz>` | either | no | about half an hour |
| `convert_to_tensors` | robot | no | about two minutes |
| `train --model all` | main | yes | about three minutes for all three models |
| `train_adversarial` | main | yes | under two minutes |
| `train_adversarial_second_setup --out output/robots/g1_adversarial_second_setup.json` | main | yes | about five minutes per seed, most of it scoring |
| `generate --checkpoint <model.pt> --out_dir <folder>` | main | optional | seconds |
| `score_four_measures --length raw --out output/robots/g1_four_measures_raw.json` | main | no | about twenty minutes |
| `score_four_measures --length length_controlled --out output/robots/g1_four_measures_length_controlled.json` | main | no | about twenty minutes |
| `random_clip_auc --out output/robots/g1_random_clip_auc.json` | main | no | about five minutes |
| `score --length raw --out <file.json>` | main | no | a few minutes |

Each step after the first is `python -m robots.human_to_g1.<step>`. Everything reads and
writes the data folder `data/robots/human_to_g1/` unless `--data_dir` says otherwise.

`clips.json`, the list of actions and clips, is not shipped: it copies parts of the
collection's metadata, which the collection's licence treats as confidential. `select_clips`
rebuilds it exactly from `seed_metadata_v004.parquet`, the metadata table every user of
the collection downloads.

On our GPU, `train --model all` took 173 seconds for its three models: about a minute
and a half for the unpaired objective and half a minute for each of the other two.

No trained models are distributed. `train` and `train_adversarial` save each model they
train in `data/robots/human_to_g1/checkpoints/`, and `generate` writes a saved model's
motions again. On our GPU, retraining with the settings given here reproduces the paper's
motions bit for bit (our GPU is an NVIDIA H200); on other hardware small differences are
possible.

`score` computes SIF and variation on their own, which is useful when the slower action
and realism measures are not needed. Variation is how far apart a model's robot motions
for one action are, relative to how far apart their human sources are.

## Notes

The robot description (the files that say what the robot's links and joints are and how
they connect) comes from HoloSoma, an open-source humanoid robot project from Amazon.
HoloSoma is released under Apache-2.0, but its G1 folder carries no licence of its own, so
`fetch_g1_model.sh` fetches it rather than copying it into this repository. It is a
different file from the Unitree G1 description the six-robot study uses, which comes with
the retargeting tool; the robot is the same machine but the two studies' G1 numbers are not
directly comparable point by point.

Both sides are thinned from 120 to 30 frames per second from the same starting frame, so a
clip's human and robot files stay aligned frame for frame. Every later step relies on
that.

Training data includes Motion Data by Bones Studio (https://bones.studio/). Use of the
underlying dataset is subject to the BONES Motion Capture Dataset License Agreement.
