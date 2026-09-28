# Getting the data

No motion data comes with this repository. The animal study uses the Truebones Zoo, which
you download and then convert once into the form the code reads. The two robot studies use
BONES-SEED and LAFAN1; how to get them is noted at the end, and how to use them is in
[robots.md](robots.md).

## Truebones Zoo

The Truebones Zoo is a collection of animal animations, from snakes and spiders to birds,
horses and dinosaurs, each animal with its own skeleton. The paper uses 70 of its animals.
Every method in the paper is trained on it, and the Truebones evaluation is built from it.

### Download

Download the zoo from [Truebones](https://truebones.gumroad.com/l/skZMC). Its terms of use
allow uses such as education and demonstrations, but do not allow the files to be resold or
passed on, which is why they are not included here.

The download is a zip file of about 1 GB, which unpacks to about 3 GB. Unpack it inside the
repository, from the repository root:

```bash
mkdir -p datasets/truebones/zoo
unzip <zip> -d datasets/truebones/zoo
```

where `<zip>` is the downloaded file. Each animal then has its own folder here:

```
datasets/truebones/zoo/Truebone_Z-OO/Alligator/
datasets/truebones/zoo/Truebone_Z-OO/Anaconda/
...
datasets/truebones/zoo/Truebone_Z-OO/Tyranno/
```

Each animal's folder holds its motions as BVH files (a common text format for skeleton
animation), plus FBX files and textures that the code does not use. Many animals also have a
rest-pose file, whose name contains "TPOSE".

### Convert

From the repository root:

```bash
python -m core.truebones.create_dataset
```

This reads the BVH files and writes the processed dataset to
`dataset/truebones/zoo/truebones_processed/`. Note the two similar names: the raw download
goes under `datasets/` and the processed files appear under `dataset/`. The command takes no
options; it always uses these two folders, so it has to be run from the repository root.

The conversion takes two to four hours, so on a shared machine it is best run as a batch
job; ask for at least five hours, because an interrupted conversion has to start again. When
it finishes, the processed folder holds about 850 MB.

Each animal is first placed in a common frame: turned to face the same way, moved to the
origin, scaled so that its average bone has a common length, and set on the ground. The
rest-pose file sets these for each animal; where there is none, one of its motions is used
instead, an idle motion where there is one. For six animals (Lion, Monkey, Pteranodon, Rat,
SabreToothTiger and Trex) that motion is fixed to the one the paper's data were built from.
Long motions are cut into pieces of 200 frames, with the last piece up to 240 frames. The
result is 1,070 clips from 70 animals. We repeated the conversion from a fresh copy of this
repository: every clip matched ours to within about one millionth, the size of rounding
differences.

### Give the clips the paper's names

Each clip's name ends in a running number, such as the 18 in `Alligator___Walk2_18`. The
conversion hands out these numbers in its own order, which is not the order of the paper's
run, so your clips are numbered differently from ours. The Truebones evaluation
refers to clips by the paper's names, so after the conversion run, from the repository root:

```bash
python -m core.truebones.use_paper_clip_names
```

This renames the files in `motions/`, `bvhs/` and `animations/` to the paper's names. It
only changes the running numbers, never what is inside the files. It uses
`core/truebones/paper_clip_names.json`, which lists, for each animal and raw BVH file, the
paper's name of every clip cut from that file, in time order. At the end it prints how many
clips it matched and lists anything it could not match; after a complete conversion that list
is empty. Running it a second time changes nothing.

### Measure the clip features

A few parts of the code describe each clip by what its movement achieves rather than by its
joints: the path the body travels, how fast it moves forward, which limbs touch the ground,
its step rate, and how the movement is shared among the limbs. We call these the clip
features. ANCHOR, the two random baselines, the AL-Flow models and the action-level test read
them from one file, which you make once, after the clips have the paper's names:

```bash
python -m benchmark.clip_features --build
```

This takes about a minute and writes `save/clip_features.npz`. If the file is missing, the
commands that need it stop with a message giving this command.

In 16 clips several step rates are exactly equally strong, and which one comes out on top
depends on the computer's rounding. For those clips the build takes the step rate the paper
used from `benchmark/rhythm_ties.json`, so the rebuilt file matches the paper's. We checked
this on three kinds of processor, one AMD and two Intel, and after a fresh conversion: the
step rates and ground contacts came out identical, and the other features agreed to within
rounding.

### What the processed files are

- `motions/` one file per clip, such as `Alligator___Bite6_1.npy`. Each holds an array of
  shape `(frames, joints, 13)`. For every joint and frame, the 13 numbers are its position
  relative to the body's root (3), its rotation (6), its velocity (3) and whether it touches
  the ground (1). The name is the animal, the original motion name and a running number. The
  Truebones evaluation refers to clips by these names.
- `cond.npy` a description of each animal's skeleton: which joint hangs from which, the bone
  lengths, the joint names, how far apart joints are along the skeleton, the rest pose, and
  the average and spread of its motion values, which the models use to rescale their inputs.
- `bvhs/` each clip again as a BVH file, after the common placing and scaling. These are
  handy for viewing the clips in animation software.
- `animations/` a short video of each clip, drawn from the processed values, so you can see
  at a glance that the conversion went well.
- `metadata.txt` the number of clips for each animal and in total. A complete conversion
  lists 1,070 clips and 104,305 frames.
- `positions_error_rate.txt` a list of the BVH files that were read. Its error column is
  always zero, because the check behind it is switched off in AnyTop's code.

The methods, the evaluation and the scoring read only `motions/` and `cond.npy`.

### Trying one animal first

To see the conversion work before starting the full run, you can convert a single animal
into a folder of your choice. For the chicken, which has three motions:

```bash
python -m core.truebones.process_new_skeleton --object_name Chicken \
    --bvh_dir datasets/truebones/zoo/Truebone_Z-OO/Chicken \
    --tpos_bvh datasets/truebones/zoo/Truebone_Z-OO/Chicken/Chicken_TPOSE.bvh \
    --face_joints_names Bip01_R_Thigh Bip01_L_Thigh BN_Finger_R_01 BN_Finger_L_01 \
    --save_dir chicken_check
```

This takes a few minutes and writes the same kinds of files as above. The four joint names
tell the code which way the animal faces (right hip, left hip, right shoulder, left
shoulder, or the closest match). The clips come out with the same values as in the full
conversion, but their running numbers start from one, so the evaluation cannot use this
folder. Use the full conversion for everything else.

## BONES-SEED

BONES-SEED is a collection of human motion capture from Bones Studio. Each human
recording comes with a version already transferred to the Unitree G1 humanoid robot. The
paper uses it for the human-to-G1 study.

Access is through [Bones Studio](https://bones.studio/), under the BONES Motion Capture
Dataset License Agreement. Two of its terms matter here. The data, and models whose output
could stand in for it, may not be passed on; this is why no BONES-SEED motions or trained
robot models are included. And work that uses it must carry this credit:

> Training data includes Motion Data by Bones Studio (https://bones.studio/). Use of the
> underlying dataset is subject to the BONES Motion Capture Dataset License Agreement.

Which parts of the collection to download and where to put them is in
[robots.md](robots.md).

## LAFAN1

LAFAN1 is a set of 77 human motion capture recordings from Ubisoft La Forge. The paper uses
it for the study with six robots. Download it from
[its GitHub page](https://github.com/ubisoft/ubisoft-laforge-animation-dataset). It is
licensed under Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 (CC BY-NC-ND
4.0), which does not allow sharing the recordings or anything made from them, so nothing
derived from LAFAN1 is included here. Where to unpack it and how it is used is in
[robots.md](robots.md).
