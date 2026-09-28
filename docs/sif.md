# Scoring motions with SIF

Source-Instance Fidelity (SIF) tells you whether a retargeting method keeps what made each
source clip different, or whether it returns much the same motion whatever clip it is given.
It works on any motions, not only the ones in the paper. All it needs is a set of source clips
and the outputs a method produced for them.

This guide explains how to install the toolkit, how to describe your motions to it, and how
to read the numbers it reports. It ends with a small example you can run in a few seconds.

## Install

The toolkit needs Python 3.8 or newer and NumPy. From the repository root:

```bash
pip install -e .
```

This installs the `sif` package and the `sif-score` command. To run the tests as well:

```bash
pip install -e ".[test]"
python -m pytest tests
```

## Describing your motions: groups

SIF compares clips in small groups. A group is a few source clips that share a source body and
an action, together with the output your method produced for each of them on one target body.
The paper calls such a group a triple, after the source body, target body and action it holds
fixed. Holding these three things fixed matters: it means the outputs can only differ because
their source clips differ, not because they show a different action or a different body.

Each clip is a NumPy array of joint positions with shape `(frames, joints, 3)`. Arrays with
more values per joint are accepted, and only the first three are used, so processed Truebones
motion files can be given as they are. Within a group, all source clips must use the same body,
and so must all outputs. Clips in a group may have different numbers of frames. The toolkit
does not check that the bodies match: pairs whose clips have different joint counts, or, in
raw scoring, fewer than four shared frames, are left out without a warning.

The outputs must be listed in the same order as their sources: the first output is the one
your method produced from the first source clip, and so on.

A group needs at least three clips to be scored. With only two clips there is a single
distance, and a single number cannot show whether two sets of distances agree. Groups with
fewer clips are skipped. Groups with more than six clips are cut to their first six, so that
the shuffle test described below can try every possible pairing.

Each group can also carry a block name, explained in its own section below. In short, give
groups the same block name when they reuse the same source clips, for example by using the
source body's name.

## Scoring from Python

```python
import numpy as np
from sif import score_groups

rng = np.random.default_rng(0)
groups = []
for g in range(10):
    sources = [rng.normal(size=(50, 20, 3)).cumsum(axis=0) for _ in range(4)]
    outputs = [s[:, :12] + rng.normal(size=(50, 12, 3)) for s in sources]
    groups.append({"sources": sources, "outputs": outputs, "block": f"body_{g % 5}"})

result = score_groups(groups, length="both")
r = result["raw"]
print(r.sif, r.ci, r.p, r.variation, r.n_groups)
```

Here the made-up outputs follow twelve of the twenty source joints with some added noise, so
they keep part of what makes each source clip different.

`score_groups` returns one result for raw scoring and one for length-controlled scoring, under
the keys `"raw"` and `"fixed"` (see below). Pass `length="raw"` or `length="fixed"` to get a
single result instead. Each result holds the average SIF (`sif`), the 95% interval (`ci`), the
shuffle-test p-value (`p`), the variation (`variation`) and the number of groups scored
(`n_groups`). The list `groups` holds the score of each group, with its own SIF, variation,
number of clips and block name.

The settings used in the paper are the defaults: 10,000 shuffles, 10,000 resamples for the
interval, and random seed 42. You can change them with the arguments `n_shuffles`, `n_boot`
and `seed`. The same inputs and settings always give the same numbers. The smallest and
largest number of clips per group are set by `min_clips` and `max_clips`. The Truebones
evaluation sets `min_clips=2`, as in the paper, so that a group reduced to two clips still
counts, with a score of zero.

## Scoring from the command line

If your clips are saved as `.npy` files, list the groups in a JSON file:

```json
{"groups": [
    {"block": "Horse",
     "sources": ["horse/attack_1.npy", "horse/attack_2.npy", "horse/attack_3.npy"],
     "outputs": ["out/q1.npy", "out/q2.npy", "out/q3.npy"]}
]}
```

File names are read relative to the folder that holds the JSON file. The block name may be
left out. Then score them:

```bash
sif-score groups.json --out results.json
```

The command prints one line for raw scoring and one for length-controlled scoring. With
`--out`, it also writes the same numbers to a JSON file, together with the score of every
group. Use `--length raw` or `--length fixed` for only one of the two, and `--shuffles`,
`--boot` and `--seed` to change the settings above. `sif-score --help` lists every option.

## What the numbers mean

SIF. For each group, the toolkit measures how far apart every pair of source clips is, and how
far apart the matching pair of outputs is. SIF is the correlation between these two sets of
distances. If two source clips that are far apart lead to outputs that are far apart, and two
close source clips lead to close outputs, SIF is near one. If the outputs ignore their sources,
SIF is near zero. A negative value means the outputs tend to differ where the sources are
alike. The reported SIF is the average over all groups.

The distance between two clips on the same body is measured frame by frame. Each pose is
centred, then turned and resized to fit the matching pose of the other clip as closely as
possible, and what remains is the error. So where the body stands, which way it faces and how
big it is do not count; only the shape of the pose over time does. If every output in a group
is the same, the output distances are all equal and that group's SIF is exactly zero.

The 95% interval. Groups are not always independent: the same source clips often appear in
many groups. The interval therefore redraws whole blocks at random, with repeats allowed,
recomputes the average SIF each time (10,000 times by default), and reports the range that
holds the middle 95% of these averages. With few blocks the interval is rough, and with a
single block it shrinks to one value and says nothing; use several blocks.

The shuffle-test p-value. This asks how often an average SIF this high would appear if the
outputs had nothing to do with their sources. The test repeatedly pairs each group's outputs
with its source clips in a random order, with the true order among the possibilities, and
recomputes the average SIF. Groups in the same block are shuffled together. The p-value is the
share of these shuffles that reach the observed average, counting the observed one itself, so
it is never exactly zero: with the default 10,000 shuffles the smallest possible value is about
0.0001. The test is one-sided; it only asks whether SIF is higher than chance.

Variation. SIF is a correlation, so it does not tell you how much the outputs actually differ.
Variation does: it is the average distance between outputs divided by the average distance
between their source clips, and the reported value is the median over groups. It is near one
when outputs vary as much as their sources and zero when they collapse to a single motion.
Source and output distances are measured on different bodies, each in its own units, so
values above or below one partly reflect the difference between the bodies. Variation is most
useful for comparing methods on the same targets, and a value near zero is a clear sign that a
method returns one motion for everything.

The source-blind floor. A method that ignores its source scores near zero; the paper calls
this the source-blind floor. The paper reads the numbers this way: a method is at the floor
when its SIF does not stand out from the shuffles under both raw and length-controlled
scoring, and near the floor when it does stand out but its SIF stays between -0.10 and +0.10.
That band comes from the synthetic setting, where methods known to ignore their sources score
within it ([docs/synthetic.md](synthetic.md)).

## Raw and length-controlled scoring

Clips often have different numbers of frames. The toolkit offers two ways to compare them, and
the paper reports both.

In raw scoring, each clip keeps its own length, and every pair is compared over the frames the
two clips share, starting from the first frame. A pair needs at least four shared frames.

In length-controlled scoring, every clip is first stretched or squeezed to 64 frames. How long
a clip is then carries no information, so a method cannot score well just by matching the
length of its source.

If the two scores agree, clip length is not what drives the result. If they disagree, look at
both before drawing a conclusion. In the code and on the command line, length-controlled
scoring is called `fixed`.

## Blocks

A block is a set of groups that should be treated as one unit when judging how certain the
average SIF is. Groups belong in the same block when they reuse the same or related source
clips. On the Truebones evaluation, for example, each source clip is retargeted to many target
bodies, so the groups of one source body are far from independent. There the block is the
source body.

Set a block whenever your groups share source clips. If you leave it out, each group is its own
block. When groups do share clips, that treats them as more independent than they are, and the
interval and p-value will look more certain than the data allow. Blocks change only the
interval and the p-value, never the SIF itself.

## A small example

This example makes toy motions for four source bodies and two actions each, with four source
clips per action, and three made-up methods that produce motions on a smaller body:

- `keeps` follows twelve of the twenty source joints, so it keeps what makes each clip
  different;
- `ignores` returns the same motion for every clip of an action;
- `unrelated` returns a plausible motion of the right action that comes from a different clip.

Save the following as `make_example.py` in any empty folder and run it there:

```python
import json
from pathlib import Path

import numpy as np

rng = np.random.default_rng(0)
folder = Path("sif_example")
folder.mkdir(exist_ok=True)
t = np.linspace(0, 2 * np.pi, 60)[:, None, None]
methods = {"keeps": [], "ignores": [], "unrelated": []}
for body in ["Horse", "Cat", "Bird", "Crab"]:
    for action in ["walk", "jump"]:
        pose, style = rng.normal(size=(20, 3)), rng.normal(size=(20, 3))
        names = []
        for k in range(4):
            name = f"{body}_{action}_{k}"
            source = pose * np.sin(t) + rng.uniform(0.2, 1.5) * style * np.sin(2 * t)
            other = pose * np.sin(t) + rng.uniform(0.2, 1.5) * style * np.sin(2 * t)
            np.save(folder / f"{name}.npy", source)
            np.save(folder / f"{name}_keeps.npy", source[:, :12])
            np.save(folder / f"{name}_ignores.npy", pose[:12] * np.sin(t))
            np.save(folder / f"{name}_unrelated.npy", other[:, :12])
            names.append(name)
        for method, groups in methods.items():
            groups.append({"block": body,
                           "sources": [f"{n}.npy" for n in names],
                           "outputs": [f"{n}_{method}.npy" for n in names]})
for method, groups in methods.items():
    (folder / f"{method}.json").write_text(json.dumps({"groups": groups}, indent=2))
```

Then create the clips and score each method:

```bash
python make_example.py
sif-score sif_example/keeps.json
sif-score sif_example/ignores.json
sif-score sif_example/unrelated.json
```

You should see:

```
              raw: SIF +0.998  95% CI [+0.996, +0.999]  p = 0.0001  variation 0.891  (8 groups)
length-controlled: SIF +0.998  95% CI [+0.996, +0.999]  p = 0.0001  variation 0.891  (8 groups)
              raw: SIF +0.000  95% CI [+0.000, +0.000]  p = 1.0000  variation 0.000  (8 groups)
length-controlled: SIF +0.000  95% CI [+0.000, +0.000]  p = 1.0000  variation 0.000  (8 groups)
              raw: SIF -0.135  95% CI [-0.361, +0.091]  p = 0.7910  variation 0.347  (8 groups)
length-controlled: SIF -0.135  95% CI [-0.360, +0.091]  p = 0.7910  variation 0.348  (8 groups)
```

The method that keeps each clip's differences scores close to one, well above every shuffle.
The method that returns one motion per action scores exactly zero with no variation. The
method that returns an unrelated clip of the right action also sits at the source-blind floor,
even though its outputs do vary: it shows the right action, but not the motion of the clip it
was given. This is the case an action-level test cannot see and SIF can.

For a second toy example, written as a single script, run `python examples/quickstart.py`.

## Scoring the Truebones evaluation

To score a method on the 1,891 Truebones combinations used in the paper, which builds the
groups and blocks for you, see [docs/benchmark.md](benchmark.md).
