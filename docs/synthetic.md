# The synthetic setting

On real animal motion nobody knows the one correct target motion for a source clip, so there
is no right answer to hold Source-Instance Fidelity (SIF) against. The synthetic setting is a
small made-up world where the right answer is written down. It serves two purposes in the
paper. It calibrates SIF: the true map should score near one, and answers that ignore the
source clip should score near zero. And it tests the conditional-mean effect directly: a model
trained on squared error answers with an average, and more data does not change that.

This guide builds the world, runs the calibration, trains and scores a small generator,
measures the simple answers that need no training, and runs the conditional-mean ladder.
Nothing here needs the Truebones Zoo. All commands are run from the repository root, and the
code is in [`synthetic/`](../synthetic).

## The made-up world

There are eight stick-figure skeletons, chains of 8 to 22 joints, and six actions such as
walking, falling and flying. Each clip is 64 frames long. The rule that carries a clip from
one skeleton to another spreads the source joints evenly along the target chain and then bends
time by an amount that depends only on the action. That rule is the true map: for every
source clip we know exactly what the correct target clip is.

The world comes in four versions, which is why the paper calls it the 2×2 setting. It crosses
two choices:

| Version | Clips per skeleton and action | True pairs |
|---|---|---|
| `sparse_unpaired` | one | not available |
| `sparse_paired` | one | available |
| `dense_unpaired` | fifty | not available |
| `dense_paired` | fifty | available |

With fifty clips, the clips of one skeleton and action differ in timing and in how large the
movement is, so each clip differs from the others while the action stays recognisable.

## Build the data

```bash
python -m synthetic.build_dataset
```

This takes under a minute on a CPU and writes the four versions to `save/synthetic_2x2/`,
about 270 MB in all. Each version's folder holds the clips (`clips.npy`), a list saying which
skeleton and action each clip is (`meta.json`) and, in the paired versions, the correct target
clip for each pair (`transport.npy`) with the list of pairs (`pairs.json`). The data are made
from a fixed seed, so every run writes the same files.

## Calibrate SIF

```bash
python -m synthetic.calibrate_sif --cell save/synthetic_2x2/dense_paired \
    --out results/synthetic/sif_calibration.json
```

This uses the dense, paired version and scores three kinds of output for every group of
source clips that share a source skeleton, a target skeleton and an action:

- the true map, which gives each source clip its correct target clip;
- a random clip of the same action on the target skeleton, which ignores the source;
- random noise the size of a real clip, which also ignores the source.

It takes about three minutes on a CPU. Of the 336 groups with at least three source clips,
168 are left out because their source clips are identical once rotation and scale are
removed, so they have no differences to keep and SIF is undefined there. On the other 168:

| Output | SIF | 95% interval | Variation |
|---|---|---|---|
| True map | +0.993 | +0.992 to +0.995 | 0.94 |
| Random clip of the same action | +0.002 | −0.027 to +0.029 | 0.96 |
| Random noise | +0.021 | −0.008 to +0.054 | 1.3 × 10⁵ |

Variation is how far apart the outputs of a group are compared with how far apart their
sources are. Noise has no motion structure at all, so its variation is huge.

The two source-blind answers set the scale the paper uses on real data: a method whose SIF
stands out from shuffled pairings but stays within 0.10 of zero is called near the
source-blind floor. The command above writes over the file the paper's table and figure read,
and gives the same values; use another `--out` to keep the shipped file untouched.

## Train and evaluate a small generator

```bash
python -m synthetic.train_and_evaluate --all --max_steps 10000
```

This trains the same small generator on each of the four versions in turn and asks it for the
target clip of source clips it never saw. The generator is a small transformer that starts from
noise and moves towards a motion, told which skeleton and action to aim for; where true pairs
exist it is also asked to match them. Half the clips of each skeleton and action are held back
as questions. With one clip per skeleton and action there is nothing to hold back, so every
clip is used as a question, and the test is whether the answer carries across to skeleton
pairs the training never showed.

The script needs a GPU; it has no option to run on the CPU. The four versions take five to
seven minutes together on one H200 GPU. The error of each version is printed and written to
`metrics_recovery.json` in that version's folder, and all four together to
`save/synthetic_2x2/summary_2x2.json`. To train on one version only, give
`--cell dense_paired` (or another version's name) instead of `--all`. Without
`--max_steps` it trains for 2,000 steps.

Training on a GPU is not exactly repeatable, so a rerun gives slightly different errors. In
our run the errors, averaged over actions, were 5.3 and 5.1 for the two sparse versions and
5.3 and 4.7 for dense unpaired and dense paired. The paper does not print these numbers.

## Answers that need no training

```bash
python -m synthetic.oracle_baselines
```

An error on its own means little, so this script measures the same error for five answers
that need no training. It takes under a minute on a CPU and writes
`save/synthetic_2x2/oracle_baselines_summary.json`. The results are exactly repeatable:

| Answer | Sparse versions | Dense versions |
|---|---|---|
| A motionless figure | 23.4 | 22.0 |
| Random noise | 24.4 | 23.0 |
| A random clip of the same skeleton and action | 5.4 | 5.3 |
| The source-blind average: the average correct target for that skeleton and action | 2.0 | 2.1 |
| The true map | 0 | 0 |

The source-blind average matters most. It is the best any method can do without paying
attention to which source clip it was given. The paper does not print these numbers either;
they are here so the trained errors above can be read against them.

## The conditional-mean ladder

```bash
python -m synthetic.conditional_mean_ladder --seed 42
```

A model trained to minimise squared error is trained to answer with an average. When many
source clips share one description, it gives nearly the same answer for all of them, and its
answers stop depending on which clip was asked about. The obvious objection is that this is
only a shortage of data. The ladder tests that objection. It trains the same small network
with 2, 4, 8, 16, 32 and 50 clips for every combination of source skeleton, target skeleton
and action (336 combinations). As in the setting the paper describes, each source clip is
trained against the correct target of another clip from the same combination, picked afresh
at every training step: the network sees targets of the right skeleton and action, but never
the one that belongs to the clip it was given. At each rung it asks about eight source clips
per combination that the training never saw, and compares how much the answers vary from one
source clip to the next with how much the correct answers vary. A ratio of one would mean the
model keeps all the variation that belongs to the source.

Each seed takes about a minute on one GPU; the script uses a GPU when there is one and the CPU
otherwise. It writes `save/synthetic_2x2/conditional_mean_ladder_seed42.json`. The
paper reports seeds 42, 43 and 44, so run it again with `--seed 43` and `--seed 44`. The
paper's table averages the three seeds from the files in `results/synthetic/`.

The ratio stays at about 0.02 at every rung: the answers keep about 2% of the variation that
belongs to the source, however many clips each combination has. The seed fixes both the data
and the starting weights. The shipped files come from a run on an NVIDIA H200 GPU, and a
rerun on the same kind of GPU repeats their numbers exactly. On a CPU the ratios differ slightly (for seed 42 they range from about
0.012 to 0.028) but stay near 0.02.
For comparison, `--pairing true_pairs` trains each source clip against its own correct target.

## Where the paper uses these results

| Paper | Made by | File |
|---|---|---|
| Synthetic SIF calibration table and figure | `calibrate_sif` | `results/synthetic/sif_calibration.json` |
| Conditional-mean ladder table | `conditional_mean_ladder`, seeds 42, 43, 44 | `results/synthetic/conditional_mean_ladder_seed{42,43,44}.json` |

`python -m paper.make_tables` and `python -m paper.make_figures` rebuild the table and figure
from these files; see the [main README](../README.md).
