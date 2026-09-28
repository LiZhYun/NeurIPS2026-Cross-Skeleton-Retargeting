# The Truebones evaluation

The Truebones evaluation asks a retargeting method to move animal motions from one skeleton to
another, and then checks two things. Source-Instance Fidelity (SIF) checks whether the outputs
keep what made each source clip different from other clips of the same action. The
action-level AUC checks whether an output looks more like the right action than a wrong one.
The paper's point is that a method can pass the second check and still sit at the source-blind
floor on the first, so both are reported.

This guide explains the evaluation sets, what a method has to produce, how to score it with
each of the two checks, and how the sets were built. Everything here needs the processed
Truebones Zoo; [data.md](data.md) explains how to get it. All commands are run from the
repository root.

## Queries and groups

A query asks a method for one motion. It names a source clip, such as a spider attacking, and
a target skeleton, such as a triceratops. The method's answer is a motion of the target
skeleton that should carry over the source clip.

A group is a few queries that share a source skeleton, a target skeleton and an action, and
differ only in their source clip. The paper calls a group a triple, after the three things it
holds fixed. SIF is computed within each group: the target and the action are the same for
every output, so the outputs can only differ because their source clips differ. Each group in
the evaluation holds three to five source clips.

## The three evaluation sets

The sets are in `benchmark/sets/`. Each is a JSON file with a list of groups (`triples`) and a
list of queries (`queries`).

| File | Groups | Queries | Used for |
|---|---|---|---|
| `truebones_1891.json` | 1,891 | 6,611 | The main evaluation. |
| `truebones_49.json` | 49 | 171 | The original, smaller evaluation. |
| `truebones_37.json` | 37 | 130 | The part of the 49 groups that every method answered. The appendix figures use it. |

The 1,891 groups are every combination of source skeleton, target skeleton and action where
the source skeleton has at least three clips of the action and the target skeleton has at
least one. They cover every action that can be matched across skeletons, rather than a sample.

The 49 groups are the evaluation the paper started from. They are a fixed part of the 1,891:
queries 0 to 170 of the larger set are exactly the 171 queries of the smaller one, with the
same numbers. So outputs made for the 1,891 set can also be scored on the 49 set, and on the
37 set, without generating anything again.

The 37 groups are the part of the 49 where all fourteen methods produced an output for every
query. Three methods could not answer some queries. Figures that put all the methods side by
side can only use groups every method answered, which leaves 37. Their queries keep their
numbers from the 49 set.

Each query in these files looks like this:

```json
{"query_id": 0, "triple_id": 0, "skel_a": "SpiderG", "skel_b": "Tricera",
 "src_fname": "SpiderG___Attack5_933.npy", "src_action": "attack", "cluster": "combat", ...}
```

`skel_a` is the source skeleton, `skel_b` the target skeleton, and `src_fname` the source clip,
a file in the processed dataset's `motions/` folder. `src_action` is the action read from the
clip's file name, and `cluster` is the broad action group it belongs to, such as locomotion or
combat. `triple_id` says which group the query belongs to.

## What a method has to write

A method writes one file per query into a folder of its own:

```
outputs/<method>/set1891/query_0000.npy
outputs/<method>/set1891/query_0001.npy
...
```

The number in the file name is the query's `query_id`, written with four digits. Each file
holds the motion produced for that query on the target skeleton, as a NumPy array of shape
`(frames, joints, 3)` with the position of every joint in every frame. Arrays in the
processed dataset's own format, `(frames, joints, 13)`, are also accepted; only the first
three values of each joint are used, which are its position relative to the root. The joints
are those of the target skeleton, in the order the processed dataset lists them. An output may
have any number of frames.

Every method in `methods/` writes this format when given `--set` and `--out_dir`, for example:

```bash
python -m methods.anchor.generate --set 1891 --out_dir outputs/anchor/set1891
```

[methods.md](methods.md) explains how to run each method.

## Scoring with SIF

```bash
python -m benchmark.score --outputs outputs/<method>/set1891 --set 1891 --out <method>_sif.json
```

`--set` picks the evaluation set: `1891` (the default), `49` or `37`. It prints one line for
raw scoring and one for length-controlled scoring, and `--out` writes the same numbers to a
JSON file. For example, AnyTop's outputs on the 1,891 set give the numbers the paper reports:

```
              raw: SIF -0.048  95% CI [-0.124, +0.026]  p = 0.9153  variation 1.503  (1891 groups)
length-controlled: SIF -0.061  95% CI [-0.137, +0.014]  p = 0.9541  variation 1.624  (1891 groups)
```

Each line holds the average SIF over the groups, a 95% interval, a shuffle-test p-value, the
variation, and the number of groups scored. SIF is near one when outputs differ the way their
source clips do and near zero when they ignore them. The p-value asks how often the outputs
would score this high if they were paired with their source clips at random. Variation says
how much the outputs differ from one another compared with their sources, so that a method
returning the same motion every time is easy to spot. In raw scoring each clip keeps its own
length; in length-controlled scoring every clip is first stretched to 64 frames, so that clip
length carries no information. [sif.md](sif.md) explains each of these numbers.

The scoring builds the groups for you. It reads the source clips from the processed dataset
(use `--motions` if they are somewhere else) and uses the source skeleton as the block, the
unit that is resampled as a whole for the interval and shuffled together for the p-value. A
source clip is retargeted to many target skeletons, so the groups of one source skeleton are
not independent of each other. Use `--length raw` or `--length fixed` to get only one of the
two scores.

A query without an output file is left out of its group. A group needs at least two outputs to
be scored; with only two there is a single distance, and the group scores zero. The last
number on each line shows how many groups were scored; check that it matches the size of the
set.

The 1,891 set took us about seven minutes to score, so on a shared machine it is best run as a
batch job. The 49 and 37 sets take a few seconds.

## The action-level test

The action-level AUC asks whether a method's output looks like the right action. For each
query, the output is compared with real clips of the target skeleton, and those clips are put
in order from closest to furthest. The test checks whether clips of the right action come
first. The score for one query is the chance that a randomly picked clip of the right action
ranks above a randomly picked clip of a wrong one. This is the area under the ROC curve (AUC).
One means the right action always comes first, and one half means the output says nothing
about the action.

This test uses its own queries, which ask for one output per source clip rather than groups of
outputs. They are in `benchmark/queries/`, in two sets called folds 42 and 43 after the random
seeds that drew them, so that a result can be checked on a second draw. Each fold has 300
queries: 100 where both skeletons were used in training, 100 where one was, and 100 where
neither was. Those last queries show how a method does on animals it has never seen.

A method writes its answers to the fold queries in the same way as above, one
`query_XXXX.npy` per query, into a folder per fold. Every method in `methods/` does this when
given `--fold 42` or `--fold 43` in place of `--set`:

```bash
python -m methods.anchor.generate --fold 42 --out_dir outputs/anchor/fold_42
python -m methods.ace.generate --fold 42 \
    --ckpt save/ace/ace_t/ckpt_final.pt --out_dir outputs/ace_t/fold_42
```

Then:

```bash
python -m benchmark.action_auc --method_dir outputs/<method>/fold_42 --fold 42 \
    --method_name <method> --distance procrustes
```

This takes about a minute and writes `action_auc_procrustes.json` next to the answers, or into
`--out_dir` if given. The data is read from the default location described in
[data.md](data.md). The test also reads the clip features, which you make once with
`python -m benchmark.clip_features --build` (see [data.md](data.md)). The printed summary for AnyTop on fold 42 begins:

```
=== AnyTop fold 42 procrustes ===
  cluster-tier overall:   0.454 [0.412, 0.499] (n=300)
  cluster-tier held out:  0.361 [0.288, 0.445] (n=100)
  exact-tier   overall:   0.483 [0.368, 0.609] (n=49)
  exact-tier   held out:  0.468 [0.156, 0.797] (n=9)
```

The test is run at two levels. The cluster tier asks whether clips of the right broad action
group rank above clips of other groups. The exact tier asks whether clips of the very same
action rank above clips of the same group but a different action; only queries with such
clips on the target skeleton take part. "Held out" keeps only the queries where neither
skeleton was used in training. The intervals resample whole pairs of source and target
skeletons, because queries that share a pair are not independent. The result file also holds
the score of every query and, for the held-out queries, the score of each action group.
A query without an answer file is left out; the n in each line shows how many were scored
(random-same-cluster answers 286 of the 300 fold-42 queries).

`--distance` sets how the closeness of two motions is measured. `procrustes`, the measure the
paper reports, lines the two motions up in time and then turns and resizes one pose to fit the
other, and measures what is left. The appendix also uses `zscore_dtw`, which compares the
shape of the movement over time while ignoring its size and place, and `q_component`, which
compares what the movement achieves: the path of the body, which limbs touch the ground, the
step rate and how the movement is shared among the limbs. These two need outputs in the
processed dataset's 13-value format; answers that hold only positions are skipped under them.

## The action-level test on every possible query

The appendix repeats the action-level test for ANCHOR on every possible query rather than on
the 300 of a fold, to show that its score does not depend on which queries were drawn. The
full list pairs every clip with every other skeleton that has a clip of the same action group:
30,497 queries. It is about seventy megabytes, so it is not included; build it first:

```bash
python -m benchmark.build_queries_full --out_dir queries_full
python -m methods.anchor.generate --queries queries_full/manifest.json --out_dir outputs/anchor/full
python -m benchmark.action_auc_full --method_dir outputs/anchor/full --method_name ANCHOR \
    --distance procrustes --queries queries_full/manifest.json
```

The list takes a few seconds to build and holds the same queries as the one the paper used.
Because the query numbers run past ten thousand, the answers are named with five digits,
`query_00000.npy` and so on. Answering all 30,497 queries took us about 25 minutes of
computing and scoring them about an hour and a half, so both are best run as batch jobs. The
scoring writes `action_auc_full_procrustes.json` next to the answers, with the result for all
queries and separately for queries where both, one or neither of the skeletons was used in
training. Its cluster-tier numbers are the ones in `results/action_auc/enumeration.json`.

## Rebuilding the sets

The files in `benchmark/sets/` and `benchmark/queries/` are the ones the paper used, and
you do not need to rebuild them. The builders are included so that you can see exactly how
each was made and check it.

The 1,891 set is rebuilt from the processed dataset and the 49 set:

```bash
python -m benchmark.build_set_1891 --out truebones_1891.json
```

The 49 set is copied in first with its query numbers, and every remaining group follows in a
fixed order. The result is identical to `benchmark/sets/truebones_1891.json`. Adding
`--min_src_clips 2` builds the wider version the appendix uses, where a source skeleton needs
only two clips of an action.

The 37 set is rebuilt from the output folders of all fourteen methods on the 49 set, since it
is defined as the groups where each of them answered every query:

```bash
python -m benchmark.build_set_37 --out truebones_37.json --outputs \
    outputs/anytop/set49 outputs/ace_t/set49 outputs/ace_i/set49 \
    outputs/moreflow_t/set49 outputs/moreflow_i/set49 outputs/al_flow/set49 \
    outputs/al_flow_src/set49 outputs/al_flow_src_g/set49 outputs/dpg_sb/set49 \
    outputs/motion2motion_bvh/set49 outputs/motion2motion_direct/set49 outputs/anchor/set49 \
    outputs/random_same_cluster/set49 outputs/random_same_exact_action/set49
```

Folders made for the 1,891 set work here too, since they include the 171 queries of the 49
set. Given the outputs the paper used, the result is identical to
`benchmark/sets/truebones_37.json`. With other outputs, a different set of methods may fall
short on different queries, and the result can differ.

The 49 set cannot be rebuilt exactly. It was drawn at random from the clips in the order the
file system listed them on the machine where it was made, and that order is not recorded. Use
the file in `benchmark/sets/` as it is.

The two query folds of the action-level test are rebuilt from the clip list in
`benchmark/clip_index.json`:

```bash
python -m benchmark.build_queries --folds 42 43 --out_root queries
```

The result is identical to `benchmark/queries/`.
