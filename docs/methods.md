# Training and generating with each method

The paper compares fourteen methods on the Truebones evaluation. This guide explains, for each
one, what it is, how to train it with the settings the paper used, and how to make it answer
the queries of an evaluation set. It also says roughly how long each step takes and whether it
needs a GPU.

We do not release trained Truebones models, so every learned method has to be trained before
it can generate. Four of the methods take no training at all: the two Motion2Motion variants
and the two random baselines. ANCHOR only fits a small action classifier, which it does by
itself in under a minute each time it runs.

## Before you start

Every command below is run from the repository root. The methods read the processed Truebones
Zoo from `dataset/truebones/zoo/truebones_processed/`; [data.md](data.md) explains how to get
it. Trained models are written under `save/`, and we write generated motions under
`outputs/`. Both folders are ignored by git.

Each method's training settings are kept next to its code, in `methods/<name>/configs/` or
`methods/<name>/config.json`. A training command reads them with `--config`. Anything you also
type on the command line takes precedence over the file. This makes a quick trial easy: adding
`--max_steps 3` to any of the training commands below (or `--num_steps 3` for AnyTop) runs a
few steps and writes a model, which shows that the data and everything else are in place.
Delete what the trial wrote under `save/` before the real run. The tokenizer step skips every
animal that already has a tokenizer, so it would otherwise keep the trial's.

Generating means answering every query of an evaluation set. A query names one source clip,
performed by one animal, and a target animal that should perform the same motion. `--set 1891`
picks the main evaluation, the 1,891 set (1,891 combinations of source animal, target animal
and action, with 6,611 queries), and `--set 49` the smaller 49 set (49 combinations, 171
queries). Each answer is written as one `query_XXXX.npy` file in the folder given by
`--out_dir`. All generators except the two random baselines also take `--max_queries`, to
answer only the first few queries.

Training times below come from the logs of the paper's own runs, each on a single GPU.
Generation times were measured on one NVIDIA H200 GPU, or on eight CPU cores for the methods
that need no GPU.

Once a method has answered a set, it is scored with Source-Instance Fidelity (SIF), for
example:

```bash
python -m benchmark.score --outputs outputs/anchor/set49 --set 49
```

This prints the raw and the length-controlled score, each with its interval, shuffle-test
p-value and variation. How to read them, and how to run the action-level AUC test, is in
[benchmark.md](benchmark.md).

## The shared tokenizer

Four of the methods (ACE, MoReFlow, AL-Flow and DPG-SB-v3) do not work on motion directly.
They work on short sequences of codes produced by a tokenizer: a small network, one per animal,
that turns a 32-frame piece of that animal's motion into eight codes and back again. The
tokenizers are trained once, frozen, and then shared by all four methods. They live in
`methods/tokenizer/`.

The paper trained a tokenizer for each of the 70 animals:

```bash
python -m methods.tokenizer.train --config methods/tokenizer/config.json
```

This needs a GPU. Each animal took about half an hour, so all 70 take about a day and a half
on one GPU. The tokenizers are written to `save/tokenizers/<animal>/`. An animal that already
has a finished tokenizer is skipped, so an interrupted run can simply be started again.

The ten animals held out from some of the models below still get a tokenizer. A tokenizer only
ever sees its own animal's clips, so it tells a model nothing about how motion carries over
from one animal to another.

Next, every training clip is encoded once, so the methods do not have to repeat that work on
every step. Two files are made: one for the 60 training animals and one for all 70.

```bash
python -m methods.moreflow.build_cache --scope train
python -m methods.moreflow.build_cache --scope all
```

Each takes about a minute on a GPU and writes `save/latents/cache_train.pt` or
`save/latents/cache_all.pt`. A rebuilt file will not match ours exactly: for fewer than one
code in three hundred, two candidate codes are so close that different hardware picks the
other one.

## AnyTop

AnyTop (Gat et al.) generates motion for any skeleton. The paper's AnyTop row is AnyTop with a
motion encoder added, so that it generates the target motion from the source clip. The code is
in `methods/anytop_source/`, on top of the original AnyTop code in `core/anytop/`.

```bash
python -m methods.anytop_source.train --config methods/anytop_source/config.json \
    --save_dir save/anytop_source
```

Training needs a GPU; it will not start without one. If `save/anytop_source` already exists,
for example from a trial run, delete it or add `--overwrite`. It trains on all 70 animals. The
first run downloads the T5 text model from Hugging Face, which AnyTop uses to read joint names.
The settings file schedules 600,000 steps and saves the model every 5,000 steps. The paper
evaluates the checkpoint at step 175,000 (`model000175000.pt`), which took about a day to reach.

```bash
python -m methods.anytop_source.generate --set 1891 \
    --ckpt save/anytop_source/model000175000.pt --out_dir outputs/anytop/set1891
```

Generation takes about two seconds per query on a GPU, so about three and a half hours for the
1,891 set. On a CPU it is far slower, close to an hour per query. AnyTop generates from
random noise, and the same seed gives different noise on different hardware, so its outputs
will not match the files the paper scored. Regenerated on a GPU for the smaller set
(`--set 49`), the motions differ slightly from the paper's saved ones, and SIF moves from
-0.033 to -0.053 (raw) and from -0.065 to -0.072 (length-controlled). The conclusion is the
same: AnyTop sits at the source-blind floor.

## ACE-T and ACE-I

ACE is trained like a generative adversarial network: a generator turns the source animal's
codes into the target animal's, and a second network judges whether the result looks like that
animal really moving. ACE-T is trained on all 70 animals. ACE-I is trained on the 60 training
animals, so the other ten are new to it when it is evaluated. The code is in `methods/ace/`.

```bash
python -m methods.ace.train --config methods/ace/configs/ace_t.json
python -m methods.ace.train --config methods/ace/configs/ace_i.json
```

Each run is 50,000 steps and took about an hour on one GPU. The models are written to
`save/ace/ace_t/` and `save/ace/ace_i/`.

```bash
python -m methods.ace.generate --set 1891 \
    --ckpt save/ace/ace_t/ckpt_final.pt --out_dir outputs/ace_t/set1891
python -m methods.ace.generate --set 1891 \
    --ckpt save/ace/ace_i/ckpt_final.pt --out_dir outputs/ace_i/set1891
```

Generation needs a GPU and takes well under a second per query, so under an hour for the main
set. It involves no randomness.

### Taking the source away

The paper asks how much of ACE's output really depends on the source clip. There are two ways
to take the source away. The first is to take it away while the model learns, using the same
settings as ACE-I otherwise. The resulting model generates like any other:

```bash
python -m methods.ace.train --config methods/ace/configs/no_source_code.json
python -m methods.ace.generate --set 49 \
    --ckpt save/ace/ace_no_source_code/ckpt_final.pt --out_dir outputs/ace_no_source_code/set49
```

The second is to take it away from the trained ACE-T model at the moment of generation. With
`--source zero` the model is given nothing where the source clip would be. With
`--source shuffled` each query gets the source clip of another query from the same
combination, so the requested motion changes while the two animals and the action stay the
same.

```bash
python -m methods.ace.controls --source zero --set 49 \
    --ckpt save/ace/ace_t/ckpt_final.pt --out_dir outputs/ace_t_source_zero/set49
python -m methods.ace.controls --source shuffled --set 49 \
    --ckpt save/ace/ace_t/ckpt_final.pt --out_dir outputs/ace_t_source_shuffled/set49
```

Two further training runs remove one of ACE's two training terms. Without the judging network
the paper reports three runs that differ only in their random seed (`no_adversarial_seed42`,
`43` and `44` in the same folder). Without the term that ties the output to the source
(`no_feature_loss`), training fails: the output grows without bound after about ten thousand
steps and the run stops itself. The paper reports that failure, and the settings are kept so it
can be seen again.

```bash
python -m methods.ace.train --config methods/ace/configs/no_adversarial_seed42.json
python -m methods.ace.train --config methods/ace/configs/no_feature_loss.json
```

## MoReFlow-T and MoReFlow-I

MoReFlow learns a path that carries the source animal's codes, step by step, to codes that
read as the target animal's. It is steered by a few measurements taken from the source clip,
such as how fast the body travels and where the feet are. As with ACE, the T model is trained on
all 70 animals and the I model on the 60 training animals. The code is in `methods/moreflow/`,
and it reads the encoded clips made in the tokenizer step above.

```bash
python -m methods.moreflow.train --config methods/moreflow/configs/moreflow_t.json
python -m methods.moreflow.train --config methods/moreflow/configs/moreflow_i.json
```

Each run is 200,000 steps and took about three and a half hours on one GPU.

```bash
python -m methods.moreflow.generate --set 1891 \
    --ckpt save/moreflow/moreflow_t/ckpt_final.pt --out_dir outputs/moreflow_t/set1891
python -m methods.moreflow.generate --set 1891 \
    --ckpt save/moreflow/moreflow_i/ckpt_final.pt --out_dir outputs/moreflow_i/set1891
```

Generation needs a GPU and takes about three seconds per query, so about five and a half hours
for the 1,891 set. It involves no randomness.

## AL-Flow, AL-Flow-Src and AL-Flow-Src-G

The AL-Flow models generate a motion from an action label rather than from a source clip. They
show how far a method can get from knowing the action alone. AL-Flow is told the action and the
target animal. AL-Flow-Src is also given the source clip. AL-Flow-Src-G is given the same, but
knows each animal only by the shape of its skeleton, never by name, so it can also answer for
animals it never trained on. The other two skip such queries. All three are trained on the 60
training animals, and the code is in `methods/alflow/`.

```bash
python -m methods.alflow.train --config methods/alflow/configs/al_flow.json
python -m methods.alflow.train --config methods/alflow/configs/al_flow_src.json
python -m methods.alflow.train --config methods/alflow/configs/al_flow_src_g.json
```

Each run is 50,000 steps and took four to five hours on one GPU. When generating, `--variant`
says which of the three models is being run:

```bash
python -m methods.alflow.generate --variant labels --set 1891 \
    --ckpt save/alflow/al_flow/ckpt_final.pt --out_dir outputs/al_flow/set1891
python -m methods.alflow.generate --variant labels_source --set 1891 \
    --ckpt save/alflow/al_flow_src/ckpt_final.pt --out_dir outputs/al_flow_src/set1891
python -m methods.alflow.generate --variant labels_source_graph --set 1891 \
    --ckpt save/alflow/al_flow_src_g/ckpt_final.pt --out_dir outputs/al_flow_src_g/set1891
```

Generation needs a GPU and takes well under a second per query. It starts from random noise
with a fixed seed.

## DPG-SB-v3

DPG-SB-v3 starts from a real clip instead of from noise: for each query it takes a clip of the
same action that the target animal has already performed, adds noise, and moves it towards what
the source clip asks for. The clips a query will be scored against are never used as a starting
point. It is trained on the 60 training animals, and the code is in `methods/dpg_sb/`.

```bash
python -m methods.dpg_sb.train --config methods/dpg_sb/configs/dpg_sb.json
```

The run is 15,000 steps and took about ten minutes on one GPU.

```bash
python -m methods.dpg_sb.generate --set 1891 \
    --ckpt save/dpg_sb/dpg_sb/final.pt --out_dir outputs/dpg_sb/set1891
```

Generation needs a GPU and takes well under a second per query. The noise it adds is not
seeded, which is how the paper's run was made, so two runs give slightly different outputs.
Add `--seed 42` (or any number) to make a run repeatable.

## Motion2Motion-Direct and Motion2Motion-BVH

Motion2Motion (Chen et al.) copies a motion onto a new animal by example: it is given one clip
of the target animal and a few bones that correspond between the two animals, and builds the
answer by patching together pieces of that example clip. Neither variant is trained. Both run on
the CPU, and the code is in `methods/motion2motion/`.

Motion2Motion-Direct is our implementation of the same algorithm, working directly on the
dataset's motion files:

```bash
python -m methods.motion2motion.generate_direct --set 1891 \
    --out_dir outputs/motion2motion_direct/set1891
```

On eight CPU cores it takes about a sixth of a second per query, so under 20 minutes for the
1,891 set.

Motion2Motion-BVH runs the authors' own program, which reads and writes BVH motion files. Their
code carries no licence, so it is not included here. A script downloads the exact version we
used into `external/motion2motion/`:

```bash
bash methods/motion2motion/fetch_official.sh
python -m methods.motion2motion.generate_bvh --set 1891 \
    --out_dir outputs/motion2motion_bvh/set1891
```

It takes about two and a half seconds per query, so about four and a half hours for the main
set.

Both variants use as many processor threads as the machine reports. On a shared machine that
lets each user run only a couple of cores at once, this can make them very slow: on such a
machine one query took us about four minutes instead of a second or two. There, set
`OMP_NUM_THREADS` to the number of cores you can actually use.

Both variants draw the example clip in the same way the paper did, so they pick the same clips.
The motions they build from it are not exactly repeatable. Motion2Motion-Direct repeats exactly
on one computer but can give a noticeably different motion on another: our rerun differed from
the paper's saved outputs on about 45% of the main-set queries. The authors' program also
starts from random noise without a fixed seed, so Motion2Motion-BVH gives a different output
every time it runs. The scores barely move. Regenerated for this guide, Motion2Motion-Direct's
SIF on the 1,891 set moves from +0.143 to +0.140 (raw) and from +0.129 to +0.127
(length-controlled), and Motion2Motion-BVH's stays at +0.032 (raw) and moves from +0.032 to
+0.029 (length-controlled). The conclusion is the same: Motion2Motion-Direct stays modestly
above the source-blind floor and Motion2Motion-BVH stays near it.

## ANCHOR

ANCHOR makes no new motion. For each query it guesses what the source clip is doing, then
returns an existing clip of the target animal, unchanged: the one that best combines the right
action with movement similar to the source. It shows what can be scored without generating
anything. The code is in `methods/anchor/`.

```bash
python -m methods.anchor.generate --set 1891 --out_dir outputs/anchor/set1891
```

It runs on the CPU. It first fits its action classifier, which takes under a minute, and then
answers each query in a fraction of a second, so the 1,891 set takes about a quarter of an hour.

## random-same-cluster and random-same-exact-action

These two baselines answer each query with a randomly chosen existing clip of the target
animal. random-same-cluster picks from clips in the broad action group the classifier assigns
to the source clip. random-same-exact-action picks from clips carrying the same action name as
the source clip. Neither looks at how the source actually moves, so they show what knowing the
action's name is worth. The code is in `methods/random_label/`.

```bash
python -m methods.random_label.generate --baseline random_same_cluster \
    --set 1891 --out_dir outputs/random_same_cluster/set1891
python -m methods.random_label.generate --baseline random_same_exact_action \
    --set 1891 --out_dir outputs/random_same_exact_action/set1891
```

Both run on the CPU in a few minutes, with a fixed seed. A query is left unanswered when the
target animal has no suitable clip to draw from.

## The action-level test

The action-level AUC uses its own queries, called folds 42 and 43, described in
[benchmark.md](benchmark.md). Every method answers them when given `--fold` in place of
`--set`, in the same way the paper's runs did, for example:

```bash
python -m methods.random_label.generate --baseline random_same_cluster \
    --fold 42 --out_dir outputs/random_same_cluster/fold_42
python -m methods.anytop_source.generate --fold 42 \
    --ckpt save/anytop_source/model000175000.pt --out_dir outputs/anytop/fold_42
python -m methods.alflow.generate --variant labels --fold 42 \
    --ckpt save/alflow/al_flow/ckpt_final.pt --out_dir outputs/al_flow/fold_42
```

The other options stay as they are for the 1,891 set. Each fold has 300 queries, so answering
one takes a small fraction of the time the 1,891 set takes. The notes above about randomness
apply here too: AnyTop's outputs depend on the hardware, and DPG-SB-v3 needs `--seed` to
give the same outputs twice.

## Summary

| Method | Training | Generating the 1,891 set | GPU |
|---|---|---|---|
| Tokenizers (shared) | about a day and a half for all 70 animals, then about a minute per encoded file | none | yes |
| AnyTop | about a day | about 3.5 hours | yes |
| ACE-T, ACE-I | about an hour each | under an hour | yes |
| MoReFlow-T, MoReFlow-I | about 3.5 hours each | about 5.5 hours | yes |
| AL-Flow, AL-Flow-Src, AL-Flow-Src-G | 4 to 5 hours each | under an hour | yes |
| DPG-SB-v3 | about 10 minutes | under an hour | yes |
| Motion2Motion-Direct | none | under an hour | no |
| Motion2Motion-BVH | none | 4 to 5 hours | no |
| ANCHOR | under a minute, each run | about 15 minutes | no |
| random-same-cluster, random-same-exact-action | none | a few minutes | no |
