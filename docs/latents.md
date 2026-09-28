# The latent-code checks

Most of the trained methods work in two stages. They first settle on a short numeric code for
the motion they are about to produce, called the latent code, and only then turn that code
into joint positions. The paper's appendix looks at these codes directly, for two reasons.

The first is to find where the source clip gets lost. A low Source-Instance Fidelity (SIF) on
the motions can mean that the model never took note of which source clip it was shown, or that
it took note and then lost it on the way out. Running SIF on the codes tells the two apart.

The second is to test the paper's argument about training runs. The theory says that the
training evidence does not fix how one skeleton's latent space lines up with another's, so two
training runs can end up with codes that are turned relative to each other. The checks ask
whether that is what happens, and whether such a turn changes the motion that comes out.

There are four checks:

| Check | Question it answers | Script |
|---|---|---|
| Effective rank | How many directions of its 256-number code does a model actually use? | `latents/effective_rank.py` |
| Cross-seed alignment | Do two training runs that differ only in their seed have codes that are turns of one another? | `latents/cross_seed_alignment.py` |
| Rotation versus noise | Does turning a code change the decoded motion more than noise of the same size does? | `latents/rotation_test.py`, `latents/rotation_significance.py` |
| Latent SIF | Is SIF on the codes high while SIF on the motions is low? | `latents/latent_sif.py` |

All four read folders of codes written by `latents/dump_latents.py`, so the first step is
always to save the codes. The checks use the 37 set, the 130 queries every method
answered (see [benchmark.md](benchmark.md)). All commands are run from the repository root.

## What you need

The processed Truebones Zoo in its default place, `dataset/truebones/zoo/truebones_processed/`
([data.md](data.md)).

The per-animal tokenizers in `save/tokenizers/`, which ACE, MoReFlow and AL-Flow build their
codes on. The rotation test needs them too, because it turns codes back into motion. For ACE
you also need the encoded training clips, `save/latents/cache_all.pt`, which
`methods/moreflow/build_cache.py` writes. [methods.md](methods.md) explains how to make both.

The trained models. None are released, so each has to be trained with the settings in
`methods/` ([methods.md](methods.md)). This table lists every model the appendix uses, the
name the result files give it, and where its training run saves it:

| Name in the results | Settings | Saved model |
|---|---|---|
| `ACE-T-seed42` | `methods/ace/configs/ace_t.json` | `save/ace/ace_t/ckpt_final.pt` |
| `ACE-T-seed43`, `ACE-T-seed44` | the same, with another seed (see below) | `save/ace/ace_t_seed43/ckpt_final.pt`, and so on |
| `ACE-I-seed42` | `methods/ace/configs/ace_i.json` | `save/ace/ace_i/ckpt_final.pt` |
| `ACE-I-seed43`, `ACE-I-seed44` | the same, with another seed | `save/ace/ace_i_seed43/ckpt_final.pt`, and so on |
| `ACE-no-adversarial-loss-seed42` to `-seed44` | `methods/ace/configs/no_adversarial_seed42.json` to `_seed44.json` | `save/ace/ace_no_adversarial_seed42/ckpt_final.pt`, and so on |
| `ACE-no-source-latent-seed42` | `methods/ace/configs/no_source_code.json` | `save/ace/ace_no_source_code/ckpt_final.pt` |
| `MoReFlow-T`, `MoReFlow-I` | `methods/moreflow/configs/moreflow_t.json`, `moreflow_i.json` | `save/moreflow/moreflow_t/ckpt_final.pt`, `save/moreflow/moreflow_i/ckpt_final.pt` |
| `AL-Flow`, `AL-Flow-Src`, `AL-Flow-Src-G` | `methods/alflow/configs/al_flow.json`, `al_flow_src.json`, `al_flow_src_g.json` | `save/alflow/al_flow/ckpt_final.pt`, and so on |
| `AnyTop` | `methods/anytop_source/config.json` | `save/anytop_source/model000175000.pt` |

`ACE-no-adversarial-loss` is ACE trained without its adversarial term, the part of training in
which a second network judges whether an output looks like the target animal really moving.
`ACE-no-source-latent` is ACE trained with the source motion replaced by zeros. `AnyTop` is
AnyTop with our added motion encoder, so that it generates from the source clip, and the paper
uses the model saved after 175,000 training steps.

The settings folder has only the seed-42 run of ACE-T and ACE-I. The other two seeds, which
the cross-seed check needs, are the same settings with the seed and the run name changed:

```bash
python -m methods.ace.train --config methods/ace/configs/ace_t.json --seed 43 --run_name ace_t_seed43
```

Anything typed on the command line wins over the settings file. Do the same with seed 44, and
with `ace_i.json` for ACE-I.

Which models each check needs:

| Check | Models |
|---|---|
| Effective rank | seven: `ACE-no-adversarial-loss-seed42`, `ACE-T-seed42`, `ACE-I-seed42`, `AL-Flow`, `MoReFlow-I`, `MoReFlow-T`, `AnyTop` |
| Cross-seed alignment | nine: the three seeds each of ACE-T, ACE-I and ACE without the adversarial term |
| Rotation versus noise | fifteen: every model in the table except AnyTop, whose code the tokenizers cannot turn back into motion |
| Latent SIF | all sixteen |

We ran every step below on one H200 GPU. Saving the codes of one model takes under a minute
for ACE, AL-Flow and AnyTop and about five minutes for MoReFlow; each check takes seconds
to a few minutes.

## Save the codes

One call per model:

```bash
python -m latents.dump_latents --method ace \
    --ckpt save/ace/ace_t/ckpt_final.pt --out_dir outputs/latent_codes/ace_t_seed42/set37
python -m latents.dump_latents --method moreflow \
    --ckpt save/moreflow/moreflow_t/ckpt_final.pt --out_dir outputs/latent_codes/moreflow_t/set37
python -m latents.dump_latents --method alflow --variant labels \
    --ckpt save/alflow/al_flow/ckpt_final.pt --out_dir outputs/latent_codes/al_flow/set37
python -m latents.dump_latents --method anytop \
    --ckpt save/anytop_source/model000175000.pt --out_dir outputs/latent_codes/anytop/set37
```

For the other two AL-Flow models, use `--variant labels_source` for AL-Flow-Src and
`--variant labels_source_graph` for AL-Flow-Src-G. AnyTop reads the `args.json` its training
run saved next to the model.

Each folder gets one file per query, `z_query_0000.npy` and so on, named by the query's
number in the benchmark set, and an `index.json` that says which query and which skeletons
each file belongs to. Nothing is decoded and no motion is written.

Two details differ from how the models run when they produce motions, and both are on purpose.
The starting noise is drawn from NumPy with the seed given by `--seed` (42 unless you change
it), so that two models, or two training runs of one model, start from the same noise. And ACE
cuts its source clip to a whole number of 32-frame windows, where its motion generator cuts to
4-frame steps; the appendix numbers were measured this way. Saving the codes of the same model
twice gives identical files.

## Effective rank

```bash
python -m latents.effective_rank \
    --run ACE-no-adversarial-loss-seed42 outputs/latent_codes/ace_no_adversarial_seed42/set37 \
    --run ACE-T-seed42 outputs/latent_codes/ace_t_seed42/set37 \
    --run ACE-I-seed42 outputs/latent_codes/ace_i_seed42/set37 \
    --run AL-Flow outputs/latent_codes/al_flow/set37 \
    --run MoReFlow-I outputs/latent_codes/moreflow_i/set37 \
    --run MoReFlow-T outputs/latent_codes/moreflow_t/set37 \
    --run "AnyTop encoder z" outputs/latent_codes/anytop/set37 \
    --out results/latents/effective_rank.json
```

Every code of every query is stacked into one table per model. Effective rank counts the
directions that carry real variation. Spectral flatness is near zero when a handful of
directions carry nearly everything. A model that has squeezed its codes into two or three
directions has little room left to describe a particular source clip. The name after each
`--run` is the label the table prints, and these are the labels the paper uses.

The effective ranks reproduce exactly. Spectral flatness depends on the very smallest
directions, so it can move on a different machine: on one of ours, AnyTop's came out as 0.170
instead of the 0.161 the paper prints, while every effective rank agreed.

## Cross-seed alignment

```bash
python -m latents.cross_seed_alignment --name ACE-T --seeds 42 43 44 \
    --run ACE-T-seed42 outputs/latent_codes/ace_t_seed42/set37 \
    --run ACE-T-seed43 outputs/latent_codes/ace_t_seed43/set37 \
    --run ACE-T-seed44 outputs/latent_codes/ace_t_seed44/set37 \
    --out results/latents/cross_seed_alignment_ace_t.json
```

For every target skeleton the script lines up the codes of each pair of runs query by query
and finds the single rotation that brings one closest to the other. It reports how much of the
gap that rotation closes and how well the codes line up afterwards, averaged over the 21
target skeletons with enough codes. Run it the same way for ACE-I, with `--name ACE-I`, and for
ACE without the adversarial term, with
`--name "the ACE model trained without the adversarial term"`, writing
`cross_seed_alignment_ace_i.json` and `cross_seed_alignment_no_adversarial_loss.json`. The
name only goes into the file's description.

The fractions explained by one rotation and the cosines after it reproduce to five decimal
places. Some fields in these files are numerically unstable and should not be quoted: the
per-skeleton `eig_min`, `rotation_trace_over_d` and `spectral_flatness`, and the per-run
`spectral_flatness_mean` and `spectral_flatness_std`. All of them are effectively zero, and the
paper prints none of them.

## Rotation versus noise

```bash
python -m latents.rotation_test --run ACE-T-seed42 --codes outputs/latent_codes/ace_t_seed42/set37
```

The paper's argument only matters if turning a code changes the motion. A decoder that ignored
part of its code would give the same motion however that part were turned. For each query,
this script turns the saved code in a random direction 20 times, decodes it with the target
skeleton's tokenizer, and measures how far the motion moves. It then adds noise of the same
size 20 times instead. The ratio of the two says whether a turn does anything noise does not
already do; a ratio near one means it does not.

It writes `results/latents/perturbation/<run>.json`, named after `--run`, which replaces the
shipped file of that name; give `--out` to write somewhere else. The turns and the noise are
drawn from a fixed seed, and the summary numbers reproduce to about four decimal places.

Once the fifteen files are there,

```bash
python -m latents.rotation_significance
```

prints whether each model's ratios are convincingly above one, and whether one model's ratios
are above another's, query by query, using a rank test. It reads
`results/latents/perturbation/` by default and writes nothing.

## Latent SIF

```bash
python -m latents.latent_sif \
    --run ACE-T-seed42 outputs/latent_codes/ace_t_seed42/set37 \
    --run MoReFlow-T outputs/latent_codes/moreflow_t/set37
```

For every group of queries that share a source skeleton, a target skeleton and an action, this
compares how far apart the source clips are with how far apart the model's codes are, and
averages the correlation over the 37 groups, just as SIF does on motions. The paper calls it
L-SIF. Codes are averaged over time first, so that clips of different lengths can be compared.

The result goes into `results/latents/latent_sif.json`, beside the SIF of the same run's
motions. Only the runs you name are updated; everything else in the file stays as it was. The
motion score comes from the motions the same run wrote for the benchmark:

```bash
python -m latents.latent_sif --motions ACE-T-seed42 outputs/ace_t/set1891
```

The 37 set keeps the query numbers of the 49 and 1,891 sets, so a folder
of outputs written for either of those works here. Both scores reproduce the shipped file to
seven decimal places or better.

## Where the paper uses these results

| Paper | File |
|---|---|
| Effective-rank table and figure | `results/latents/effective_rank.json` |
| Cross-seed alignment table and figure | `results/latents/cross_seed_alignment_{ace_t,ace_i,no_adversarial_loss}.json` |
| Rotation-versus-noise table and figures | `results/latents/perturbation/<run>.json` |
| SIF and L-SIF table and figure | `results/latents/latent_sif.json` |

`python -m paper.make_tables` and `python -m paper.make_figures` rebuild them from these files;
see the [main README](../README.md).
