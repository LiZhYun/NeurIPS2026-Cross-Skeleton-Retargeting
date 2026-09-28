# synthetic

A small made-up world where the correct retargeting answer is known: eight stick-figure
skeletons, six actions, and a written-down rule that carries a clip from one skeleton to
another. The paper uses it to calibrate Source-Instance Fidelity (SIF), where the true map
scores near one and answers that ignore the source score near zero, and to show that a model
trained on squared error keeps only a small part of the variation that belongs to the source,
however many clips it gets. None of it needs the Truebones Zoo. The full guide is
[docs/synthetic.md](../docs/synthetic.md).

```bash
python -m synthetic.build_dataset
python -m synthetic.calibrate_sif --cell save/synthetic_2x2/dense_paired \
    --out results/synthetic/sif_calibration.json
python -m synthetic.conditional_mean_ladder --seed 42
```

| File | What it does |
|---|---|
| `build_dataset.py` | Builds the four versions of the world (one or fifty clips per skeleton and action, with or without true pairs) in `save/synthetic_2x2/`. |
| `calibrate_sif.py` | Scores the true map, a random clip of the same action and random noise with SIF. |
| `train_and_evaluate.py` | Trains a small generator on each version and measures its error against the true answer. Needs a GPU. |
| `oracle_baselines.py` | The same error for simple answers that need no training, such as the source-blind average. |
| `conditional_mean_ladder.py` | Trains a squared-error model with 2 to 50 clips per combination, each clip trained against the correct target of another clip of the same combination, and measures how much of the true variation its answers keep. |
