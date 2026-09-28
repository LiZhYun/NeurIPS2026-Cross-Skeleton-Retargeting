# benchmark

The Truebones evaluation: the sets of retargeting requests every method answers, and the two
ways the paper scores the answers. Source-Instance Fidelity (SIF) asks whether outputs keep
what made each source clip different. The action-level AUC asks whether an output looks like
the right action. The full guide is [docs/benchmark.md](../docs/benchmark.md).

To score a method's outputs with SIF:

```bash
python -m benchmark.score --outputs outputs/<method>/set1891 --set 1891
```

To run the action-level test on one of the two query folds:

```bash
python -m benchmark.action_auc --method_dir outputs/<method>/fold_42 --fold 42 \
    --method_name <method> --distance procrustes
```

What is here:

| File | What it is |
|---|---|
| `sets/` | The three evaluation sets: 1,891 groups (the main evaluation), 49 (the original set, part of the 1,891) and 37 (the part of the 49 every method answered). |
| `queries/` | The two query folds of the action-level test, 300 queries each. |
| `score.py` | SIF scoring of one method's outputs on a set. |
| `action_auc.py` | The action-level test on a query fold. |
| `action_auc_full.py`, `build_queries_full.py` | The same test on every possible query, used in the appendix; the guide gives the commands. |
| `build_set_1891.py`, `build_set_37.py`, `build_queries.py` | Rebuild the sets and folds. |
| `action_taxonomy.py` | Reads the action from a clip's file name and puts it in one of ten action groups. |
| `build_clip_index.py`, `clip_index.json` | The list of clips the evaluation can use, by skeleton and action. |
| `clip_features.py` | Body-independent features of a clip, such as its path and which limbs touch the ground. `python -m benchmark.clip_features --build` measures them for every processed clip and writes `save/clip_features.npz`. |
| `rhythm_ties.json` | The paper's step rate for the 16 clips in which several step rates are exactly equally strong, so the rebuilt clip features match the paper's. |
| `build_contact_groups.py`, `contact_groups.json` | Which joints of each skeleton can touch the ground. |
| `distances.py`, `slot_encoder.py`, `slot_vocabulary.json` | How the action-level test measures the distance between two motions. |
| `action_classifier.py` | Guesses a clip's action group from its features. ANCHOR, random-same-cluster and the AL-Flow models use it. |
