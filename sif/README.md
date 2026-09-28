# sif

The Source-Instance Fidelity (SIF) toolkit. It scores any set of retargeted motions and needs
only NumPy. Install it from the repository root with `pip install -e .`, which also adds the
`sif-score` command.

`distance.py` measures how different two clips on the same body are. `score.py` turns these
distances into SIF and variation for each group of clips and averages them. `stats.py` computes
the 95% interval and the shuffle-test p-value. `cli.py` is the `sif-score` command.

How to use it and how to read its numbers: [docs/sif.md](../docs/sif.md).
