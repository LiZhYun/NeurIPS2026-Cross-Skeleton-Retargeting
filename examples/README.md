# examples

`quickstart.py` makes toy source clips and three made-up methods (one that keeps what makes
each clip different, one that returns the same motion for every clip of an action, and one that
returns an unrelated clip of the right action), then prints what SIF reports for each. After
`pip install -e .`, run it from the repository root:

```bash
python examples/quickstart.py
```

It takes a few seconds. A second example, which works with clips saved as files, is in
[docs/sif.md](../docs/sif.md).
