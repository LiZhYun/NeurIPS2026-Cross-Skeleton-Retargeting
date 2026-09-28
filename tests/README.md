# tests

Tests for the SIF toolkit. They check that the distance ignores position, facing and size,
that outputs equal to their sources score one and identical outputs score zero, and that the
toolkit gives exactly the numbers of the code used for the paper on a stored example
(`data/reference_case.json` and `data/reference_case.npz`). From the repository root:

```bash
pip install -e ".[test]"
python -m pytest tests
```
