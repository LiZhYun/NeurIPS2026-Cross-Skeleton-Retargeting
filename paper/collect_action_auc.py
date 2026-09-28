"""Collect the action-level AUC of many methods into the single file the paper tables read.

`benchmark.action_auc` scores one method on one query fold and writes
`action_auc_<distance>.json` into the folder given by its `--out_dir`. Write one folder per
method and fold:

    python -m benchmark.action_auc --method_dir <outputs of the method for fold 42> \\
        --fold 42 --method_name AnyTop --distance procrustes \\
        --out_dir <root>/AnyTop/fold_42

Then this script walks `<root>`, and copies for every method, fold and distance the
cluster-tier AUC (an interval with the estimate in the middle), the query counts and the
split by action group.

    python -m paper.collect_action_auc --root <root> --out results/action_auc/fold_auc.json

The method name in the output is the folder name. Pass --names names.json, a
{"folder": "name in the paper"} mapping, when a folder is named differently.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDS = ('42', '43')
DISTANCES = ('procrustes', 'zscore_dtw', 'q_component')
FIELDS = ('cluster_tier_overall_auc_ci', 'cluster_tier_n',
          'cluster_tier_test_test_auc_ci', 'cluster_tier_test_test_n',
          'per_cluster_test_test_cluster_tier')
DESCRIPTION = (
    'Action-level AUC of every method on the two query folds, for three distances. '
    '"cluster_tier" asks whether a target clip of the right action group ranks above clips '
    'of other groups; "test_test" keeps only queries where both skeletons are held out. '
    'Each interval is [lower, point estimate, upper]. '
    '"per_cluster_test_test_cluster_tier" splits the held-out queries by action group.')


def collect(root, names):
    methods = {}
    for folder in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        got = {}
        for fold in FOLDS:
            for distance in DISTANCES:
                f = folder / f'fold_{fold}' / f'action_auc_{distance}.json'
                if f.exists():
                    d = json.loads(f.read_text())
                    got.setdefault(fold, {})[distance] = {k: d[k] for k in FIELDS if k in d}
        if got:
            methods[names.get(folder.name, folder.name)] = got
    return methods


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--root', required=True, help='folder holding one folder per method')
    ap.add_argument('--names', help='JSON file mapping folder name to name in the paper')
    ap.add_argument('--out', default=str(ROOT / 'results' / 'action_auc' / 'fold_auc.json'),
                    help='where to write the collected file')
    a = ap.parse_args()
    names = json.loads(Path(a.names).read_text()) if a.names else {}
    methods = collect(a.root, names)
    Path(a.out).write_text(json.dumps({'description': DESCRIPTION, 'methods': methods},
                                      indent=1) + '\n')
    print(f'wrote {a.out} ({len(methods)} methods)')


if __name__ == '__main__':
    main()
