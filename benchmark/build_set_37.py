"""Build the part of the smaller evaluation set that every method managed to answer.

The figures in the appendix put all the methods side by side, so they can only use triples
that every method has a result for. A triple is kept when each of the result folders you name
holds a file for every one of that triple's queries. Three of the fourteen methods came up
short on some queries, which brings 49 triples down to 37 and 171 queries down to 130.

The triples and queries that survive keep their original order, their contents and their
numbers, so results produced for the full 49-triple set can be scored against this subset
without being regenerated.

    python -m benchmark.build_set_37 --outputs outputs/anytop/set49 outputs/ace_t/set49 ... \\
        --out benchmark/sets/truebones_37.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def present_query_ids(folder: Path):
    """The numbers of the queries this folder has a result for."""
    return {int(p.name[6:10]) for p in folder.glob('query_*.npy')}


def build(base_set_path, output_dirs):
    base = json.loads(Path(base_set_path).read_text())
    have = [present_query_ids(Path(d)) for d in output_dirs]

    by_triple = {}
    for q in base['queries']:
        by_triple.setdefault(q['triple_id'], []).append(q)

    keep = set()
    for tid, queries in sorted(by_triple.items()):
        qids = [q['query_id'] for q in queries]
        if all(all(qid in ids for qid in qids) for ids in have):
            keep.add(tid)

    triples = [base['triples'][tid] for tid in sorted(keep)]
    queries = [q for q in base['queries'] if q['triple_id'] in keep]

    out = {k: v for k, v in base.items() if k not in ('triples', 'queries')}
    out['n_triples'] = len(triples)
    out['n_queries'] = len(queries)
    out['note'] = (f'Triples of the 49-triple set for which all {len(output_dirs)} method '
                   f'result folders have a file for every query of the triple.')
    out['triples'] = triples
    out['queries'] = queries
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--outputs', nargs='+', required=True,
                        help='one result folder per method, each holding query_0000.npy and so on')
    parser.add_argument('--base_set', default=str(ROOT / 'benchmark/sets/truebones_49.json'))
    parser.add_argument('--out', default=str(ROOT / 'benchmark/sets/truebones_37.json'))
    args = parser.parse_args()

    out = build(args.base_set, args.outputs)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f'wrote {out_path}: {out["n_triples"]} triples / {out["n_queries"]} queries '
          f'(from {len(args.outputs)} method folders)')


if __name__ == '__main__':
    main()
