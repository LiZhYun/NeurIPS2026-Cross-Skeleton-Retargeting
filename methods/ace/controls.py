"""Ask whether ACE's answer really depends on the source clip.

A retargeting model is supposed to reproduce a particular motion on another animal. If its
answer barely changes when the source clip is taken away or swapped for another, then most
of what it produces comes from the target animal alone, not from the clip it was asked
about. These runs put that to the test, on the trained model the paper scores, and their
answers are scored the same way the ordinary runs are.

methods/ace/train.py has a related setting, --no_source_code, which removes the source
while the model learns. This file removes it at the moment of generation instead, which is
the sharper question: the model was trained with the source and still has to manage
without it.

--source chooses what the model is given:

  real       the query's own source clip, which is ordinary ACE, kept here so the three
             runs can be compared on equal footing
  zero       nothing at all: the model is handed zeros where the source clip would go. The
             starting vector it uses for the first chunk belongs to the target animal and
             never saw the source, so this removes the source completely.
  shuffled   the source clip of a different query. The benchmark groups queries into
             triples that share the same pair of animals and differ only in which clip is
             the source, so swapping within a triple changes the motion being asked for and
             leaves everything else alone. No query keeps its own clip.

--fixed_length also makes every answer the same length, instead of following the source
clip, which separates how long an answer is from what is in it.

    python -m methods.ace.controls --source zero --set 49 \\
        --ckpt save/ace/ace_t/ckpt_final.pt --out_dir outputs/ace_t_source_zero/set49
"""
from __future__ import annotations
import argparse
import collections
from pathlib import Path

import numpy as np
import torch

from methods.ace.generate import load_ace, load_set, run, DEFAULT_LATENT_CACHE


def build_shuffled_sources(queries, seed):
    """Give every query another query's source clip, from within its own triple.

    Returns {query id: the clip it should be given}. No query keeps its own clip, unless a
    triple has only one query, in which case there is nothing to swap with.
    """
    rng = np.random.RandomState(seed)
    by_triple = collections.defaultdict(list)
    for q in queries:
        by_triple[q['triple_id']].append(q)
    shuffled = {}
    n_singletons = 0
    for tid, qs in by_triple.items():
        n = len(qs)
        if n == 1:
            shuffled[qs[0]['query_id']] = qs[0]['src_fname']  # nothing to swap with
            n_singletons += 1
            continue
        while True:
            perm = rng.permutation(n).tolist()
            if all(perm[i] != i for i in range(n)):
                break
        for i, q in enumerate(qs):
            shuffled[q['query_id']] = qs[perm[i]]['src_fname']
    if n_singletons:
        print(f"  Note: {n_singletons} triple(s) hold a single query, which kept its own clip.")
    return shuffled


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--source', choices=['real', 'zero', 'shuffled'], required=True)
    ap.add_argument('--set', choices=['49', '1891'], default='49')
    ap.add_argument('--fixed_length', type=int, default=None,
                    help='make every answer this many frames long, a multiple of four')
    ap.add_argument('--shuffle_seed', type=int, default=42,
                    help='which swap of the source clips to use')
    ap.add_argument('--max_queries', type=int, default=None)
    ap.add_argument('--latent_cache', type=str, default=str(DEFAULT_LATENT_CACHE))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  source={args.source} fixed_length={args.fixed_length}")

    G, skel_enc, starts, registry, skel_features, max_J, skel_to_id = load_ace(
        args.ckpt, device, latent_cache=args.latent_cache)

    queries = load_set(args.set, args.max_queries)
    if args.source == 'shuffled':
        shuffled = build_shuffled_sources(queries, args.shuffle_seed)
        queries = [dict(q, src_fname=shuffled[q['query_id']]) for q in queries]
    print(f"Processing {len(queries)} queries of set {args.set}...")

    run(queries, out_dir, G, skel_enc, starts, registry, skel_features, max_J, skel_to_id,
        device, zero_source=(args.source == 'zero'), fixed_length=args.fixed_length)


if __name__ == '__main__':
    main()
