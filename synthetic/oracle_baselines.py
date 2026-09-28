"""What error the simplest possible answers get in the made-up world.

A trained model's error means nothing on its own. This script measures the same error for
five answers that need no training, so the trained number in train_and_evaluate.py can be
placed between them:

  zero                     answer with a motionless figure
  random Gaussian noise    answer with noise
  random clip of the same action   answer with some other clip of that skeleton and action
  source-blind average     answer with the average correct target for that skeleton and
                           action, ignoring which source clip was asked about
  the true answer          answer with the correct target clip, which is error-free

The source-blind average is the important one: it is the best any method can do without
paying attention to which source clip it was given.

    python -m synthetic.oracle_baselines
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

from synthetic.build_dataset import (
    JOINTS_PER_SKEL, ACTIONS, K_SKELETONS, true_transport,
)
from synthetic.train_and_evaluate import split_train_eval

ROOT = Path(__file__).resolve().parents[1]


def evaluate_baseline(predict_fn, eval_pairs, J_max, T_frames):
    """Measure one answer strategy over every question.

    eval_pairs: list of (source clip [T, J, 3], source skeleton, target skeleton, action).
    """
    errors_by_action = {a: [] for a in range(len(ACTIONS))}
    for x_a, sa, sb, ai in eval_pairs:
        # The source clip is cut back to its own joints first. The padding joints are zeros
        # and would distort the interpolation along the chain.
        x_a_real = x_a[:, :JOINTS_PER_SKEL[sa], :]
        x_b_true = true_transport(x_a_real, JOINTS_PER_SKEL[sb], ACTIONS[ai])
        x_b_hat = predict_fn(x_a, sa, sb, ai, J_max, T_frames)
        if x_b_hat.shape[1] != JOINTS_PER_SKEL[sb]:
            x_b_hat = x_b_hat[:, :JOINTS_PER_SKEL[sb], :]
        err = float(np.mean((x_b_hat - x_b_true) ** 2))
        errors_by_action[ai].append(err)
    flat = [e for v in errors_by_action.values() for e in v]
    return {
        'macro_mse_per_action': {ACTIONS[a]: float(np.mean(v)) if v else None
                                 for a, v in errors_by_action.items()},
        'overall_macro_mse': float(np.mean([np.mean(v) for v in errors_by_action.values()
                                            if v])),
        'overall_mean_mse': float(np.mean(flat)),
        'overall_std_mse': float(np.std(flat)),
        'n_tested': len(flat),
    }


def predict_zero(x_a, sa, sb, ai, J_max, T):
    return np.zeros((T, JOINTS_PER_SKEL[sb], 3), dtype=np.float32)


def predict_random_gaussian(x_a, sa, sb, ai, J_max, T):
    return np.random.randn(T, JOINTS_PER_SKEL[sb], 3).astype(np.float32)


class CellMeanPredictor:
    """The average correct target for a skeleton and action, ignoring the source clip."""
    def __init__(self, training_clips, training_meta):
        self.cache = {}
        for sb in range(K_SKELETONS):
            for ai, action in enumerate(ACTIONS):
                src_indices = [i for i, m in enumerate(training_meta)
                               if m['action'] == ai and m['skel'] != sb]
                if not src_indices:
                    continue
                T_b_list = []
                for si in src_indices:
                    x_a = training_clips[si][:, :JOINTS_PER_SKEL[training_meta[si]['skel']], :]
                    x_b = true_transport(x_a, JOINTS_PER_SKEL[sb], action)
                    T_b_list.append(x_b)
                self.cache[(sb, ai)] = np.mean(np.stack(T_b_list), axis=0)

    def __call__(self, x_a, sa, sb, ai, J_max, T):
        return self.cache.get((sb, ai), predict_zero(x_a, sa, sb, ai, J_max, T))


class RandomSameActionPredictor:
    """Some other training clip of the target skeleton performing the same action.

    With one clip per skeleton and action there is only one to choose from, so this answers
    with that clip.
    """
    def __init__(self, training_clips, training_meta, seed=0):
        self.rng = np.random.RandomState(seed)
        self.by_cell = {}
        for i, m in enumerate(training_meta):
            self.by_cell.setdefault((m['skel'], m['action']), []).append(i)
        self.training_clips = training_clips

    def __call__(self, x_a, sa, sb, ai, J_max, T):
        idxs = self.by_cell.get((sb, ai), [])
        if not idxs:
            return predict_zero(x_a, sa, sb, ai, J_max, T)
        i = idxs[self.rng.choice(len(idxs))]
        return self.training_clips[i][:, :JOINTS_PER_SKEL[sb], :]


def predict_oracle_with_xa(x_a, sa, sb, ai, J_max, T):
    """The correct target clip itself, which has no error by construction."""
    x_a_real = x_a[:, :JOINTS_PER_SKEL[sa], :]
    return true_transport(x_a_real, JOINTS_PER_SKEL[sb], ACTIONS[ai])


def build_eval_pairs(meta, clips, n_pairs=200, seed=10):
    """Draw the questions, in the same way train_and_evaluate.py draws them."""
    rng = np.random.RandomState(seed)
    pairs = []
    for _ in range(n_pairs * 2):
        src = rng.choice(len(meta))
        sa = meta[src]['skel']
        ai = meta[src]['action']
        sb_choices = [k for k in range(K_SKELETONS) if k != sa]
        sb = int(sb_choices[rng.choice(len(sb_choices))])
        pairs.append((clips[src], sa, sb, ai))
        if len(pairs) >= n_pairs:
            break
    return pairs


def run_cell(cell_dir: Path, n_pairs=200, seed=0):
    print(f"\n=== Simple answers on: {cell_dir.name} ===")
    clips = np.load(cell_dir / 'clips.npy')
    meta = json.load(open(cell_dir / 'meta.json'))['clips']
    print(f"  {len(meta)} clips, up to {clips.shape[2]} joints, {clips.shape[1]} frames")

    # The same halving of the clips that train_and_evaluate.py uses, so the questions are
    # the same ones.
    train_idx, eval_idx = split_train_eval(meta, eval_frac=0.5, seed=seed)
    if len(eval_idx) == 0:
        eval_idx = list(range(len(meta)))
    eval_meta = [meta[i] for i in eval_idx]
    eval_clips_list = [clips[i] for i in eval_idx]
    train_meta_list = [meta[i] for i in train_idx]
    train_clips_list = [clips[i] for i in train_idx]
    print(f"  train={len(train_idx)}, questions drawn from {len(eval_idx)}")

    eval_pairs = build_eval_pairs(eval_meta, eval_clips_list, n_pairs=n_pairs, seed=seed+10)
    print(f"  {len(eval_pairs)} questions")

    cell_mean = CellMeanPredictor(train_clips_list, train_meta_list)
    random_same = RandomSameActionPredictor(train_clips_list, train_meta_list, seed=seed)

    printed_names = {'zero': 'A motionless figure', 'random_gaussian': 'Random noise',
                     'random_same_action_clip': 'A random clip of the same skeleton and action',
                     'source_blind_cell_mean': 'The source-blind average',
                     'oracle_with_xa': 'The true map'}
    out = {}
    for name, fn in [
        ('zero', predict_zero),
        ('random_gaussian', predict_random_gaussian),
        ('random_same_action_clip', random_same),
        ('source_blind_cell_mean', cell_mean),
        ('oracle_with_xa', predict_oracle_with_xa),
    ]:
        np.random.seed(seed)
        m = evaluate_baseline(fn, eval_pairs, clips.shape[2], clips.shape[1])
        out[name] = m
        print(f"  {printed_names[name]:46s} error {m['overall_macro_mse']:.4f}")

    out_path = cell_dir / 'oracle_baselines.json'
    with open(out_path, 'w') as f:
        json.dump({'cell': cell_dir.name, 'n_train': len(train_idx),
                   'n_eval': len(eval_idx), 'n_pairs': len(eval_pairs),
                   'baselines': out}, f, indent=2)
    print(f"  saved {out_path}")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base_dir', type=str, default='save/synthetic_2x2',
                        help='where build_dataset.py wrote the four versions')
    parser.add_argument('--n_pairs', type=int, default=200)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    base_dir = ROOT / args.base_dir
    cells = ['sparse_unpaired', 'sparse_paired', 'dense_unpaired', 'dense_paired']

    summary = {}
    for c in cells:
        summary[c] = run_cell(base_dir / c, n_pairs=args.n_pairs, seed=args.seed)

    print(f"\n=== Error of each simple answer, averaged over actions ===")
    print(f"{'version':22s} {'motionless':>10s} {'noise':>8s} {'random clip':>12s} "
          f"{'source-blind':>12s} {'true map':>10s}")
    for c in cells:
        b = summary[c]
        print(f"{c:22s} {b['zero']['overall_macro_mse']:10.4f} "
              f"{b['random_gaussian']['overall_macro_mse']:8.4f} "
              f"{b['random_same_action_clip']['overall_macro_mse']:12.4f} "
              f"{b['source_blind_cell_mean']['overall_macro_mse']:12.4f} "
              f"{b['oracle_with_xa']['overall_macro_mse']:10.4f}")

    out_summary = base_dir / 'oracle_baselines_summary.json'
    with open(out_summary, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  saved summary: {out_summary}")


if __name__ == '__main__':
    main()
