"""Answer a whole set of queries by running the Motion2Motion authors' own program.

Their program reads and writes motion files in the BVH format, so every query goes through it
unchanged. The source clip and the example clip are taken from the BVH copies of the dataset,
the corresponding bones are written to a file the program understands, the program is run, and
its answer is read back as the position of every joint in every frame. Answers are therefore
stored as positions rather than in the dataset's own thirteen-number form; the scoring code
recognises the difference and handles both.

Their program is not included here and carries no licence, so fetch it first:

    bash methods/motion2motion/fetch_official.sh

then

    python -m methods.motion2motion.generate_bvh --set 49 --out_dir outputs/motion2motion_bvh/set49

The example clip is drawn exactly as in methods/motion2motion/generate_direct.py.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import shutil
import subprocess
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

from core.truebones.param_utils import DATASET_DIR
from methods.common.queries import (
    REFERENCE_KEYS, per_query_seed, reference_length, stretch_to_length,
)
from methods.motion2motion.core import author_sparse_mapping, load_assets

ROOT = Path(__file__).resolve().parents[2]
BVH_DIR = Path(DATASET_DIR) / 'bvhs'
DEFAULT_OFFICIAL_DIR = ROOT / 'external/motion2motion'

def list_target_skel_bvhs(skel_name):
    """Every clip of one skeleton, in the format the authors' program reads."""
    return sorted([f for f in os.listdir(BVH_DIR)
                   if (f.startswith(skel_name + '___') or f.startswith(skel_name + '_'))
                   and f.endswith('.bvh')])


def load_bvh_positions(bvh_path):
    """Read a motion file and work out where each joint ends up in each frame."""
    import BVH
    from core.truebones.motion_process import positions_global
    anim, names, _ = BVH.load(str(bvh_path))
    pos = positions_global(anim)  # [T, J, 3]
    return pos


def get_motion2motion_visible_names(bvh_path):
    """The joint names the authors' program can see in a motion file, in the order it reads
    them. It ignores the leaf markers at the ends of limbs, so those are skipped here too."""
    import re
    visible = []
    in_end_site = False
    with open(bvh_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith('End Site'):
                in_end_site = True
                continue
            if in_end_site:
                # skip to the end of the marker
                if stripped == '}':
                    in_end_site = False
                continue
            m = re.match(r'^(JOINT|ROOT)\s+([A-Za-z0-9_\-:]+)', stripped)
            if m:
                visible.append(m.group(2))
            if stripped.startswith('MOTION'):
                break  # done with hierarchy
    return visible


# Per-skeleton normalizer cache: skel -> (cond_to_visible: dict, visible_names: list)
_NORMALIZER_CACHE = {}


def get_normalizer(skel_name, cond):
    """For one skeleton, map each of its joints to a joint name the authors' program can see.

    Their program does not see every joint, so a joint it cannot see is represented by the
    nearest joint above it that it can.
    """
    if skel_name in _NORMALIZER_CACHE:
        return _NORMALIZER_CACHE[skel_name]
    bvhs = list_target_skel_bvhs(skel_name)
    if not bvhs:
        raise RuntimeError(f'No BVH for {skel_name}')
    rep_bvh = BVH_DIR / bvhs[0]
    visible = get_motion2motion_visible_names(rep_bvh)
    visible_set = set(visible)
    parents = cond[skel_name]['parents']
    names = list(cond[skel_name]['joints_names'])
    cond_to_visible = {}
    for i, name in enumerate(names):
        # Climb towards the root until we reach a joint the program can see
        cur = i
        steps = 0
        while steps < 50:  # never loop forever on a malformed skeleton
            cur_name = names[cur]
            if cur_name in visible_set:
                cond_to_visible[i] = cur_name
                break
            p = parents[cur]
            if p < 0 or p == cur:
                # Reached the root without finding one, so use the root
                cond_to_visible[i] = visible[0] if visible else cur_name
                break
            cur = p
            steps += 1
    _NORMALIZER_CACHE[skel_name] = (cond_to_visible, visible)
    return cond_to_visible, visible


def preprocess_bvh_with_suffix(src_bvh, dst_bvh):
    """Write a copy of a motion file with three extra characters on every joint name.

    The authors' program drops the last five characters of a joint name before comparing it
    with the correspondence file, so the names only line up if the extra characters are there.
    """
    import re, random, string
    with open(src_bvh, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    rng = random.Random(42)  # the same file always gets the same characters
    joint_pat = re.compile(r"^(\s*)(JOINT|ROOT)\s+([A-Za-z0-9_\-:]+)(\s*)$")
    out_lines = []
    for line in lines:
        m = joint_pat.match(line)
        if m:
            spaces, jtype, jname, trail = m.groups()
            suffix = ''.join(rng.choices(string.ascii_letters + string.digits, k=3))
            out_lines.append(f"{spaces}{jtype} {jname}__{suffix}{trail}")
        else:
            out_lines.append(line)
    with open(dst_bvh, 'w', encoding='utf-8') as f:
        f.writelines(out_lines)


def auto_generate_mapping(skel_a, skel_b, cond, contact_groups, max_pairs=8):
    """Write the file of corresponding bones that the authors' program expects.

    The pairs are chosen the same way as in the direct variant, then each joint is replaced by
    the nearest one above it that their program can see.
    """
    pairs_ij, _ = author_sparse_mapping(skel_a, skel_b, cond, contact_groups,
                                        max_pairs=max_pairs)

    src_to_vis, src_visible = get_normalizer(skel_a, cond)
    tgt_to_vis, tgt_visible = get_normalizer(skel_b, cond)

    # Always pair the roots: their program assumes the first joint is the root
    root_pair = (src_visible[0], tgt_visible[0])

    mapping_set = set()
    mapping_set.add(root_pair)
    for src_idx, tgt_idx in pairs_ij:
        s_name = src_to_vis.get(src_idx)
        t_name = tgt_to_vis.get(tgt_idx)
        if s_name is None or t_name is None:
            continue
        mapping_set.add((s_name, t_name))

    mapping = [{'source': s, 'target': t} for (s, t) in sorted(mapping_set)]

    return {
        'source_name': skel_a,
        'target_name': skel_b,
        'root_joint': src_visible[0],
        'mapping': mapping,
    }


def run_official(official_dir, src_bvh, ex_bvh, mapping_json, output_dir, device='cpu',
                 timeout_sec=300):
    """Run the authors' program once, on one source clip and one example clip.

    Returns the file it wrote, or an explanation of why there is none.
    """
    cmd = [
        sys.executable,
        '-u', 'run_M2M.py',
        '-e', str(ex_bvh),  # example/target BVH
        '-d', device,
        '--source', str(src_bvh),
        '--mapping_file', str(mapping_json),
        '--output_dir', str(output_dir),
        '--sparse_retargeting',
        '--matching_alpha', '0.9',
    ]
    try:
        result = subprocess.run(cmd, cwd=str(official_dir),
                                capture_output=True, text=True, timeout=timeout_sec)
        if result.returncode != 0:
            err_summary = ((result.stderr[-800:] if result.stderr else '')
                           + ' || STDOUT: ' + (result.stdout[-400:] if result.stdout else ''))
            return None, err_summary
        # Their program decides where to put its answer, so search for it
        out_bvhs = []
        for root_dir, _, files in os.walk(output_dir):
            for f in files:
                if f.endswith('_syn.bvh'):
                    out_bvhs.append(Path(root_dir) / f)
        if not out_bvhs:
            return None, ('No output BVH (rc=0, stdout: '
                          + (result.stdout[-400:] if result.stdout else 'empty') + ')')
        out_bvhs.sort(key=lambda p: p.stat().st_mtime)
        return out_bvhs[-1], None
    except subprocess.TimeoutExpired:
        return None, f'Timeout after {timeout_sec}s'


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--set', choices=['37', '49', '1891'],
                       help='which evaluation set to answer, by its number of triples')
    group.add_argument('--fold', type=int, choices=[42, 43],
                       help='answer one of the two query sets in benchmark/queries instead')
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--official_dir', default=str(DEFAULT_OFFICIAL_DIR),
                        help='where the Motion2Motion authors\' program was downloaded to')
    parser.add_argument('--max_queries', type=int, default=10000)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--timeout', type=int, default=300)
    args = parser.parse_args()

    if args.set:
        queries_path = ROOT / f'benchmark/sets/truebones_{args.set}.json'
        seed_tag = 999   # the value the paper used when answering the evaluation sets
    else:
        queries_path = ROOT / f'benchmark/queries/fold_{args.fold}/manifest.json'
        seed_tag = args.fold

    official_dir = Path(args.official_dir)
    if not (official_dir / 'run_M2M.py').exists():
        raise SystemExit(f'{official_dir}/run_M2M.py not found. '
                         f'Run methods/motion2motion/fetch_official.sh first.')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Queries: {queries_path}")
    print(f"Output dir: {out_dir}")
    print(f"Device: {args.device}, timeout: {args.timeout}s")

    cond, contact_groups, _ = load_assets()

    with open(queries_path) as f:
        manifest = json.load(f)
    queries = manifest['queries'][:args.max_queries]
    print(f"Running {len(queries)} queries")

    work_root = Path(tempfile.mkdtemp(prefix='motion2motion_'))
    print(f"Working dir: {work_root}")

    tgt_bvh_cache = {}
    per_query = []
    t_total = time.time()

    for i, q in enumerate(queries):
        qid = q['query_id']
        skel_a = q['skel_a']
        skel_b = q['skel_b']
        src_fname = q['src_fname']

        rec = {'query_id': qid, 'cluster': q.get('cluster'), 'split': q.get('split'),
               'skel_a': skel_a, 'skel_b': skel_b, 'status': 'pending'}

        try:
            # 1. The clip to copy
            src_bvh_name = src_fname.replace('.npy', '.bvh')
            src_bvh = BVH_DIR / src_bvh_name
            if not src_bvh.exists():
                rec['status'] = 'skipped_no_src_bvh'
                per_query.append(rec)
                continue

            # 2. The example clip, never one the answer will be compared against
            if skel_b not in tgt_bvh_cache:
                tgt_bvh_cache[skel_b] = list_target_skel_bvhs(skel_b)
            full_pool = tgt_bvh_cache[skel_b]
            forbidden = set()
            for key in REFERENCE_KEYS:
                for x in q.get(key, []):
                    forbidden.add(x['fname'].replace('.npy', '.bvh'))
            pool = [f for f in full_pool if f not in forbidden]
            if not pool:
                pool = full_pool
                rec['example_fallback'] = 'all_clips_are_references'
            qseed = per_query_seed(seed_tag, qid, skel_b)
            qrng = np.random.RandomState(qseed)
            ex_bvh_name = pool[qrng.randint(0, len(pool))]
            ex_bvh = BVH_DIR / ex_bvh_name

            # 3. Which bones correspond
            if skel_a not in contact_groups or skel_b not in contact_groups:
                rec['status'] = 'skipped_no_contact_groups'
                per_query.append(rec)
                continue
            mapping = auto_generate_mapping(skel_a, skel_b, cond, contact_groups)
            # Fewer than three pairs leaves the pose too loosely determined to be useful
            if not mapping['mapping'] or len(mapping['mapping']) < 3:
                rec['status'] = 'skipped_few_pairs'
                rec['n_pairs'] = len(mapping['mapping']) if mapping else 0
                per_query.append(rec)
                continue

            q_work = work_root / f'q_{qid:04d}'
            q_work.mkdir(exist_ok=True)
            src_bvh_pp = q_work / src_bvh.name
            ex_bvh_pp = q_work / ex_bvh.name
            preprocess_bvh_with_suffix(src_bvh, src_bvh_pp)
            preprocess_bvh_with_suffix(ex_bvh, ex_bvh_pp)
            mapping_json = q_work / 'mapping.json'
            with open(mapping_json, 'w') as f:
                json.dump(mapping, f, indent=2)
            motion2motion_out_dir = q_work / 'output'
            motion2motion_out_dir.mkdir(exist_ok=True)

            # 4. Run their program
            t0 = time.time()
            output_bvh, err = run_official(
                official_dir, src_bvh_pp, ex_bvh_pp, mapping_json, motion2motion_out_dir,
                device=args.device, timeout_sec=args.timeout)
            runtime = time.time() - t0
            if output_bvh is None:
                rec['status'] = 'official_failed'
                rec['error'] = err
                per_query.append(rec)
                continue

            # 5. Read its answer back as joint positions
            positions = load_bvh_positions(output_bvh)  # [T_out, J_out, 3]

            # 6. Stretch to the typical length of the clips this will be compared against
            positions = stretch_to_length(positions, reference_length(q, positions.shape[0]))

            # 7. Save the positions; the scoring code recognises this form
            np.save(out_dir / f'query_{qid:04d}.npy', positions.astype(np.float32))
            rec['status'] = 'ok'
            rec['example_bvh'] = ex_bvh_name
            rec['n_pairs'] = len(mapping['mapping'])
            rec['runtime_sec'] = runtime
            rec['T_out'] = int(positions.shape[0])
            rec['J_out'] = int(positions.shape[1])

            shutil.rmtree(q_work, ignore_errors=True)

            if (i + 1) % 5 == 0 or i == 0:
                elapsed = time.time() - t_total
                eta = elapsed / (i + 1) * (len(queries) - i - 1)
                print(f"  [{i+1}/{len(queries)}] {skel_a}->{skel_b} "
                      f"T={rec['T_out']} J={rec['J_out']} ({runtime:.1f}s, ETA {eta:.0f}s)")

        except Exception as e:
            rec['status'] = 'failed'
            rec['error'] = str(e) + '\n' + traceback.format_exc(limit=3)
            print(f"  FAILED query {qid}: {e}")

        per_query.append(rec)

    shutil.rmtree(work_root, ignore_errors=True)

    total_time = time.time() - t_total
    n_ok = sum(1 for r in per_query if r['status'] == 'ok')
    n_failed = sum(1 for r in per_query if r['status'] in ('failed', 'official_failed'))
    n_skipped = sum(1 for r in per_query if 'skipped' in r['status'])
    print(f"\nTotal: {total_time:.0f}s, OK: {n_ok}/{len(per_query)}, "
          f"failed: {n_failed}, skipped: {n_skipped}")

    summary = {
        'method': 'Motion2Motion-BVH',
        'queries': str(queries_path),
        'n_queries': len(per_query),
        'n_ok': n_ok,
        'n_failed': n_failed,
        'n_skipped': n_skipped,
        'total_time_sec': total_time,
        'output_format': 'positions_T_J_3',
        'per_query': per_query,
    }
    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved: {out_dir}/metrics.json")


if __name__ == '__main__':
    main()
