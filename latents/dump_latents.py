"""Save the internal code each model builds for every query, before it writes a motion.

Every method here works in two stages: it first settles on a short numeric code for the
motion it is about to produce, and only then turns that code into joint positions. The
appendix checks look at the code rather than the motion, because a model that never wrote
down anything about the particular source clip cannot have lost it later in the decoder.

This script runs one trained model over a benchmark set and writes its code for each query.
Nothing is decoded and no motion is written. The output is one file per query plus a small
index:

    <out_dir>/z_query_0000.npy   the code for that query
    <out_dir>/index.json         which query each file belongs to

The other scripts in this folder read that folder: effective_rank.py, cross_seed_alignment.py,
rotation_test.py and latent_sif.py.

Two details differ from how the models are run when they produce motions, and both are here
on purpose, because the appendix compares models against each other rather than against
their own published motions:

  * the starting noise is drawn from NumPy with a seed given on the command line, so that
    two models, or two training runs of one model, start from the same noise;
  * ACE crops the source clip to a whole number of 32-frame windows rather than 4-frame
    steps, which is what the appendix measured.

Example, one model per call:

    python -m latents.dump_latents --method ace \\
        --ckpt save/ace/ace_t/ckpt_final.pt --out_dir outputs/latent_codes/ace_t_seed42/set37
    python -m latents.dump_latents --method alflow --variant labels \\
        --ckpt save/alflow/al_flow/ckpt_final.pt --out_dir outputs/latent_codes/al_flow/set37
    python -m latents.dump_latents --method anytop \\
        --ckpt save/anytop_source/model000175000.pt --out_dir outputs/latent_codes/anytop/set37
"""
from __future__ import annotations
import argparse
import json
import os
import time
from os.path import join as pjoin
from pathlib import Path

import numpy as np
import torch

from core.truebones.param_utils import OBJECT_SUBSETS_DICT, DATASET_DIR
from benchmark.action_classifier import (
    train_classifier, feature_vector, CLUSTER_TO_IDX as CLASSIFIER_CLUSTER_TO_IDX,
)
from benchmark.clip_features import CLIP_FEATURES_PATH, open_clip_features
from methods.tokenizer.registry import TokenizerRegistry
from methods.tokenizer.skel_graph import (
    SkelGraphEncoder, build_skel_features, pad_to_max_joints,
)
from methods.common.motion_descriptors import (
    D_COND_PADDED, phi as phi_np, precompute_skel_descriptors, load_contact_groups,
)
from methods.moreflow.conditions import (
    COND_TYPE_TO_INT, DEFAULT_INFERENCE_CONDS, DEFAULT_CFG_GAMMA, DEFAULT_HEUN_STEPS,
)
from methods.moreflow.flow_transformer import DiscreteFlowTransformer
from methods.moreflow.generate import heun_integrate_ensemble
from methods.ace.generate import load_ace
from methods.alflow.generate import (
    load_checkpoint as load_alflow, encode_source,
    sample_labels, sample_labels_source, sample_labels_source_graph,
    CLUSTER_TO_IDX, EXACT_ACTION_TO_IDX,
)

ROOT = Path(__file__).resolve().parents[1]
SETS = ROOT / 'benchmark' / 'sets'
DATA_ROOT = ROOT / DATASET_DIR
COND_PATH = DATA_ROOT / 'cond.npy'
MOTION_DIR = DATA_ROOT / 'motions'

WINDOW = 32
STRIDE = 4
TOKENS_PER_WINDOW = 8
# ACE's source clip is cropped to whole 32-frame windows here, which is the crop the
# appendix measured; methods/ace/generate.py crops to whole 4-frame steps instead.
ACE_CROP = STRIDE * TOKENS_PER_WINDOW
N_TOKENS = 8
ANYTOP_SECONDS = 5.0


# --------------------------------------------------------------------------------------
# One model at a time. Each of these returns the code for one query as [n_tokens, width].
# --------------------------------------------------------------------------------------

def graph_of(skel, skel_enc, skel_features, max_J, device):
    pj, mask = pad_to_max_joints(skel_features[skel]['per_joint'], max_J)
    agg = skel_features[skel]['agg'].unsqueeze(0).to(device)
    return skel_enc(pj.unsqueeze(0).to(device), mask.unsqueeze(0).to(device), agg)


def encode_source_clip(registry, skel, motion_phys, crop):
    """Crop the source clip to a whole number of windows and encode it with its tokenizer."""
    T_crop = (motion_phys.shape[0] // crop) * crop
    motion_phys = motion_phys[:T_crop]
    motion_t = torch.from_numpy(motion_phys).to(registry.device)
    motion_norm = registry.normalize(skel, motion_t.unsqueeze(0))
    z_src, _ = registry.encode_window(skel, motion_norm.squeeze(0))
    return z_src.squeeze(0), motion_phys, T_crop


def chunk_starts_for(n_tokens):
    """Where the 8-token chunks begin, overlapping by half, with the tail always covered."""
    if n_tokens <= TOKENS_PER_WINDOW:
        return [0]
    stride = TOKENS_PER_WINDOW // 2
    starts = list(range(0, n_tokens - TOKENS_PER_WINDOW + 1, stride))
    last = n_tokens - TOKENS_PER_WINDOW
    if starts[-1] != last:
        starts.append(last)
    return starts


def take_chunk(z_src_full, start, end):
    chunk = z_src_full[start:end]
    if end - start < TOKENS_PER_WINDOW:
        pad = TOKENS_PER_WINDOW - chunk.shape[0]
        chunk = torch.cat([chunk, torch.zeros(pad, chunk.shape[1], device=chunk.device)])
    return chunk.unsqueeze(0)


def extract_ace(query, model, device):
    G, skel_enc, starts, registry, skel_features, max_J, skel_to_id = model
    src_skel, tgt_skel = query['skel_a'], query['skel_b']

    unknown = len(skel_to_id)
    src_id = skel_to_id.get(src_skel, unknown)
    tgt_id = skel_to_id.get(tgt_skel, unknown)
    if src_id == unknown: src_id = 0
    if tgt_id == unknown: tgt_id = 0

    src_graph = graph_of(src_skel, skel_enc, skel_features, max_J, device)
    tgt_graph = graph_of(tgt_skel, skel_enc, skel_features, max_J, device)

    motion_phys = np.load(MOTION_DIR / query['src_fname']).astype(np.float32)
    z_src_full, _, _ = encode_source_clip(registry, src_skel, motion_phys, ACE_CROP)
    n_tok = z_src_full.shape[0]

    width = registry.get(tgt_skel)['model'].codebook_dim
    acc = torch.zeros(n_tok, width, device=device)
    count = torch.zeros(n_tok, device=device)

    src_id_t = torch.tensor([src_id], dtype=torch.long, device=device)
    tgt_id_t = torch.tensor([tgt_id], dtype=torch.long, device=device)
    previous = starts(tgt_id_t)

    for start in chunk_starts_for(n_tok):
        end = min(start + TOKENS_PER_WINDOW, n_tok)
        chunk = take_chunk(z_src_full, start, end)
        predicted = G(chunk, previous, src_id_t, tgt_id_t, src_graph, tgt_graph)
        previous = predicted.detach()
        acc[start:end] += predicted.squeeze(0)[:end - start]
        count[start:end] += 1.0

    return (acc / count.clamp(min=1.0).unsqueeze(-1)).cpu().numpy(), tgt_skel


def extract_moreflow(query, model, device, settings, noise):
    net, skel_enc, registry, skel_features, max_J, skel_to_id, skel_descs = model
    src_skel, tgt_skel = query['skel_a'], query['skel_b']

    unknown = len(skel_to_id)
    src_id = skel_to_id.get(src_skel, unknown)
    tgt_id = skel_to_id.get(tgt_skel, unknown)
    if src_id == unknown: src_id = 0
    if tgt_id == unknown: tgt_id = 0

    src_graph = graph_of(src_skel, skel_enc, skel_features, max_J, device)
    tgt_graph = graph_of(tgt_skel, skel_enc, skel_features, max_J, device)

    motion_phys = np.load(MOTION_DIR / query['src_fname']).astype(np.float32)
    z_src_full, motion_phys, T_crop = encode_source_clip(registry, src_skel, motion_phys, STRIDE)
    n_tok = z_src_full.shape[0]

    width = registry.get(tgt_skel)['model'].codebook_dim
    acc = torch.zeros(n_tok, width, device=device)
    count = torch.zeros(n_tok, device=device)

    # The chunks follow the source token sequence, but MoReFlow starts each one from noise
    # rather than from the source tokens, so the source only enters through the conditions.
    for start in chunk_starts_for(n_tok):
        end = min(start + TOKENS_PER_WINDOW, n_tok)
        z_init = torch.from_numpy(
            noise.randn(1, TOKENS_PER_WINDOW, width).astype(np.float32)).to(device)

        frame_start = start * STRIDE
        frame_end = frame_start + WINDOW
        if frame_end > T_crop:
            frame_end = T_crop
            frame_start = T_crop - WINDOW
        chunk_motion = motion_phys[frame_start:frame_end]

        cond_type_ints, cond_vecs = [], []
        for name in settings['inference_conds']:
            cond_type_ints.append(COND_TYPE_TO_INT[name])
            if name == 'null':
                cond_vecs.append(torch.zeros(1, D_COND_PADDED, device=device))
            else:
                vec = phi_np(chunk_motion, name, skel_descs[src_skel])
                cond_vecs.append(torch.from_numpy(vec).float().unsqueeze(0).to(device))

        predicted = heun_integrate_ensemble(
            net, skel_enc, z_init, src_id, tgt_id, src_graph, tgt_graph,
            cond_type_ints, cond_vecs,
            settings['heun_steps'], settings['cfg_gamma'], device,
        ).squeeze(0)

        acc[start:end] += predicted[:end - start]
        count[start:end] += 1.0

    return (acc / count.clamp(min=1.0).unsqueeze(-1)).cpu().numpy(), tgt_skel


def extract_alflow(query, model, device, settings, noise, cluster_of_clip):
    variant, G, skel_enc, registry, skel_features, max_J, skel_to_id = model
    src_skel, tgt_skel = query['skel_a'], query['skel_b']
    src_fname = query.get('src_fname', '')

    cluster_id = CLUSTER_TO_IDX.get(cluster_of_clip.get(src_fname, ''), 0)
    exact_id = EXACT_ACTION_TO_IDX.get(query.get('src_action', ''), 0)
    cid = torch.full((1,), cluster_id, dtype=torch.long, device=device)
    eid = torch.full((1,), exact_id, dtype=torch.long, device=device)

    width = registry.get(tgt_skel)['model'].codebook_dim
    z_init = torch.from_numpy(
        noise.randn(1, N_TOKENS, width).astype(np.float32)).to(device)

    tgt_graph = graph_of(tgt_skel, skel_enc, skel_features, max_J, device)

    if variant == 'labels':
        tgt_id = torch.full((1,), skel_to_id.get(tgt_skel, 0), dtype=torch.long, device=device)
        z_b = sample_labels(G, z_init, tgt_id, tgt_graph, cid, eid,
                            n_steps=settings['n_euler_steps'], cfg_scale=settings['cfg_scale'])
    else:
        motion_phys = np.load(MOTION_DIR / src_fname).astype(np.float32)
        src_z = encode_source(registry, src_skel, motion_phys)
        if src_z is None or src_z.shape[0] != N_TOKENS:
            raise RuntimeError(f'could not encode the source clip of {src_skel}')
        src_z = src_z.unsqueeze(0)
        src_graph = graph_of(src_skel, skel_enc, skel_features, max_J, device)
        if variant == 'labels_source':
            tgt_id = torch.full((1,), skel_to_id.get(tgt_skel, 0), dtype=torch.long, device=device)
            src_id = torch.full((1,), skel_to_id.get(src_skel, 0), dtype=torch.long, device=device)
            z_b = sample_labels_source(G, z_init, tgt_id, tgt_graph, src_z, src_id, src_graph,
                                       cid, eid, n_steps=settings['n_euler_steps'],
                                       cfg_scale=settings['cfg_scale'])
        else:
            z_b = sample_labels_source_graph(G, z_init, tgt_graph, src_z, src_graph,
                                             cid, eid, n_steps=settings['n_euler_steps'])

    return z_b.squeeze(0).cpu().numpy(), tgt_skel


def source_tensors_for_clip(src_fname, skel, cond_dict, opt, n_frames):
    """Prepare one source clip for AnyTop's encoder: normalize, fit the length, pad the joints."""
    raw = np.load(pjoin(opt.motion_dir, src_fname))              # [T, J_src, 13]
    T, J_src, _ = raw.shape

    mean = cond_dict[skel]['mean']
    std = cond_dict[skel]['std'] + 1e-6
    norm = np.nan_to_num((raw - mean[None, :]) / std[None, :])

    if T >= n_frames:
        norm = norm[:n_frames]
    else:
        norm = np.concatenate([norm, np.zeros((n_frames - T, J_src, 13))], axis=0)

    max_joints = opt.max_joints
    motion = np.zeros((n_frames, max_joints, 13))
    motion[:, :J_src, :] = norm
    offsets = np.zeros((max_joints, 3))
    offsets[:J_src, :] = cond_dict[skel]['offsets']

    motion_t = torch.tensor(motion).permute(1, 2, 0).float().unsqueeze(0)
    offsets_t = torch.tensor(offsets).float().unsqueeze(0)
    mask = torch.zeros(1, max_joints, dtype=torch.bool)
    mask[0, :J_src] = True
    return motion_t, offsets_t, mask


def extract_anytop(query, model, device):
    net, cond_dict, opt, n_frames = model
    motion, offsets, mask = source_tensors_for_clip(
        query['src_fname'], query['skel_a'], cond_dict, opt, n_frames)
    out = net.encoder(motion.to(device), offsets.to(device), mask.to(device))
    z = out[0] if isinstance(out, tuple) else out
    return z.squeeze(0).cpu().numpy(), query['skel_b']


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------

def load_moreflow(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    skels, skel_to_id = ckpt['skels'], ckpt['skel_to_id']
    print(f"  trained on {len(skels)} skeletons")

    net = DiscreteFlowTransformer(n_skels=len(skels)).to(device)
    net.load_state_dict(ckpt['model'])
    net.eval()
    skel_enc = SkelGraphEncoder().to(device)
    skel_enc.load_state_dict(ckpt['skel_enc'])
    skel_enc.eval()

    registry = TokenizerRegistry(list(OBJECT_SUBSETS_DICT['all']), device=device)
    cond_dict = np.load(COND_PATH, allow_pickle=True).item()
    skel_features = build_skel_features(cond_dict)
    max_J = max(skel_features[s]['n_joints'] for s in skels)
    skel_descs = precompute_skel_descriptors(cond_dict, load_contact_groups())
    return net, skel_enc, registry, skel_features, max_J, skel_to_id, skel_descs


def load_anytop(ckpt_path, device_id):
    from core.anytop.utils.fixseed import fixseed
    from core.anytop.utils import dist_util
    from core.anytop.utils.model_util import load_model
    from core.truebones.get_opt import get_opt
    from methods.anytop_source.model_util import create_conditioned_model_and_diffusion

    fixseed(42)
    dist_util.setup_dist(device_id)
    with open(os.path.join(os.path.dirname(str(ckpt_path)), 'args.json')) as f:
        saved = json.load(f)

    class Namespace:
        def __init__(self, d): self.__dict__.update(d)

    opt = get_opt(device_id)
    n_frames = int(ANYTOP_SECONDS * opt.fps)
    cond_dict = np.load(opt.cond_file, allow_pickle=True).item()

    net, _ = create_conditioned_model_and_diffusion(Namespace(saved))
    load_model(net, torch.load(ckpt_path, map_location='cpu'))
    net.to(dist_util.dev())
    net.eval()
    return (net, cond_dict, opt, n_frames), dist_util.dev()


def predicted_clusters(queries, clip_features_path):
    """The action cluster the shared classifier reads off each source clip's motion.

    AL-Flow is told the action the same way it is when it produces motions: the coarse
    cluster is predicted from the clip's measurements, never taken from its filename.
    """
    clip_features = open_clip_features(clip_features_path)
    clf = train_classifier(clip_features, set(OBJECT_SUBSETS_DICT['train']))
    fname_to_idx = {m['fname']: i for i, m in enumerate(clip_features['meta'])}
    idx_to_cluster = {v: k for k, v in CLASSIFIER_CLUSTER_TO_IDX.items()}

    out = {}
    n_unmeasured = 0
    for q in queries:
        fname = q['src_fname']
        if fname in out:
            continue
        if fname in fname_to_idx:
            i = fname_to_idx[fname]
            feat = feature_vector(clip_features['com_path'][i], clip_features['heading_vel'][i],
                                  clip_features['contact_sched'][i], clip_features['cadence'][i],
                                  clip_features['limb_usage'][i])
            out[fname] = idx_to_cluster.get(int(clf.predict(feat.reshape(1, -1))[0]), '')
        else:
            out[fname] = ''
            n_unmeasured += 1
    print(f"  {len(out)} source clips, {n_unmeasured} without measurements")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--method', required=True,
                   choices=['anytop', 'ace', 'moreflow', 'alflow'])
    p.add_argument('--variant', choices=['labels', 'labels_source', 'labels_source_graph'],
                   default='labels', help='which AL-Flow model; ignored by the others')
    p.add_argument('--ckpt', required=True)
    p.add_argument('--set', choices=['37', '49', '1891'], default='37',
                   help='benchmark set, read from benchmark/sets/truebones_<set>.json; the '
                        'appendix uses the 37-triple set')
    p.add_argument('--out_dir', required=True)
    p.add_argument('--max_queries', type=int, default=None)
    p.add_argument('--seed', type=int, default=42,
                   help='seed for the starting noise, so that runs start from the same noise')
    p.add_argument('--device', type=int, default=0, help='which GPU, for AnyTop')
    p.add_argument('--clip_features', default=str(CLIP_FEATURES_PATH),
                   help='the per-clip measurements the action classifier is fitted on')
    p.add_argument('--inference_conds', nargs='+', default=DEFAULT_INFERENCE_CONDS)
    p.add_argument('--cfg_gamma', type=float, default=DEFAULT_CFG_GAMMA)
    p.add_argument('--heun_steps', type=int, default=DEFAULT_HEUN_STEPS)
    p.add_argument('--n_euler_steps', type=int, default=1)
    p.add_argument('--cfg_scale', type=float, default=1.0)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(SETS / f'truebones_{args.set}.json') as f:
        queries = json.load(f)['queries']
    if args.max_queries:
        queries = queries[:args.max_queries]
    print(f"set {args.set}: {len(queries)} queries")

    settings = {'inference_conds': args.inference_conds, 'cfg_gamma': args.cfg_gamma,
                'heun_steps': args.heun_steps, 'n_euler_steps': args.n_euler_steps,
                'cfg_scale': args.cfg_scale}
    noise = np.random.RandomState(args.seed)
    cluster_of_clip = {}

    print(f"Loading {args.ckpt}")
    if args.method == 'anytop':
        model, device = load_anytop(args.ckpt, args.device)
        extract = lambda q: extract_anytop(q, model, device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {device}")
        if args.method == 'ace':
            model = load_ace(args.ckpt, device)
            extract = lambda q: extract_ace(q, model, device)
        elif args.method == 'moreflow':
            model = load_moreflow(args.ckpt, device)
            extract = lambda q: extract_moreflow(q, model, device, settings, noise)
        else:
            G, skel_enc, skels, skel_to_id, _ = load_alflow(args.variant, args.ckpt, device)
            needed = sorted(set([q['skel_a'] for q in queries] + [q['skel_b'] for q in queries]))
            registry = TokenizerRegistry(needed, device=str(device))
            cond_dict = np.load(COND_PATH, allow_pickle=True).item()
            skel_features = build_skel_features(cond_dict)
            if args.variant == 'labels_source_graph':
                max_J = max(skel_features[s]['n_joints'] for s in needed if s in skel_features)
            else:
                max_J = max(skel_features[s]['n_joints'] for s in skels if s in skel_features)
            model = (args.variant, G, skel_enc, registry, skel_features, max_J, skel_to_id)
            print("Fitting the action cluster classifier on the training skeletons...")
            cluster_of_clip = predicted_clusters(queries, args.clip_features)
            extract = lambda q: extract_alflow(q, model, device, settings, noise,
                                               cluster_of_clip)

    index = []
    t0 = time.time()
    with torch.no_grad():
        for i, q in enumerate(queries):
            # Files are named after the benchmark set's own query number, not the position
            # in the set, so that a code can never be read back against another query.
            qid = q['query_id']
            try:
                z, tgt_skel = extract(q)
                np.save(out_dir / f'z_query_{qid:04d}.npy', z)
                index.append({
                    'query_id': qid,
                    'source_skeleton': q.get('skel_a'),
                    'target_skeleton': tgt_skel,
                    'source_clip': q.get('src_fname'),
                    'shape': list(z.shape),
                })
            except Exception as e:
                print(f"  query {qid} ({q.get('skel_a')} to {q.get('skel_b')}): "
                      f"failed, {type(e).__name__}: {e}")
            if (i + 1) % 25 == 0:
                print(f"  [{i+1}/{len(queries)}] {time.time() - t0:.0f}s")

    (out_dir / 'index.json').write_text(json.dumps(index, indent=2))
    print(f"\nSaved {len(index)}/{len(queries)} codes to {out_dir} in {time.time() - t0:.0f}s")


if __name__ == '__main__':
    main()
