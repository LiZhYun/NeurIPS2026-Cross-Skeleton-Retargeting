"""How ANCHOR chooses an existing clip of the target skeleton to answer one query.

ANCHOR makes no new motion. It works out what the source clip is doing, using the classifier
in benchmark/action_classifier.py, and then looks through the clips the target skeleton
already has. Two kinds of clip go on the shortlist: those doing what it thinks the source is
doing, and those whose movement is most similar to the source's, whatever they are labelled.
Each shortlisted clip earns points for

  - doing the action group the classifier predicted,
  - moving similarly to the source,
  - carrying the same action name as the source.

The three can be weighted differently. The clip with the most points is returned unchanged.
Mixing the two kinds of shortlist is what makes the first of the three worth anything: a clip
that moves similarly but is doing something else loses those points.
"""
from __future__ import annotations

import numpy as np

from benchmark.action_classifier import feature_vector, CLUSTERS
from benchmark.clip_features import feature_signature


def build_signature_table(qc):
    """Summarise every stored clip as one vector, so clips can be compared quickly."""
    qsig_table = {}
    for i, m in enumerate(qc['meta']):
        q = {
            'com_path': qc['com_path'][i],
            'heading_vel': qc['heading_vel'][i],
            'contact_sched': qc['contact_sched'][i],
            'cadence': float(qc['cadence'][i]),
            'limb_usage': qc['limb_usage'][i],
        }
        qsig_table[m['fname']] = feature_signature(q)
    return qsig_table


def feature_vector_of_entry(qc_entry):
    return feature_vector(qc_entry['com_path'], qc_entry['heading_vel'],
                          qc_entry['contact_sched'], qc_entry['cadence'],
                          qc_entry['limb_usage'])


def rerank_one_query(q_manifest, clf, clip_index, qsig_table, qc,
                     fname_to_idx, weights, topk_q=10):
    """Score the shortlist for one query and return the clip that wins, with its scores."""
    skel_b = q_manifest['skel_b']
    src_fname = q_manifest['src_fname']
    src_action_raw = q_manifest['src_action']

    # Work out what the source clip is doing
    src_idx = fname_to_idx.get(src_fname)
    if src_idx is None:
        raise RuntimeError(f'no stored features for the source clip {src_fname}')
    src_feat_30d = feature_vector_of_entry({
        'com_path': qc['com_path'][src_idx],
        'heading_vel': qc['heading_vel'][src_idx],
        'contact_sched': qc['contact_sched'][src_idx],
        'cadence': float(qc['cadence'][src_idx]),
        'limb_usage': qc['limb_usage'][src_idx],
    })
    pred_cluster_idx = int(clf.predict(src_feat_30d[None, :])[0])
    pred_cluster = CLUSTERS[pred_cluster_idx]

    # Summarise how the source moves
    src_sig = qsig_table[src_fname]
    src_sig_norm = src_sig / (np.linalg.norm(src_sig) + 1e-9)

    # Everything the target skeleton has
    all_skel_b_clips = []
    for cluster, clips in clip_index['index'][skel_b].items():
        for c in clips:
            all_skel_b_clips.append({**c, 'cluster': cluster})

    pool_by_fname = {}

    # Shortlist, part one: clips doing the predicted action group
    for c in all_skel_b_clips:
        if c['cluster'] == pred_cluster:
            pool_by_fname[c['fname']] = c

    # Shortlist, part two: the clips that move most like the source, whatever they do
    q_ranked = []
    for c in all_skel_b_clips:
        if c['fname'] not in qsig_table:
            continue
        cand_sig = qsig_table[c['fname']]
        cand_norm = cand_sig / (np.linalg.norm(cand_sig) + 1e-9)
        q_sim = float(src_sig_norm @ cand_norm)
        q_ranked.append((q_sim, c))
    q_ranked.sort(key=lambda x: -x[0])
    for q_sim, c in q_ranked[:topk_q]:
        pool_by_fname[c['fname']] = c

    if not pool_by_fname:
        # Nothing made the shortlist, so consider everything
        for c in all_skel_b_clips:
            pool_by_fname[c['fname']] = c

    # Score the shortlist
    cand_records = []
    for fname, cand in pool_by_fname.items():
        if fname not in qsig_table:
            continue
        cand_sig = qsig_table[fname]
        cand_norm = cand_sig / (np.linalg.norm(cand_sig) + 1e-9)
        q_sim = float(src_sig_norm @ cand_norm)
        action_match = 1.0 if cand['action'] == src_action_raw else 0.0
        cluster_match = 1.0 if cand['cluster'] == pred_cluster else 0.0
        composite = (weights['cluster'] * cluster_match
                     + weights['q'] * q_sim
                     + weights['action'] * action_match)
        cand_records.append({
            'fname': fname,
            'action': cand['action'],
            'cluster': cand['cluster'],
            'q_sim': q_sim,
            'action_match': action_match,
            'cluster_match': cluster_match,
            'composite': composite,
        })
    if not cand_records:
        raise RuntimeError(f'No scored candidates for {src_fname} -> {skel_b}')

    cand_records.sort(key=lambda r: -r['composite'])
    best = cand_records[0]
    return {
        'picked_fname': best['fname'],
        'picked_composite': best['composite'],
        'picked_q_sim': best['q_sim'],
        'picked_action': best['action'],
        'picked_cluster': best['cluster'],
        'picked_cluster_match': best['cluster_match'],
        'pred_cluster': pred_cluster,
        'n_candidates': len(cand_records),
        'pool_size_cluster': sum(1 for c in pool_by_fname.values()
                                 if c['cluster'] == pred_cluster),
        'pool_size_q': topk_q,
    }
