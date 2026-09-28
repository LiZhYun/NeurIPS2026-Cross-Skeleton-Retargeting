"""
Build the conditioning inputs the source-conditioned AnyTop decoder needs for a target skeleton.

The decoder is told which skeleton to move (rest pose, bone offsets, parent list, joint relations,
joint-name embeddings) and, separately, which source motion to follow (the latent z built by the
encoder). This module produces the skeleton half of that pair; the caller attaches z.
"""
import numpy as np

from core.anytop.data_loaders.tensors import truebones_batch_collate
from core.anytop.data_loaders.truebones.data.dataset import create_temporal_mask_for_window


def encode_joints_names(joints_names, t5_conditioner):
    tokens = t5_conditioner.tokenize(joints_names)
    return t5_conditioner(tokens).detach().cpu().numpy()


def build_target_condition(target_object_type, cond_dict, n_frames, temporal_window,
                           t5_conditioner, max_joints, feature_len):
    """Build model_kwargs for the target skeleton, as create_condition in
    core/anytop/sample/generate.py does."""
    obj = cond_dict[target_object_type]
    parents = obj['parents']
    n_joints = len(parents)
    mean     = obj['mean']
    std      = obj['std']
    tpos     = (obj['tpos_first_frame'] - mean) / (std + 1e-6)
    tpos     = np.nan_to_num(tpos)
    joints_names_embs = encode_joints_names(obj['joints_names'], t5_conditioner)

    batch = [
        np.zeros((n_frames, n_joints, feature_len)),                  # motion (ignored, noised)
        n_frames,                                                      # m_length
        parents,
        tpos,
        obj['offsets'],
        create_temporal_mask_for_window(temporal_window, n_frames),
        obj['joints_graph_dist'],
        obj['joint_relations'],
        target_object_type,
        joints_names_embs,
        0,                                                             # crop_start_ind
        mean,
        std,
        max_joints,
    ]
    return truebones_batch_collate([batch])
