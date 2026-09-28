"""Command-line arguments for the source-conditioned AnyTop model.

The base AnyTop option groups are reused as they are; this module adds the options of the
motion encoder and of the losses that train it, and lets a run start from a saved configuration
file (methods/anytop_source/config.json holds the settings the paper used).
"""
from argparse import ArgumentParser

from core.anytop.utils.parser_util import (
    add_base_options, add_data_options, add_model_options, add_training_options,
)
from methods.common.training import apply_config


def add_encoder_options(parser):
    group = parser.add_argument_group('encoder')
    group.add_argument("--enc_num_queries", default=4, type=int,
                       help="Number of spatial query slots K in attention pooling.")
    group.add_argument("--enc_d_z", default=64, type=int,
                       help="Latent dimension D_z for VAE mu/logvar heads.")
    group.add_argument("--no_rest_pe", action='store_true',
                       help="Do not give the encoder the rest-pose bone offsets. "
                            "Makes encoder fully topology/morphology-free.")
    group.add_argument("--z_drop_prob", default=0.1, type=float,
                       help="Probability of replacing z with the null embedding during training, which classifier-free guidance needs.")
    group.add_argument("--geom_drop_prob", default=0.3, type=float,
                       help="Probability of dropping fine geometry (rest-pose) in decoder.")
    group.add_argument("--geom_jitter_prob", default=0.2, type=float,
                       help="Probability of jittering fine geometry instead of dropping.")
    group.add_argument("--topo_drop_prob", default=0.15, type=float,
                       help="Probability of dropping ALL skeleton conditioning (geometry+topology). "
                            "Forces decoder to rely on z.")
    group.add_argument("--cfg_scale", default=1.0, type=float,
                       help="Classifier-free guidance scale at inference (w >= 1.0).")
    group.add_argument("--lambda_inv", default=None, type=float,
                       help="View-consistency loss weight (default: 1.0). Set 0.0 to switch it off.")
    group.add_argument("--beta_max", default=None, type=float,
                       help="KL loss weight after warmup (default: 0.05).")
    group.add_argument("--beta_warmup", default=None, type=int,
                       help="Steps to ramp beta from 0 to beta_max (default: 2000).")
    group.add_argument("--lambda_rank", default=None, type=float,
                       help="Contrastive denoising loss weight (default: 1.0). "
                            "Forces matched z to outperform shuffled z.")
    group.add_argument("--rank_margin", default=None, type=float,
                       help="Margin for contrastive denoising hinge loss (default: 0.05).")
    group.add_argument("--z_norm_target", default=None, type=float,
                       help="Target L2 norm for z_embed before cross-attention. "
                            "Default: sqrt(latent_dim). Equalizes z and null_z magnitude.")


def train_conditioned_args():
    parser = ArgumentParser()
    add_base_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_encoder_options(parser)
    add_training_options(parser)
    parser.add_argument("--config", default=None, type=str,
                        help="JSON file of settings; anything also typed on the command line wins.")
    args = parser.parse_args()
    if args.config:
        apply_config(args, args.config)
    return args
