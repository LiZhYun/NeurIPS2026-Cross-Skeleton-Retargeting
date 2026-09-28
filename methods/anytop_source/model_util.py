"""Build the source-conditioned AnyTop model and its diffusion process from parsed arguments."""
from core.anytop.utils.model_util import create_gaussian_diffusion, get_gmdm_args


def get_conditioned_args(args):
    base = get_gmdm_args(args)
    base.update({
        'enc_num_queries':  args.enc_num_queries,
        'enc_d_z':          args.enc_d_z,
        'z_drop_prob':      args.z_drop_prob,
        'geom_drop_prob':   args.geom_drop_prob,
        'geom_jitter_prob': args.geom_jitter_prob,
        'topo_drop_prob':   getattr(args, 'topo_drop_prob', 0.15),
        'z_norm_target':    getattr(args, 'z_norm_target', None),
        'no_rest_pe':       getattr(args, 'no_rest_pe', False),
    })
    return base


def create_conditioned_model_and_diffusion(args):
    from methods.anytop_source.model import AnyTopConditioned
    model     = AnyTopConditioned(**get_conditioned_args(args))
    diffusion = create_gaussian_diffusion(args)
    return model, diffusion
