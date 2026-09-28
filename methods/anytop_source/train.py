"""Train the source-conditioned AnyTop model.

The model is AnyTop plus a motion encoder: a source clip is encoded into a latent z, and the
diffusion decoder is asked to produce a motion for a target skeleton that follows z.

    python -m methods.anytop_source.train --config methods/anytop_source/config.json \\
        --save_dir save/anytop_source
"""
import os
import json
from core.anytop.utils.fixseed import fixseed
from core.anytop.utils import dist_util
from core.anytop.utils.ml_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform, WandBPlatform  # chosen by name in eval() below
from methods.anytop_source.args import train_conditioned_args
from methods.anytop_source.training_loop import TrainLoopConditioned
from methods.anytop_source.get_data import get_dataset_loader_conditioned
from methods.anytop_source.model_util import create_conditioned_model_and_diffusion


def main():
    args = train_conditioned_args()
    fixseed(args.seed)

    save_dir = args.save_dir
    if save_dir is None:
        prefix = args.model_prefix if args.model_prefix is not None else "AnyTopConditioned"
        model_name = f'{prefix}_bs_{args.batch_size}_latentdim_{args.latent_dim}'
        existing = [m for m in os.listdir(os.path.join(os.getcwd(), 'save'))
                    if m.startswith(model_name)]
        if existing and not args.overwrite:
            model_name = f'{model_name}_{len(existing)}'
        save_dir = os.path.join(os.getcwd(), 'save', model_name)
        args.save_dir = save_dir

    ml_platform_type = eval(args.ml_platform_type)
    ml_platform      = ml_platform_type(save_dir=args.save_dir,
                                        project=args.wandb_project,
                                        entity=args.wandb_entity or None)
    ml_platform.report_args(args, name='Args')

    if os.path.exists(save_dir) and not args.overwrite:
        raise FileExistsError(f'save_dir [{save_dir}] already exists.')
    elif not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Resolve None defaults so args.json is fully self-describing
    resolved = vars(args).copy()
    resolved.pop('config', None)
    if resolved.get('lambda_inv') is None:
        resolved['lambda_inv'] = TrainLoopConditioned.LAMBDA_INV
    if resolved.get('beta_max') is None:
        resolved['beta_max'] = TrainLoopConditioned.BETA_MAX
    if resolved.get('beta_warmup') is None:
        resolved['beta_warmup'] = TrainLoopConditioned.BETA_WARMUP
    if resolved.get('lambda_rank') is None:
        resolved['lambda_rank'] = TrainLoopConditioned.LAMBDA_RANK
    if resolved.get('rank_margin') is None:
        resolved['rank_margin'] = TrainLoopConditioned.RANK_MARGIN
    with open(os.path.join(save_dir, 'args.json'), 'w') as fw:
        json.dump(resolved, fw, indent=4, sort_keys=True)

    dist_util.setup_dist(args.device)

    print("creating data loader...")
    data = get_dataset_loader_conditioned(
        batch_size=args.batch_size, num_frames=args.num_frames,
        temporal_window=args.temporal_window, t5_name='t5-base',
        balanced=args.balanced, objects_subset=args.objects_subset)

    print("creating model and diffusion...")
    model, diffusion = create_conditioned_model_and_diffusion(args)

    model.to(dist_util.dev())
    ml_platform.watch_model(model)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'Total params: {n_params:.2f}M')
    print("Training...")
    TrainLoopConditioned(args, ml_platform, model, diffusion, data).run_loop()
    ml_platform.close()


if __name__ == "__main__":
    main()
