"""
Train a diffusion model on images.
"""

import argparse
import numpy as np

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data_bdd100k
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist(args.gpus)
    logger.configure(dir=args.output_path)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        num_classes=None,
        multiclass=False,
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    if args.init_step != -1:
        # when sampling t with the scheduler, there will be
        # a probability of 0 to extract a t > args.init_step
        schedule_sampler._weights[args.init_step:] = 0

    logger.log("creating data loader...")
    data = load_data_bdd100k(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size='256,512',
        class_cond=args.class_cond,
        random_crop=False,
        random_flip=False,
        deterministic=True
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="/data/chercheurs/jeanner211/DATASETS/celeba",
        schedule_sampler="uniform",
        init_step=-1,  # learn only the first init_step steps for cheap training ?
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        output_path='/data/chercheurs/jeanner211/RESULTS/DCF-CelebA/ddpm',
        gpus='',
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
