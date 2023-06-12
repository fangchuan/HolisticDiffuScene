"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import os
import sys
sys.path.append(".") # Adds higher directory to python modules path.
sys.path.append("..") # Adds higher directory to python modules path.
import argparse
import datetime
import time

import numpy as np
import torch as th
import torch.distributed as dist

from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    log_dir = os.path.join(args.log_dir, datetime.datetime.now().strftime("openai-%Y-%m-%d-%H-%M-%S-%f"))
    logger.configure(dir=log_dir, format_strs=['tensorboard','stdout','log','csv'])

    logger.log("creating UNet model and diffusion model ...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    layout_channel_size = args.layout_channels
    layout_size = args.layout_size
    logger.log("sampling layout...")
    all_layout_lst = []
    all_layout_type_lst = []
    while len(all_layout_lst) * args.batch_size < args.num_samples:
        begin_tms = time.time()
        model_kwargs = {}
        if args.b_class_cond:
            # ignore 'undefined' class
            max_layout_types = (NUM_CLASSES-1)
            layout_type_lst = th.randint(low=0, high=max_layout_types, size=(args.batch_size,), device=dist_util.dev())
            layout_type_lst = th.full((args.batch_size,), 2, device=dist_util.dev())
            model_kwargs["y"] = layout_type_lst
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model=model,
            shape=(args.batch_size, layout_channel_size, layout_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        # calc sampling time
        elaps_time = time.time() - begin_tms
        logger.log(f'sample shape: {sample.shape}')
        logger.log(f'sample time: {elaps_time}')

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_layout_lst.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.b_class_cond:
            gathered_labels = [
                th.zeros_like(layout_type_lst) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, layout_type_lst)
            all_layout_type_lst.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_layout_lst) * args.batch_size} samples")

    arr = np.concatenate(all_layout_lst, axis=0)
    arr = arr[: args.num_samples]
    if args.b_class_cond:
        label_arr = np.concatenate(all_layout_type_lst, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.b_class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        log_dir='sample_results',
        clip_denoised=True,
        num_samples=10,
        batch_size=1,
        use_ddim=False,
        model_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
