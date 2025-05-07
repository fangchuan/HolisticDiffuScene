import os
import sys

sys.path.append(".")  # Adds higher directory to python modules path.
sys.path.append("..")  # Adds higher directory to python modules path.
import argparse
import datetime

import numpy as np
import torch
from thop import profile
from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def create_argparser():
    defaults = dict(
        data_dir='/mnt/nas_3dv/hdd1/datasets/Structured3d/preprocessed/text2pano/test/bedroom/',
        log_dir='sample_results',
        clip_denoised=True,
        num_samples=10,
        batch_size=1,
        use_ddim=False,
        model_path=
        "/mnt/nas_3dv/hdd1/fangchuan/HolisticDiffuScene/log/ST3D_bedroom_textcondition_openai-2023-09-08-15-04-50-375770/ema_0.9999_180000.pt",
        room_type='bedroom',
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def test_layout_diffusion_network():
    args = create_argparser().parse_args()

    # model config
    args.layout_channels = 32
    args.layout_size = 23
    args.num_channels = 128
    args.num_res_blocks = 3
    args.b_learn_sigma = True
    args.b_class_cond = False
    args.b_text_cond = True
    args.use_input_encoding = False

    # diffusion config
    args.diffusion_steps = 4000
    args.noise_schedule = "cosine"
    args.timestep_respacing = '250'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.log("creating UNet model and diffusion model ...")
    model, diffusion = create_model_and_diffusion(**args_to_dict(args, model_and_diffusion_defaults().keys()))

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)

    # calculate the number of trainable parameters
    n_all_params = int(sum([np.prod(p.size()) for p in model.parameters()]))
    n_trainable_params = int(sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, model.parameters())]))
    print(f"Number of parameters in {model.__class__.__name__}:  {n_trainable_params} / {n_all_params}")

    # profile the model
    input = torch.randn(1, args.layout_channels, args.layout_size).to(device)
    timestep = torch.tensor([4000] * input.shape[0], device=device)
    text_emb = torch.randn(1, 77, 768).to(device)
    context = {'context': text_emb}
    ops, params = profile(model, inputs=(input, timestep, text_emb))
    print(f"MACs: {ops}, params: {params}")
    print('FLOPs = ' + str(ops / 1000**3) + 'G')
    print('Params = ' + str(params / 1000**2) + 'M')

def test_appearance_diffusion_network():
    import random
    import datetime
    import argparse
    import glob
    import gc

    from share import *
    import config

    import cv2
    from PIL import Image, ImageOps
    import einops
    import gradio as gr
    import numpy as np
    import torch

    from typing import List, Dict, Any

    from pytorch_lightning import seed_everything
    from annotator.util import resize_image, HWC3
    from cldm.model import create_model, load_state_dict
    from cldm.ddim_hacked import DDIMSampler

    from annotator.oneformer.oneformer.data.datasets.register_ade20k_panoptic import ADE20K_150_CATEGORIES

    ade_labels = [label_dict["name"] for label_dict in ADE20K_150_CATEGORIES]
    # print(f'ade_labels: {ade_labels}')
    ade_colors = [list(label_dict["color"]) for label_dict in ADE20K_150_CATEGORIES]


    def load_pano_gen_model(ckpt_filepath:str, device: str = 'cuda'):
    model_name = 'control_v11p_sd15_seg'
    model = create_model(f'../models/{model_name}.yaml').cpu()
    # model.load_state_dict(load_state_dict(f'../ckpts/{model_name}_livingroom_fullres_40000.ckpt', location=device), strict=False)
    model.load_state_dict(load_state_dict(ckpt_filepath, location=device), strict=False)
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)

    return model, ddim_sampler

if __name__ == "__main__":
    test_layout_diffusion_network()
