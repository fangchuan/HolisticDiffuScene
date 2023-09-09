"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import os
import sys

sys.path.append(".")  # Adds higher directory to python modules path.
sys.path.append("..")  # Adds higher directory to python modules path.
import argparse
import datetime
import time

import numpy as np
import torch as th
import torch.distributed as dist
from PIL import Image

from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

from dataset.st3d_dataset import ST3DDataset
from improved_diffusion.clip_util import FrozenCLIPEmbedder
from dataset.metadata import ST3D_BEDROOM_QUAD_WALL_MAX_LEN, \
                            ST3D_LIVINGROOM_QUAD_WALL_MAX_LEN, \
                            COLOR_TO_ADEK_LABEL

from postprocess.gen_room_layout_mesh import recover_quad_wall_layout_mesh
from misc.equirect_projection import vis_objs3d, vis_floor_ceiling_simple
from preprocess.prepare_st3d_dataset import vis_scene_mesh
from misc.utils import reconstrcut_floor_ceiling_from_quad_walls

TEXT_PROMPT_LST = [
    "The bedroom has five walls. The room has a door, a bed and a nightstand. The nightstand is beside the door.",
    "The bedroom has four walls. There is a bed in the middle of the room. There is a window on the wall. There is a desk next to the bed.",
    "The bedroom has four walls. There is a chair next to the desk. There is a lamp on the desk. There is a door on the wall. ",
    "The bedroom has six walls. There is a cabinet next to the door. There is a mirror on the cabinet. There is a carpet on the floor.",
    "The bedroom has four walls. There is a bed in the middle of the room. There is a window on the wall. There is a desk next to the bed.",
    "The bedroom has four walls.The room has a curtain , a cabinet and a door .There is a lamp to the right of the door .",
    "The bedroom has eight walls. There is a picture on the wall. The television is in front of the bed. There is a chair next to the bed.",
    "The bedroom has seven walls. The room has a bed, a window and a cabinet. The window is beside the door.",
    "The bedroom has fiv walls. The room has a bed, a window and a curtain, but doesnot has a lamp.",
    "The bedroom has six walls. The room has a door, a bed and a window.",
]


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    log_dir = os.path.join(args.log_dir, datetime.datetime.now().strftime("openai-%Y-%m-%d-%H-%M-%S-%f"))
    logger.configure(dir=log_dir, format_strs=['tensorboard', 'stdout', 'log', 'csv'])

    text_encoder = FrozenCLIPEmbedder(device=dist_util.dev())
    dataset = ST3DDataset(root_dir=args.data_dir, max_text_sentences=4, return_scene_name=True)

    logger.log("creating UNet model and diffusion model ...")
    model, diffusion = create_model_and_diffusion(**args_to_dict(args, model_and_diffusion_defaults().keys()))
    model.load_state_dict(dist_util.load_state_dict(args.model_path, map_location="cpu"))
    model.to(dist_util.dev())
    model.eval()

    layout_channel_size = args.layout_channels
    layout_size = args.layout_size
    logger.log("sampling layout...")
    all_layout_lst = []
    all_layout_type_lst = []

    cond_text_prompt_lst = []
    scene_names_lst = []
    while len(all_layout_lst) * args.batch_size < args.num_samples:
        begin_tms = time.time()
        model_kwargs = {}
        if args.b_class_cond:
            # ignore 'undefined' class
            max_layout_types = (NUM_CLASSES - 1)
            layout_type_lst = th.randint(low=0, high=max_layout_types, size=(args.batch_size,), device=dist_util.dev())
            layout_type_lst = th.full((args.batch_size,), 2, device=dist_util.dev())
            model_kwargs["y"] = layout_type_lst
        if args.b_text_cond:
            cond_data_lst = []
            for i in range(args.batch_size):
                scene_idx = np.random.choice(len(dataset))
                gt_scene, gt_cond_dict, scene_name = dataset[scene_idx]
                # text prompt from eval dataset
                cond_data_lst.append(gt_cond_dict['context'])
                text_prompt = gt_cond_dict['text']
                cond_text_prompt_lst.append(text_prompt)
                scene_names_lst.append(scene_name)
                logger.log('text_prompt: {}'.format(text_prompt))

                # text prompt from predefined list
                # text_prompt = TEXT_PROMPT_LST[np.random.choice(len(TEXT_PROMPT_LST))]
                # logger.log('text_prompt: {}'.format(text_prompt))
                # cond_text_prompt_lst.append(text_prompt)
                # cond_data_lst.append(text_encoder.get_text_embeds(text_prompt).unsqueeze(0).cpu().numpy())
            model_kwargs["context"] = th.from_numpy(np.stack(cond_data_lst)).to(dist_util.dev(), dtype=th.float32)
        sample_fn = (diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop)
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
            gathered_labels = [th.zeros_like(layout_type_lst) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_labels, layout_type_lst)
            all_layout_type_lst.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_layout_lst) * args.batch_size} samples")

    samples_arr = np.concatenate(all_layout_lst, axis=0)
    samples_arr = samples_arr[:args.num_samples]
    samples_arr = np.transpose(samples_arr, (0, 2, 1))

    sample_result_folder = os.path.join(logger.get_dir(), f'{args.room_type}')
    if not os.path.exists(sample_result_folder):
        os.makedirs(sample_result_folder)

    cond_text_prompt_lst = cond_text_prompt_lst[:args.num_samples]
    if args.b_class_cond:
        label_arr = np.concatenate(all_layout_type_lst, axis=0)
        label_arr = label_arr[:args.num_samples]
    elif args.b_text_cond:
        text_prompt_path = os.path.join(sample_result_folder, f"text_prompt.txt")
        with open(text_prompt_path, 'w') as f:
            for i in range(args.num_samples):
                f.write(f'{cond_text_prompt_lst[i]}\n')

    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in samples_arr.shape])
        out_path = os.path.join(sample_result_folder, f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.b_class_cond:
            np.savez(out_path, samples_arr, label_arr)
        else:
            np.savez(out_path, samples_arr)

    dist.barrier()

    # load sample results: BxNxChannel
    for idx, scene_name in enumerate(scene_names_lst):
        scene_sample_result = samples_arr[idx]
        scene_sample_label = args.room_type
        print(f'scene_sample_label: {scene_sample_label}')

        if args.room_type == 'bedroom':
            room_layout_size = np.array([3.64073229, 3.73553261, 2.81591231])  # bedroom
            wall_max_num = ST3D_BEDROOM_QUAD_WALL_MAX_LEN
        elif args.room_type == 'livingroom':
            room_layout_size = np.array([7.239328956952291, 7.6231320936720675, 2.857928654068745])  # livingroom
            wall_max_num = ST3D_LIVINGROOM_QUAD_WALL_MAX_LEN
        # quad walls
        quad_wall_lst = scene_sample_result[:wall_max_num, :]
        # objects
        obj_bbox_lst = scene_sample_result[wall_max_num:, :]

        wall_dict_lst, obj_bbox_dict_lst = recover_quad_wall_layout_mesh(dataset_type='st3d',
                                                                         room_type=args.room_type,
                                                                         quad_wall_lst=quad_wall_lst,
                                                                         object_bbox_lst=obj_bbox_lst,
                                                                         room_layout_bbox_size=room_layout_size)

        # save synthesis results as image
        out_img = np.zeros((512, 1024, 3), np.uint8)
        cam_position = np.zeros((3,), np.float32)
        floor_points, ceiling_points = reconstrcut_floor_ceiling_from_quad_walls(quad_walls_lst=wall_dict_lst)
        out_img = vis_floor_ceiling_simple(image=out_img, color_to_labels=COLOR_TO_ADEK_LABEL)
        out_img = vis_objs3d(out_img,
                             v_bbox3d=(wall_dict_lst + obj_bbox_dict_lst),
                             camera_position=cam_position,
                             color_to_labels=COLOR_TO_ADEK_LABEL,
                             b_show_axes=False,
                             b_show_centroid=False,
                             b_show_bbox3d=False,
                             b_show_info=False,
                             b_show_polygen=True)

        curr_sample_folder = os.path.join(sample_result_folder, f'{idx}')
        os.makedirs(curr_sample_folder, exist_ok=True)
        save_img_filepath = os.path.join(curr_sample_folder, f'{scene_name}.png')
        Image.fromarray(out_img).save(save_img_filepath)

        # save synthetic object and room_layout as ply
        scene_bbox_ply_fname = f'{scene_name}.ply'
        scene_bbox_ply_filepath = os.path.join(curr_sample_folder, scene_bbox_ply_fname)
        scene_mesh = vis_scene_mesh(room_layout_mesh=None,
                                    obj_bbox_lst=(wall_dict_lst + obj_bbox_dict_lst),
                                    room_layout_bbox=None)
        scene_mesh.export(scene_bbox_ply_filepath)

    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        data_dir='data',
        log_dir='sample_results',
        clip_denoised=True,
        num_samples=10,
        batch_size=1,
        use_ddim=False,
        model_path="",
        room_type='bedroom',
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
