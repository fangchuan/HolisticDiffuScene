import os
import sys

sys.path.append(".")  # Adds higher directory to python modules path.
sys.path.append("..")  # Adds higher directory to python modules path.

import argparse
import datetime

import numpy as np

import torch as th
import torch.nn as nn

from shapely.geometry.polygon import Polygon

from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)

from misc.utils import euler_angle_to_matrix

from improved_diffusion.losses import (
    bbox_corners,
    #    iou_among_layout_and_predicted_3d_bbox,
    verify_object_box_on_wall,
)
from improved_diffusion import logger
from dataset.st3d_dataset import ROOM_TYPE_DICT, get_room_type
from dataset.metadata import ST3D_BEDROOM_FURNITURE, ST3D_LIVINGROOM_FURNITURE, ST3D_DININGROOM_FURNITURE


def create_argparser():
    """ create argparser for data, model, and training configuration

    Returns:
        parser: _description_
    """
    defaults = dict(
        data_dir="/mnt/nas_3dv/hdd1/datasets/Structured3d/preprocessed/all_raw_light/bedroom",
        log_dir='log',
        samples_filepath='sample_results/openai-2023-07-09-16-55-49-751667/samples_10x23x32.npz',
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    # automatically add arguments
    add_dict_to_argparser(parser, defaults)
    return parser


def iou_among_layout_and_predicted_3d_bbox(x_pred: th.Tensor, room_type_lst: th.Tensor, iou_loss_weights: th.Tensor):
    # quad_walls: x_pred[:, :10, :]
    # object_bbox: x_pred[:, 10:, :]
    B, C, feat_size = x_pred.shape
    object_chann_idx = 10
    assert th.all(room_type_lst == room_type_lst[0]), "The input room types should be equal"
    assert iou_loss_weights.shape == (B, C, feat_size), "The loss weights tensor should be (B, 23, 32)"

    if get_room_type(room_type_lst[0]) == 'bedroom':
        class_labels_lst = ST3D_BEDROOM_FURNITURE
        max_wall_num, max_obj_num, obj_feat_dim = 10, 13, 32
    elif get_room_type(room_type_lst[0]) == 'living room':
        class_labels_lst = ST3D_LIVINGROOM_FURNITURE
        max_wall_num, max_obj_num, obj_feat_dim = 10, 23, 32
    elif get_room_type(room_type_lst[0]) == 'dining room':
        class_labels_lst = ST3D_DININGROOM_FURNITURE
        max_wall_num, max_obj_num, obj_feat_dim = 10, 23, 32
    else:
        raise NotImplementedError

    class_idx = 0
    centroid_idx = class_idx + len(class_labels_lst)
    size_idx = 3 + centroid_idx
    angle_idx = 3 + size_idx

    # get valid quad wall bbox
    pred_quad_wall_bbox = x_pred[:, 0:object_chann_idx, :].reshape(B, max_wall_num, obj_feat_dim)
    # pred_quad_wall_bbox = pred_quad_wall_bbox[invalid_masks[:, 0:object_chann_idx, :] == False]

    # get valid object bbox
    pred_object_bbox = x_pred[:, object_chann_idx:, :].reshape(B, max_obj_num, obj_feat_dim)
    # pred_object_bbox = pred_object_bbox[invalid_masks[:, object_chann_idx:, :] == False]

    pred_quad_wall_class_prob = th.where(pred_quad_wall_bbox[:, :, 0:centroid_idx] > 0.5, 1, 0)
    pred_object_class_prob = th.where(pred_object_bbox[:, :, 0:centroid_idx] > 0.5, 1, 0)
    logger.debug(f'pred_quad_wall_class_prob: {pred_quad_wall_class_prob.shape}')
    # skip probability of empty object
    no_wall_mask = th.all(pred_quad_wall_class_prob == 0, dim=2, keepdim=True)
    no_object_mask = th.all(pred_object_class_prob == 0, dim=2, keepdim=True)
    pred_quad_wall_class = th.argmax(pred_quad_wall_class_prob, dim=2, keepdim=True)
    pred_object_class = th.argmax(pred_object_class_prob, dim=2, keepdim=True)
    logger.debug(f'pred_quad_wall_class: {pred_quad_wall_class.shape}')
    # skip empty object, door, window, curtain
    no_wall_mask = th.logical_or(no_wall_mask,
                                 th.all(pred_quad_wall_class == class_labels_lst.index('empty'), dim=2, keepdim=True))
    no_object_mask = th.logical_or(no_object_mask,
                                   th.all(pred_object_class == class_labels_lst.index('empty'), dim=2, keepdim=True))
    no_object_mask = th.logical_or(no_object_mask,
                                   th.all(pred_object_class == class_labels_lst.index('door'), dim=2, keepdim=True))
    no_object_mask = th.logical_or(no_object_mask,
                                   th.all(pred_object_class == class_labels_lst.index('window'), dim=2, keepdim=True))
    no_object_mask = th.logical_or(no_object_mask,
                                   th.all(pred_object_class == class_labels_lst.index('curtain'), dim=2, keepdim=True))
    no_object_mask = th.logical_or(no_object_mask,
                                   th.all(pred_object_class == class_labels_lst.index('picture'), dim=2, keepdim=True))
    no_object_mask = th.logical_or(
        no_object_mask, th.all(pred_object_class == class_labels_lst.index('television'), dim=2, keepdim=True))
    valid_wall_masks = ~no_wall_mask
    valid_object_masks = ~no_object_mask

    def not_door_or_window(obj_sem_cls, all_cls_labels):
        return obj_sem_cls != all_cls_labels.index('door') and \
               obj_sem_cls != all_cls_labels.index('window') and \
                obj_sem_cls != all_cls_labels.index('curtain') and \
                obj_sem_cls != all_cls_labels.index('picture') and \
                obj_sem_cls != all_cls_labels.index('television')

    pred_object_centroid = pred_object_bbox[:, :, centroid_idx:size_idx].clamp(min=-1.0, max=1.0)
    pred_object_centroid = th.where(pred_object_centroid.isnan(), 0.0, pred_object_centroid)
    pred_quad_wall_centroid = pred_quad_wall_bbox[:, :, centroid_idx:size_idx].clamp(min=-1.0, max=1.0)
    pred_quad_wall_centroid = th.where(pred_quad_wall_centroid.isnan(), 0.0, pred_quad_wall_centroid)
    logger.debug(f'pred_object_centroid: {pred_object_centroid.shape}')

    # pred_object_size = ((pred_object_bbox[:, :, size_idx:angle_idx] + 1) * 0.5).clamp(min=1e-4, max=1.0)
    pred_object_size = (pred_object_bbox[:, :, size_idx:angle_idx]).clamp(min=1e-4, max=1.0)
    pred_object_size = th.where(pred_object_size.isnan(), 0.0, pred_object_size)
    pred_quad_wall_size = (pred_quad_wall_bbox[:, :, size_idx:angle_idx]).clamp(min=1e-4, max=1.0)
    pred_quad_wall_size = th.where(pred_quad_wall_size.isnan(), 0.0, pred_quad_wall_size)
    logger.debug(f'pred_object_size: {pred_object_size.shape}')

    pred_object_cos_angle = pred_object_bbox[:, :, angle_idx:angle_idx + 1].clamp(min=-0.999999, max=0.999999)
    pred_object_sin_angle = pred_object_bbox[:, :, angle_idx + 1:angle_idx + 2].clamp(min=-0.999999, max=0.999999)
    # TODO: here we choose 5e-3 as threshold, but it is not a good choice, try to add it into hyper-parameters
    # pred_object_angle = th.where(
    #     th.abs(pred_object_cos_angle) < 5e-3, th.arcsin(pred_object_sin_angle), th.arccos(pred_object_cos_angle))
    pred_object_angle = th.arccos(pred_object_cos_angle)
    pred_object_eulers = th.concat(
        (th.zeros_like(pred_object_angle), th.zeros_like(pred_object_angle), pred_object_angle), dim=2)
    logger.debug(f'pred_object_eulers: {pred_object_eulers.shape}')

    # recover wall normal
    pred_quad_wall_cos_angle = pred_quad_wall_bbox[:, :, angle_idx:angle_idx + 1].clamp(min=-0.999999, max=0.999999)
    pred_quad_wall_sin_angle = pred_quad_wall_bbox[:, :, angle_idx + 1:angle_idx + 2].clamp(min=-0.999999, max=0.999999)
    pred_quad_wall_angle = th.where(
        th.abs(pred_quad_wall_cos_angle) < 5e-3, th.arcsin(pred_quad_wall_sin_angle),
        th.arccos(pred_quad_wall_cos_angle))
    # B x wall_num x 3
    pred_quad_wall_eulers = th.concat(
        (th.zeros_like(pred_quad_wall_angle), th.zeros_like(pred_quad_wall_angle), pred_quad_wall_angle), dim=2)
    logger.debug(f'pred_quad_wall_eulers: {pred_quad_wall_eulers.shape}')
    # B x wall_num x 1 x 3
    camera_orientation = th.tensor([0.0, -1.0, 0.0], device=x_pred.device).reshape(1, 1, 1,
                                                                                   3).repeat(B, max_wall_num, 1, 1)
    # logger.debug(f'camera_orientation: {camera_orientation.shape}')
    pred_quad_wall_normal = (
        euler_angle_to_matrix(pred_quad_wall_eulers) @ camera_orientation.transpose(2, 3)).transpose(2, 3)
    pred_quad_wall_normal = pred_quad_wall_normal.squeeze(2)
    logger.debug(f'pred_quad_wall_normal: {pred_quad_wall_normal.shape}')

    # Bx13x8x3
    pred_object_box_corners_3d = bbox_corners(pred_object_centroid.unsqueeze(2), pred_object_size.unsqueeze(2),
                                              pred_object_eulers)
    pred_quad_wall_box_corners_3d = bbox_corners(pred_quad_wall_centroid.unsqueeze(2), pred_quad_wall_size.unsqueeze(2),
                                                 pred_quad_wall_eulers)
    # get x-y plane box corners
    pred_object_bbox_corners_2d = pred_object_box_corners_3d[:, :, 0:4, 0:2]
    pred_wall_bbox_corners_2d = pred_quad_wall_box_corners_3d[:, :, 0:4, 0:2]

    batch_physical_constraint_loss = []
    for batch_idx in range(B):
        phy_cons_loss = 0.0

        valid_object = valid_object_masks[batch_idx, :, 0].reshape(-1)
        valid_object_bbox_corners2d = pred_object_bbox_corners_2d[batch_idx, valid_object, :, :]
        obj_num = valid_object_bbox_corners2d.shape[0]
        obj_2d_corner_points = valid_object_bbox_corners2d.reshape(-1, 2)
        logger.debug(f'obj_2d_corner_points: {obj_2d_corner_points.shape}')

        valid_wall = valid_wall_masks[batch_idx, :, 0].reshape(-1)
        valid_quad_wall_bbox_corners2d = pred_wall_bbox_corners_2d[batch_idx, valid_wall, :, :]
        valid_quad_wall_centroid = pred_quad_wall_centroid[batch_idx, valid_wall, :]
        valid_quad_wall_normal = pred_quad_wall_normal[batch_idx, valid_wall, :]
        valid_quad_wall_size = pred_quad_wall_size[batch_idx, valid_wall, :]
        wall_num = valid_quad_wall_centroid.shape[0]

        # 2d box corners of predicted quad walls in x-y plane
        batch_pred_wall_corners_2d = []
        for wall_idx in range(wall_num):

            phy_cons_loss, collision = verify_object_box_on_wall(obj_2d_corner_points,
                                                                 valid_quad_wall_centroid[wall_idx],
                                                                 valid_quad_wall_normal[wall_idx],
                                                                 valid_quad_wall_size[wall_idx])

            logger.debug(f'wall {wall_idx} centroid: {valid_quad_wall_centroid[wall_idx]}')
            # logger.debug(f'wall {wall_idx} cos angle: {pred_quad_wall_cos_angle[batch_idx, wall_idx]}')
            # logger.debug(f'wall {wall_idx} sin angle: {pred_quad_wall_sin_angle[batch_idx, wall_idx]}')
            # logger.debug(f'wall {wall_idx} angle: {pred_quad_wall_angle[batch_idx, wall_idx]}')
            logger.debug(f'wall {wall_idx} normal: {valid_quad_wall_normal[wall_idx]}')
            logger.debug(f'wall {wall_idx} physical violation loss: {phy_cons_loss}')
            phy_cons_loss = phy_cons_loss + phy_cons_loss / obj_num

            batch_pred_wall_corners_2d.append(valid_quad_wall_bbox_corners2d[wall_idx, :, :])
        wall_2d_corner_points = th.cat(tuple(batch_pred_wall_corners_2d), 0)
        logger.debug(f'wall_2d_corner_points: {wall_2d_corner_points.shape}')
        # visualize collision
        # draw 2D projection on the horizontal plane (x-y plane)
        wall_polygen_x_lst, wall_polygen_y_lst = [], []
        obj_polygen_x_lst, obj_polygen_y_lst = [], []
        from matplotlib import pyplot

        for i in range(0, wall_2d_corner_points.shape[0], 4):
            corner1 = wall_2d_corner_points[i].cpu().numpy()
            corner2 = wall_2d_corner_points[i + 1].cpu().numpy()
            corner3 = wall_2d_corner_points[i + 2].cpu().numpy()
            corner4 = wall_2d_corner_points[i + 3].cpu().numpy()
            polygon2D_1 = Polygon([(corner1[0], corner1[1]), (corner2[0], corner2[1]), (corner3[0], corner3[1]),
                                   (corner4[0], corner4[1])])
            xx, yy = polygon2D_1.exterior.xy
            wall_polygen_x_lst.extend(xx.tolist())
            wall_polygen_y_lst.extend(yy.tolist())
            # from matplotlib import pyplot
            pyplot.plot(xx, yy)
            # pyplot.axis('equal')
            # pyplot.title('Walls')
            # pyplot.show()
        for j in range(0, obj_2d_corner_points.shape[0], 4):
            corner1 = obj_2d_corner_points[j].cpu().numpy()
            corner2 = obj_2d_corner_points[j + 1].cpu().numpy()
            corner3 = obj_2d_corner_points[j + 2].cpu().numpy()
            corner4 = obj_2d_corner_points[j + 3].cpu().numpy()
            polygon2D_2 = Polygon([(corner1[0], corner1[1]), (corner2[0], corner2[1]), (corner3[0], corner3[1]),
                                   (corner4[0], corner4[1])])
            xx, yy = polygon2D_2.exterior.xy
            obj_polygen_x_lst.extend(xx.tolist())
            obj_polygen_y_lst.extend(yy.tolist())
            # from matplotlib import pyplot
            pyplot.plot(xx, yy)
            # pyplot.title('Objects')
            # pyplot.axis('equal')
            # pyplot.show()

        pyplot.axis('equal')
        pyplot.show()
        batch_physical_constraint_loss.append(phy_cons_loss.reshape(1, 1))

    # Bx1x1
    batch_pred_physical_constraint_loss = th.stack(batch_physical_constraint_loss, dim=0)
    iou_loss_shape = batch_pred_physical_constraint_loss.shape[-1]
    batch_iou_loss = batch_pred_physical_constraint_loss * iou_loss_weights.reshape(B, 1, -1)[..., :iou_loss_shape]
    return batch_iou_loss


def main():
    args = create_argparser().parse_args()
    print(args)

    log_dir = os.path.join(args.log_dir, datetime.datetime.now().strftime("openai-%Y-%m-%d-%H-%M-%S-%f"))
    logger.configure(dir=log_dir, format_strs=['tensorboard', 'stdout', 'log', 'csv'])
    logger.set_level(logger.DEBUG)

    # load samples
    samples_filepath = args.samples_filepath
    samples = np.load(samples_filepath)
    print(f"loaded samples  {samples['arr_0'].shape}")
    # setup device
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    predictions = th.from_numpy(samples['arr_0']).to(device)
    room_types = th.from_numpy(samples['arr_1']).to(device)
    iou_weights = th.ones_like(predictions).to(device)
    physical_losses = iou_among_layout_and_predicted_3d_bbox(x_pred=predictions,
                                                             room_type_lst=room_types,
                                                             iou_loss_weights=iou_weights)
    print(f"physical_losses: {physical_losses}")


if __name__ == '__main__':
    main()