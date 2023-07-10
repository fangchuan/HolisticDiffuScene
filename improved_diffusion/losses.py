"""
Helpers for various likelihood-based losses. These are ported from the original
Ho et al. diffusion models codebase:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/utils.py
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import torch as th

from dataset.st3d_dataset import get_room_type
from dataset.metadata import ST3D_BEDROOM_FURNITURE, ST3D_LIVINGROOM_FURNITURE, ST3D_DININGROOM_FURNITURE
from . import logger
from shapely.geometry.polygon import Polygon
from misc.utils import euler_angle_to_matrix
from .rotated_iou_loss import bdb3d_iou

l1_critertion = th.nn.SmoothL1Loss(reduction='none')


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, th.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [x if isinstance(x, th.Tensor) else th.tensor(x).to(tensor) for x in (logvar1, logvar2)]

    return 0.5 * (-1.0 + logvar2 - logvar1 + th.exp(logvar1 - logvar2) + ((mean1 - mean2)**2) * th.exp(-logvar2))


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + th.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * th.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    用标准高斯分布的累积分布函数(概率密度函数)来近似离散高斯分布的概率
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    # 减去均值
    centered_x = x - means
    inv_stdv = th.exp(-log_scales)

    # 将[-1,1]分成255份，最左边的CDF为0，最右边的CDF记为1，
    # 那么每个bin中的CDF为 1/255的CDF
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)

    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)

    log_cdf_plus = th.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = th.log((1.0 - cdf_min).clamp(min=1e-12))

    # 用两个CDF的差值来近似离散高斯分布的概率
    cdf_delta = cdf_plus - cdf_min

    log_probs = th.where(
        x < -0.999,
        log_cdf_plus,
        th.where(x > 0.999, log_one_minus_cdf_min, th.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs


def continuous_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution on continuous value field.

    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    # 减去均值
    # centered_x = x - means
    # inv_stdv = th.exp(-log_scales)

    # # calculate gaussian log probability
    # log_probs = -0.5 * th.log(th.tensor(2 * th.pi, dtype=th.float32, device=means.device)) - log_scales + (
    #     -0.5 * (centered_x**2) * inv_stdv**2)

    predict_gaussian_dist = th.distributions.Normal(means, th.exp(log_scales))
    log_probs = predict_gaussian_dist.log_prob(x)

    # plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    # cdf_plus = approx_standard_normal_cdf(plus_in)

    # min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    # cdf_min = approx_standard_normal_cdf(min_in)

    # log_cdf_plus = th.log(cdf_plus.clamp(min=1e-12))
    # log_one_minus_cdf_min = th.log((1.0 - cdf_min).clamp(min=1e-12))
    # cdf_delta = cdf_plus - cdf_min
    # log_probs = th.where(
    #     x < -0.999,
    #     log_cdf_plus,
    #     th.where(x > 0.999, log_one_minus_cdf_min, th.log(cdf_delta.clamp(min=1e-12))),
    # )
    assert log_probs.shape == x.shape
    return log_probs


def bdb3d_corners(bdb3d: Dict) -> th.Tensor:
    """
    Get ordered corners of given 3D bounding box dict or disordered corners

    Parameters
    ----------
    bdb3d: 3D bounding box dict

    Returns
    -------
    8 x 3 numpy array of bounding box corner points in the following order:
    right-forward-down
    left-forward-down
    right-back-down
    left-back-down
    right-forward-up
    left-forward-up
    right-back-up
    left-back-up
    """
    # corners = np.unpackbits(np.arange(8, dtype=np.uint8)[..., np.newaxis],
    #                         axis=1, bitorder='little', count=-5).astype(np.float32)
    device = bdb3d['center'].device
    corners = th.zeros((8, 3), dtype=th.float32)
    # corners[0, :] = th.tensor([1., 1., 0.])
    # corners[1, :] = th.tensor([0., 1., 0.])
    # corners[2, :] = th.tensor([1., 0., 0.])
    # corners[3, :] = th.tensor([0., 0., 0.])
    # corners[4, :] = th.tensor([1., 1., 1.])
    # corners[5, :] = th.tensor([0., 1., 1.])
    # corners[6, :] = th.tensor([1., 0., 1.])
    # corners[7, :] = th.tensor([0., 0., 1.])
    corners = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]) - 0.5
    corners = th.tensor(corners, dtype=th.float32, device=device)
    # corners = corners.to(device)
    # corners = corners - th.tensor(0.5, dtype=th.float32, device=device)
    # logger.info(f'corners: {corners.device}')
    # logger.info(f'corners: {corners}')

    rotation = euler_angle_to_matrix(bdb3d['angles']).to(device)
    # logger.info(f'rotation: {rotation.device}')
    centroid = bdb3d['center']
    # logger.info(f'centroid: {centroid.device}')
    sizes = bdb3d['size']
    # logger.info(f'sizes: {sizes.device}')

    corners = th.mul(corners, sizes)
    # logger.info(f'corners: {corners.device}')
    corners = th.matmul(rotation, corners.t()).t()
    # logger.info(f'corners: {corners.device}')
    return corners + centroid


def bbox_corners(bbox_centroids: th.Tensor, bbox_sizes: th.Tensor, bbox_angles: th.Tensor) -> th.Tensor:
    device = bbox_centroids.device
    box_corner_vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1],
                                    [0, 1, 1]]) - 0.5
    box_corner_vertices = th.tensor(box_corner_vertices, dtype=th.float32, device=device)
    box_corner_vertices = box_corner_vertices.repeat(bbox_centroids.shape[0], bbox_centroids.shape[1], 1, 1)
    # logger.info(f'box_corner_vertices.shape: {box_corner_vertices.shape}')
    # Bx13x8x3
    corners = th.mul(box_corner_vertices, bbox_sizes)
    # Bx13x3x3
    rotation = euler_angle_to_matrix(bbox_angles).to(device)
    # logger.info(f'rotation.shape: {rotation.shape}')
    corners = th.matmul(rotation, corners.transpose(dim0=2, dim1=3)).transpose(dim0=2, dim1=3)
    # logger.info(f'corners.shape: {corners.shape}')
    return corners + bbox_centroids


def verify_object_box_on_wall(point: th.Tensor, wall_center: th.Tensor, wall_normal: th.Tensor, wall_size: th.Tensor):
    """verify if object 2d box is on wall plane

    Args:
        point (th.Tensor): 2d box corners of observed objects
        wall_center (th.Tensor): wall centroid
        wall_normal (th.Tensor): wall normal 
        wall_size (th.Tensor): wall size

    Returns:
        physical constriant loss: object box intersect with wall plane
        physical collision number: how many times object box intersect with wall plane
    """
    # (Nx4)x2
    P = point.shape[0]
    a = wall_normal[0]
    b = wall_normal[1]
    d = -(a * wall_center[0] + b * wall_center[1])

    k = -(a * point[:, 0] + b * point[:, 1] + d)

    x = point[:, 0] + a * k
    y = point[:, 1] + b * k

    t = th.cat((x.reshape(P, 1), y.reshape(P, 1)), dim=-1)
    w = th.norm(t - wall_center[0:2], dim=1)

    point_mask = th.zeros(P).to(point.device)
    collision = th.zeros(P).to(point.device)

    point_mask[w < wall_size[0] / 2] = 1
    quad = th.cat((a.view([1]), b.view([1])))
    delta = point.matmul(quad) + d
    physical_constraint_loss = th.relu(-delta) * point_mask
    collision[physical_constraint_loss > 1e-4] = 1
    return physical_constraint_loss.sum(), collision.sum()


def iou_among_layout_and_predicted_3d_bbox(x_pred: th.Tensor, room_type_lst: th.Tensor, iou_loss_weights: th.Tensor):
    # quad_walls: x_pred[:, :10, :]
    # object_bbox: x_pred[:, 10:, :]
    B, C, feat_size = x_pred.shape
    object_chann_idx = 10
    assert th.all(room_type_lst == room_type_lst[0]), "The input room types should be equal"
    assert iou_loss_weights.shape == (B, C, feat_size), "The loss weights tensor should be (B, 23, 32)"

    if get_room_type(room_type_lst[0]) == 'bedroom':
        class_labels_lst = ST3D_BEDROOM_FURNITURE
        wall_num, obj_feat_num, obj_feat_dim = 10, 13, 32
    elif get_room_type(room_type_lst[0]) == 'living room':
        class_labels_lst = ST3D_LIVINGROOM_FURNITURE
        wall_num, obj_feat_num, obj_feat_dim = 10, 23, 32
    elif get_room_type(room_type_lst[0]) == 'dining room':
        class_labels_lst = ST3D_DININGROOM_FURNITURE
        wall_num, obj_feat_num, obj_feat_dim = 10, 23, 32
    else:
        raise NotImplementedError

    class_idx = 0
    centroid_idx = class_idx + len(class_labels_lst)
    size_idx = 3 + centroid_idx
    angle_idx = 3 + size_idx

    # Bx10x32
    pred_quad_wall_bbox = x_pred[:, 0:object_chann_idx, :].reshape(B, wall_num, obj_feat_dim)
    pred_object_bbox = x_pred[:, object_chann_idx:, :].reshape(B, obj_feat_num, obj_feat_dim)
    pred_quad_wall_class_prob = th.where(pred_quad_wall_bbox[:, :, 0:centroid_idx] > 0.5, 1, 0)
    pred_object_class_prob = th.where(pred_object_bbox[:, :, 0:centroid_idx] > 0.5, 1, 0)
    logger.debug(f'pred_quad_wall_class_prob: {pred_quad_wall_class_prob.shape}')
    # skip probability of empty object
    no_wall_mask = th.all(pred_quad_wall_class_prob == 0, dim=2, keepdim=True)
    no_object_mask = th.all(pred_object_class_prob == 0, dim=2, keepdim=True)
    pred_quad_wall_class = th.argmax(pred_quad_wall_class_prob, dim=2, keepdim=True)
    pred_object_class = th.argmax(pred_object_class_prob, dim=2, keepdim=True)
    logger.debug(f'pred_quad_wall_class: {pred_quad_wall_class.shape}')
    # skip empty object
    no_wall_mask = th.logical_or(no_wall_mask,
                                 th.all(pred_quad_wall_class == class_labels_lst.index('empty'), dim=2, keepdim=True))
    no_object_mask = th.logical_or(no_object_mask,
                                   th.all(pred_object_class == class_labels_lst.index('empty'), dim=2, keepdim=True))

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
    pred_object_angle = th.where(
        th.abs(pred_object_cos_angle) < 5e-3, th.arcsin(pred_object_sin_angle), th.arccos(pred_object_cos_angle))
    pred_object_eulers = th.concat(
        (th.zeros_like(pred_object_angle), th.zeros_like(pred_object_angle), pred_object_angle), dim=2)
    logger.debug(f'pred_object_eulers: {pred_object_eulers.shape}')

    pred_quad_wall_cos_angle = pred_quad_wall_bbox[:, :, angle_idx:angle_idx + 1].clamp(min=-0.999999, max=0.999999)
    pred_quad_wall_sin_angle = pred_quad_wall_bbox[:, :, angle_idx + 1:angle_idx + 2].clamp(min=-0.999999, max=0.999999)
    pred_quad_wall_angle = th.where(
        th.abs(pred_quad_wall_cos_angle) < 5e-3, th.arcsin(pred_quad_wall_sin_angle),
        th.arccos(pred_quad_wall_cos_angle))
    pred_quad_wall_eulers = th.concat(
        (th.zeros_like(pred_quad_wall_angle), th.zeros_like(pred_quad_wall_angle), pred_quad_wall_angle), dim=2)
    logger.debug(f'pred_quad_wall_eulers: {pred_quad_wall_eulers.shape}')
    camera_orientation = th.tensor([0.0, -1.0, 0.0], device=x_pred.device).reshape(1, 1, 1, 3).repeat(B, wall_num, 1, 1)
    logger.debug(f'camera_orientation: {camera_orientation.shape}')
    pred_quad_wall_normal = (
        euler_angle_to_matrix(pred_quad_wall_eulers) @ camera_orientation.transpose(2, 3)).transpose(2, 3)
    pred_quad_wall_normal = pred_quad_wall_normal.squeeze(2)
    logger.debug(f'pred_quad_wall_normal: {pred_quad_wall_normal.shape}')

    # Bx13x8x3
    pred_object_box_corners_3d = bbox_corners(pred_object_centroid.unsqueeze(2), pred_object_size.unsqueeze(2),
                                              pred_object_eulers)
    # get x-y plane box corners
    pred_object_bbox_corners_2d = pred_object_box_corners_3d[:, :, 0:4, 0:2]

    batch_physical_constraint_loss = []
    for batch_idx in range(B):
        phy_cons_loss = 0.0
        num_obj_box = 0
        # 2d box corners of predicted objects in x-y plane
        batch_pred_obj_box_corners_2d = []
        for obj_idx in range(0, obj_feat_num):
            if (not no_object_mask[batch_idx, obj_idx]) and not_door_or_window(pred_object_class[batch_idx, obj_idx],
                                                                               class_labels_lst):
                num_obj_box = num_obj_box + 1
                batch_pred_obj_box_corners_2d.append(pred_object_bbox_corners_2d[batch_idx, obj_idx, :, :])
        if len(batch_pred_obj_box_corners_2d) == 0:
            continue
        # (num_objx4)x2
        obj_2d_corner_points = th.cat(tuple(batch_pred_obj_box_corners_2d), 0)
        logger.debug(f'obj_2d_corner_points: {obj_2d_corner_points.shape}')

        # 2d box corners of predicted quad walls in x-y plane
        for wall_idx in range(wall_num):

            if not no_wall_mask[batch_idx, wall_idx]:
                phy_cons_loss, collision = verify_object_box_on_wall(obj_2d_corner_points,
                                                                     pred_quad_wall_centroid[batch_idx, wall_idx],
                                                                     pred_quad_wall_normal[batch_idx, wall_idx],
                                                                     pred_quad_wall_size[batch_idx, wall_idx])
                phy_cons_loss = phy_cons_loss + phy_cons_loss / num_obj_box

        batch_physical_constraint_loss.append(phy_cons_loss.reshape(1, 1))

    # Bx1x1
    batch_pred_physical_constraint_loss = th.stack(batch_physical_constraint_loss, dim=0)
    iou_loss_shape = batch_pred_physical_constraint_loss.shape[-1]
    # logger.debug(f'batch_pred_bbox_iou_loss berfore weighting: {batch_pred_bbox_iou_loss}')
    batch_iou_loss = batch_pred_physical_constraint_loss * iou_loss_weights.reshape(B, 1, -1)[..., :iou_loss_shape]
    return batch_iou_loss


def iou_among_predicted_3d_bbox(x_pred, room_type_lst, iou_loss_weights):
    # quad_walls: x_pred[:, :10, :]
    # object_bbox: x_pred[:, 10:, :]
    B, C, feat_size = x_pred.shape
    object_chann_idx = 10
    assert th.all(room_type_lst == room_type_lst[0]), "The input room types should be equal"
    assert iou_loss_weights.shape == (B, C, feat_size), "The loss weights tensor should be (B, 23, 32)"

    if get_room_type(room_type_lst[0]) == 'bedroom':
        class_labels_lst = ST3D_BEDROOM_FURNITURE
        obj_feat_num, obj_feat_dim = 13, 32
    elif get_room_type(room_type_lst[0]) == 'living room':
        class_labels_lst = ST3D_LIVINGROOM_FURNITURE
        obj_feat_num, obj_feat_dim = 23, 32
    elif get_room_type(room_type_lst[0]) == 'dining room':
        class_labels_lst = ST3D_DININGROOM_FURNITURE
        obj_feat_num, obj_feat_dim = 23, 32
    else:
        raise NotImplementedError

    class_idx = 0
    centroid_idx = class_idx + len(class_labels_lst)
    size_idx = 3 + centroid_idx
    angle_idx = 3 + size_idx

    # Bx13x30
    pred_object_bbox = x_pred[:, object_chann_idx:, :].reshape(B, obj_feat_num, obj_feat_dim)
    pred_object_class_prob = th.where(pred_object_bbox[:, :, :centroid_idx] > 0.5, 1, 0)
    # logger.debug(f'pred_object_class_prob: {pred_object_class_prob.shape}')
    # skip probability of empty object
    no_object_mask = th.all(pred_object_class_prob == 0, dim=2, keepdim=True)
    pred_object_class = th.argmax(pred_object_class_prob, dim=2, keepdim=True)
    # logger.debug(f'pred_object_class: {pred_object_class.shape}')
    # skip empty object
    no_object_mask = th.logical_or(no_object_mask,
                                   th.all(pred_object_class == class_labels_lst.index('empty'), dim=2, keepdim=True))
    # skip curtain and window
    curtain_mask = th.all(pred_object_class == class_labels_lst.index('curtain'), dim=2, keepdim=True)
    window_mask = th.all(pred_object_class == class_labels_lst.index('window'), dim=2, keepdim=True)

    # skip bed and pillow
    bed_mask = th.all(pred_object_class == class_labels_lst.index('bed'), dim=2, keepdim=True)
    pillow_mask = th.all(pred_object_class == class_labels_lst.index('pillow'), dim=2, keepdim=True)

    # BxCx1
    # logger.debug(f'no_object_mask[0,...]: {no_object_mask[0, ...]}')

    pred_object_centroid = pred_object_bbox[:, :, centroid_idx:size_idx].clamp(min=-1.0, max=1.0)
    pred_object_centroid = th.where(pred_object_centroid.isnan(), 0.0, pred_object_centroid)
    # logger.debug(f'pred_object_centroid: {pred_object_centroid.shape}')

    # pred_object_size = ((pred_object_bbox[:, :, size_idx:angle_idx] + 1) * 0.5).clamp(min=1e-4, max=1.0)
    pred_object_size = (pred_object_bbox[:, :, size_idx:angle_idx]).clamp(min=1e-4, max=1.0)
    pred_object_size = th.where(pred_object_size.isnan(), 0.0, pred_object_size)
    # logger.debug(f'pred_object_size: {pred_object_size.shape}')

    pred_object_cos_angle = pred_object_bbox[:, :, angle_idx:angle_idx + 1].clamp(min=-0.999999, max=0.999999)
    pred_object_sin_angle = pred_object_bbox[:, :, angle_idx + 1:angle_idx + 2].clamp(min=-0.999999, max=0.999999)
    pred_object_angle = th.where(
        th.abs(pred_object_cos_angle) < 5e-3, th.arcsin(pred_object_sin_angle), th.arccos(pred_object_cos_angle))
    # logger.debug(f'pred_object_angle: {pred_object_angle.shape}')

    # Bx13x7
    pred_object_bboxes = th.cat((pred_object_centroid, pred_object_size, pred_object_angle), dim=2)
    # logger.debug(f'pred_object_bboxes: {pred_object_bboxes.shape}')

    is_object_mask = (~no_object_mask).float()
    # logger.info(f'pred_object_bbox[no_object_mask].shape: {pred_object_bbox[no_object_mask].shape}')
    batch_pred_bbox_iou_loss_lst = []

    self_intersect_mask = th.eye(obj_feat_num, device=x_pred.device)
    self_intersect_mask = 1 - self_intersect_mask
    for batch_idx in range(B):
        # 13x7
        object_bbox_arr = pred_object_bboxes[batch_idx, ...]
        # 13x13
        iou_3d = bdb3d_iou(object_bbox_arr, object_bbox_arr)
        # ignore empty object
        iou_3d = is_object_mask[batch_idx, ...] * iou_3d
        # ignore iou between curtain and window
        if th.any(curtain_mask[batch_idx, ...]) and th.any(window_mask[batch_idx, ...]):
            curtain_window_mask = th.mm(curtain_mask[batch_idx, ...].float(), window_mask[batch_idx, ...].t().float())
            iou_3d = (1 - curtain_window_mask) * iou_3d

        # ignore iou between bed and pillow
        if th.any(bed_mask[batch_idx, ...]) and th.any(pillow_mask[batch_idx, ...]):
            bed_pillow_mask = th.mm(bed_mask[batch_idx, ...].float(), pillow_mask[batch_idx, ...].t().float())
            iou_3d = (1 - bed_pillow_mask) * iou_3d

        # ignore self-intersection
        iou_3d = self_intersect_mask * iou_3d
        # iou_3d /2, 1x169
        iou_loss_lst = (iou_3d * 0.5).contiguous().view(1, -1)
        batch_pred_bbox_iou_loss_lst.append(iou_loss_lst)

    # Bx1x169
    batch_pred_bbox_iou_loss = th.stack(batch_pred_bbox_iou_loss_lst, dim=0)
    iou_loss_shape = batch_pred_bbox_iou_loss.shape[-1]
    # logger.debug(f'batch_pred_bbox_iou_loss berfore weighting: {batch_pred_bbox_iou_loss}')
    batch_iou_loss = batch_pred_bbox_iou_loss * iou_loss_weights.reshape(B, 1, -1)[..., :iou_loss_shape]
    # logger.debug(f'batch_pred_bbox_iou_loss after weighting: {batch_pred_bbox_iou_loss}')

    return batch_iou_loss


def pred_3d_iou_loss(x_gt, y, means, weights):
    """
    Compute the 3D IoU of a Gaussian distribution of 3D objects.

    :param x_gt: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param y: the condition Tensor.
    :param weights: weights for each timestamp .
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    B, C, feat_size = x_gt.shape
    assert y.shape[0] == B

    x_pred = means
    pyhsical_violation_weight = 0.01
    # #  calculate object iou loss
    # batch_object_iou_loss = iou_among_predicted_3d_bbox(x_pred, y, weights)
    # calculate object-layout iou
    batch_layout_iou_loss = iou_among_layout_and_predicted_3d_bbox(x_pred, y, weights)
    # Bx169
    batch_iou_loss = batch_layout_iou_loss.sum(dim=1) * pyhsical_violation_weight

    return batch_iou_loss
