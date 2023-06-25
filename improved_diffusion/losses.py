"""
Helpers for various likelihood-based losses. These are ported from the original
Ho et al. diffusion models codebase:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/utils.py
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import torch as th
from pytorch3d.ops import box3d_overlap

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


def iou_among_predicted_3d_bbox(x_pred, room_type_lst, iou_loss_weights):

    # only use x[0][3,:] to calculate iou among objects
    B, C, feat_size = x_pred.shape
    object_feat_idx = C - 1
    assert C == 4, "The input x_pred should be (B, 4, 1024)"
    assert iou_loss_weights.shape == (B, C, feat_size), "The loss weights tensor should be (B, 4, 1024)"

    logger.debug(f'iou_loss_weights: {iou_loss_weights.shape}')
    logger.debug(f'iou_loss_weights: {iou_loss_weights}')

    class_labels_lst = ST3D_BEDROOM_FURNITURE
    obj_feat_num, obj_feat_dim = 13, 30

    class_idx = 0
    centroid_idx = class_idx + len(class_labels_lst)
    size_idx = 3 + centroid_idx
    angle_idx = 3 + size_idx

    # Bx13x30
    pred_object_bbox = x_pred[:, object_feat_idx, :(obj_feat_num * obj_feat_dim)].reshape(B, obj_feat_num, obj_feat_dim)
    pred_object_class_prob = th.where(pred_object_bbox[:, :, :centroid_idx] > 0.5, 1, 0)
    logger.debug(f'pred_object_class_prob: {pred_object_class_prob.shape}')
    # skip probability of empty object
    no_object_mask = th.all(pred_object_class_prob == 0, dim=2, keepdim=True)
    pred_object_class = th.argmax(pred_object_class_prob, dim=2, keepdim=True)
    logger.debug(f'pred_object_class: {pred_object_class.shape}')
    # skip empty object
    no_object_mask = th.logical_or(no_object_mask,
                                   th.all(pred_object_class == class_labels_lst.index('empty'), dim=2, keepdim=True))
    # no_object_mask = no_object_mask.unsqueeze(2)
    logger.debug(f'no_object_mask: {no_object_mask.shape}')

    pred_object_centroid = pred_object_bbox[:, :, centroid_idx:size_idx].clamp(min=-1.0, max=1.0)
    pred_object_centroid = th.where(pred_object_centroid.isnan(), 0.0, pred_object_centroid)
    logger.debug(f'pred_object_centroid: {pred_object_centroid.shape}')

    pred_object_size = ((pred_object_bbox[:, :, size_idx:angle_idx] + 1) * 0.5).clamp(min=1e-4, max=1.0)
    pred_object_size = th.where(pred_object_size.isnan(), 0.0, pred_object_size)
    logger.debug(f'pred_object_size: {pred_object_size.shape}')

    pred_object_angle = th.clamp(pred_object_bbox[:, :, angle_idx], min=-0.999999, max=0.999999)
    pred_object_angle = th.arccos(pred_object_angle)
    pred_object_angle = pred_object_angle.unsqueeze(2)
    logger.debug(f'pred_object_angle: {pred_object_angle.shape}')

    # Bx13x7
    pred_object_bboxes = th.cat((pred_object_centroid, pred_object_size, pred_object_angle), dim=2)
    logger.debug(f'pred_object_bboxes: {pred_object_bboxes.shape}')

    # # Bx13x8x3
    # pred_object_bbox_corners = bbox_corners(pred_object_centroid, pred_object_size, pred_object_angle)
    # logger.info(f'pred_object_bbox_corners: {pred_object_bbox_corners.shape}')

    is_object_mask = (~no_object_mask).float()
    # logger.info(f'pred_object_bbox_corners[no_object_mask].shape: {pred_object_bbox_corners[is_object_mask].shape}')
    batch_pred_bbox_iou_loss_lst = []
    for batch_idx in range(B):
        # room_type = room_type_lst[batch_idx]
        # room_type_str = get_room_type(room_type)
        # if room_type_str == 'bedroom':
        #     obj_feat_num, obj_feat_dim = 13, 30
        #     class_labels_lst = (ST3D_BEDROOM_FURNITURE)
        # elif room_type_str == 'living room':
        #     obj_feat_num, obj_feat_dim = 24, 32
        #     class_labels_lst = (ST3D_LIVINGROOM_FURNITURE)
        # elif room_type_str == 'dining room':
        #     obj_feat_num, obj_feat_dim = 24, 32
        #     class_labels_lst = (ST3D_DININGROOM_FURNITURE)

        # each batch(room) has different number of objects
        # Assume inputs: boxes1 (M, 8, 3) and boxes2 (N, 8, 3)
        object_bbox_arr = pred_object_bboxes[batch_idx, ...]
        # logger.info(f'bbox_arr: {object_bbox_arr}')
        iou_3d = bdb3d_iou(object_bbox_arr, object_bbox_arr)
        # ignore empty object
        iou_3d = is_object_mask[batch_idx, ...] * iou_3d

        object_num = object_bbox_arr.shape[0]
        # logger.debug(f'object_num: {object_bbox_arr.shape[0]}')

        # ignore self-intersection
        mask = th.eye(object_num, device=x_pred.device).to(th.bool)
        iou_3d = (~mask).float() * iou_3d
        # iou_3d /2
        iou_3d = iou_3d.contiguous().view(-1)
        iou_loss_lst = th.zeros((C, feat_size), dtype=th.float32, device=x_pred.device)
        iou_loss_lst[object_feat_idx, :iou_3d.shape[0]] = iou_3d
        # iou_loss_lst = l1_critertion(iou_loss_lst, th.zeros_like(iou_loss_lst))
        logger.debug(f'iou_loss_lst: {iou_loss_lst.shape}')
        batch_pred_bbox_iou_loss_lst.append(iou_loss_lst)

    batch_pred_bbox_iou_loss = th.stack(batch_pred_bbox_iou_loss_lst, dim=0)
    batch_iou_loss = batch_pred_bbox_iou_loss * iou_loss_weights
    return batch_iou_loss

    # batch_pred_bbox_iou_loss_lst = []
    # for batch_idx in range(B):
    #     room_type = room_type_lst[batch_idx]
    #     # print(f'room_type value: {room_type}')
    #     room_type_str = get_room_type(room_type)
    #     if room_type_str == 'bedroom':
    #         obj_feat_num, obj_feat_dim = 13, 30
    #         class_labels_lst = (ST3D_BEDROOM_FURNITURE)
    #     elif room_type_str == 'living room':
    #         obj_feat_num, obj_feat_dim = 24, 32
    #         class_labels_lst = (ST3D_LIVINGROOM_FURNITURE)
    #     elif room_type_str == 'dining room':
    #         obj_feat_num, obj_feat_dim = 24, 32
    #         class_labels_lst = (ST3D_DININGROOM_FURNITURE)

    #     class_idx = 0
    #     centroid_idx = len(class_labels_lst)
    #     size_idx = 3 + centroid_idx
    #     angle_idx = 3 + size_idx
    #     pred_obj_bbox_lst = x_pred[batch_idx, object_feat_idx, :(obj_feat_num * obj_feat_dim)].reshape(
    #         (obj_feat_num, obj_feat_dim))

    #     # set room layout bbox size as 5m x 5m x 5m
    #     room_layout_bbox_size = th.tensor([1.0, 1.0, 1.0], dtype=th.float32, device=x_pred.device)
    #     obj_bbox_lst = []
    #     obj_bbox_idx_lst = []
    #     obj_bbox_cls_lst = []
    #     for i in range(len(pred_obj_bbox_lst)):
    #         # print(f'predict object bbox feature: {pred_obj_bbox_lst[i]}')
    #         obj_bbox_dict = {}

    #         # recover class label
    #         class_label_prob = pred_obj_bbox_lst[i][:centroid_idx]
    #         # print(f'class_label_prob: {class_label_prob}')
    #         class_label_prob = th.where(class_label_prob > 0.5, 1, 0)
    #         if th.all(class_label_prob == 0):
    #             logger.debug(f'object {i} has no class label')
    #             continue
    #         class_label = class_labels_lst[class_label_prob.argmax()]
    #         if class_label == 'empty':
    #             logger.debug(f'object {i} is empty')
    #             continue

    #         obj_bbox_idx_lst.append(i)
    #         obj_bbox_dict['class'] = class_label
    #         obj_bbox_cls_lst.append(class_label)

    #         # recover centroid
    #         centroid = pred_obj_bbox_lst[i][centroid_idx:size_idx]
    #         centroid = centroid * room_layout_bbox_size
    #         centroid = th.clamp(centroid, -1, 1)
    #         obj_bbox_dict['center'] = centroid
    #         # recover size
    #         size = pred_obj_bbox_lst[i][size_idx:angle_idx]
    #         size = (size + 1) * 0.5
    #         size = th.clamp(size, 1e-3, 1.0)
    #         size = size * room_layout_bbox_size
    #         obj_bbox_dict['size'] = size
    #         # recover angle
    #         angle = pred_obj_bbox_lst[i][angle_idx:]
    #         angle_0 = th.arccos(angle[0])
    #         angle_0 = th.where(th.isnan(angle_0), 0, angle_0)
    #         angles = th.tensor([0, 0, angle_0], dtype=th.float32, device=x_pred.device)
    #         obj_bbox_dict['angles'] = angles

    #         # print(f' object {class_label} centroid: {centroid} size: {size} angle: {angles}')

    #         bbox_3d = bdb3d_corners(obj_bbox_dict)
    #         obj_bbox_lst.append(bbox_3d)

    #     obj_bbox_lst = th.stack(obj_bbox_lst, dim=0)
    #     # logger.info(f'obj_bbox_lst: {obj_bbox_lst.shape}')

    #     # calculate iou between objects
    #     # bedroom: iou_loss Bx13x30
    #     iou_loss = th.zeros_like(pred_obj_bbox_lst, dtype=th.float32, device=x_pred.device)
    #     for i in range(len(obj_bbox_lst)):
    #         for j in range(i + 1, len(obj_bbox_lst)):
    #             # skip window and curtain iou loss
    #             if obj_bbox_cls_lst[i] in ['window', 'curtain'] and obj_bbox_cls_lst[j] in ['window', 'curtain']:
    #                 continue
    #             # iou = bdb3d_iou(obj_bbox_lst[i], obj_bbox_lst[j])
    #             logger.info(f'object {obj_bbox_lst[i]} and object {obj_bbox_lst[j]}')
    #             iou = box3d_overlap(obj_bbox_lst[i].unsqueeze(0), obj_bbox_lst[j].unsqueeze(0), eps=1e-6)[0][0]
    #             # if iou > 0.1:
    #             print(f'object {obj_bbox_cls_lst[i]} and object {obj_bbox_cls_lst[j]} has iou: {iou}')
    #             # only apply iou to the two objects
    #             object1_idx = obj_bbox_idx_lst[i]
    #             object2_idx = obj_bbox_idx_lst[j]
    #             iou_loss[object1_idx][centroid_idx:size_idx] += phy_viol_weight * iou
    #             iou_loss[object2_idx][centroid_idx:size_idx] += phy_viol_weight * iou
    #     # logger.info(f'iou_loss: {iou_loss}')
    #     # iou_loss_lst: Bx4x1024
    #     iou_loss_lst = th.zeros((C, feat_size), dtype=th.float32, device=x_pred.device)
    #     iou_loss_lst[object_feat_idx, :(obj_feat_num * obj_feat_dim)] = iou_loss.contiguous().view(-1)
    #     # logger.info(f'iou_loss_lst: {iou_loss_lst[object_feat_idx, :(obj_feat_num * obj_feat_dim)]}')
    #     batch_pred_bbox_iou_loss_lst.append(iou_loss_lst)

    # # batch_pred_bbox_iou_loss_lst: Bx4x1024
    # batch_pred_bbox_iou_loss_lst = th.stack(batch_pred_bbox_iou_loss_lst, dim=0)

    # # l1_critertion = th.nn.SmoothL1Loss(reduction='mean')
    # # mse_critertion = th.nn.MSELoss(reduction='mean')
    # # batch_iou_loss = mse_critertion(batch_pred_bbox_iou_loss_lst,
    # #                                 th.zeros_like(batch_pred_bbox_iou_loss_lst, device=x_pred.device))
    # batch_iou_loss = batch_pred_bbox_iou_loss_lst
    # return batch_iou_loss


def pred_3d_iou_loss(x_gt, y, means, log_scales, weights):
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
    pyhsical_violation_weight = 500
    #  calculate iou loss
    batch_iou_loss = iou_among_predicted_3d_bbox(x_pred, y, weights)
    batch_iou_loss = batch_iou_loss.sum(dim=1) * pyhsical_violation_weight
    logger.debug(f'batch_iou_loss: {batch_iou_loss.shape}')
    # batch_iou_loss = batch_iou_loss[:, batch_iou_loss > 0]
    # logger.info(f'batch_iou_loss > 0: {batch_iou_loss.shape}')

    return batch_iou_loss
