"""
Helpers for various likelihood-based losses. These are ported from the original
Ho et al. diffusion models codebase:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/utils.py
"""

import numpy as np

import torch as th

from dataset.st3d_dataset import get_room_type
from dataset.metadata import ST3D_BEDROOM_FURNITURE, ST3D_LIVINGROOM_FURNITURE, ST3D_DININGROOM_FURNITURE
from . import logger


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


def recover_predict_3d_bbox(x_pred, room_type_lst):
    phy_viol_weight = 0.001

    B, _, _ = x_pred.shape
    for batch_idx in range(B):
        room_type = room_type_lst[batch_idx]
        print(f'room_type value: {room_type}')
        room_type_str = get_room_type(room_type)
        if room_type_str == 'bedroom':
            obj_feat_num, obj_feat_dim = 13, 30
            class_labels_lst = (ST3D_BEDROOM_FURNITURE)
        elif room_type_str == 'living room':
            obj_feat_num, obj_feat_dim = 24, 32
            class_labels_lst = (ST3D_LIVINGROOM_FURNITURE)
        elif room_type_str == 'dining room':
            obj_feat_num, obj_feat_dim = 24, 32
            class_labels_lst = (ST3D_DININGROOM_FURNITURE)

    class_idx = 0
    centroid_idx = len(class_labels_lst)
    size_idx = 3 + centroid_idx
    angle_idx = 3 + size_idx
    obj_bbox_lst = x_pred[batch_idx, 0, :(obj_feat_num * obj_feat_dim)].reshape((obj_feat_num, obj_feat_dim))

    # set room layout bbox size as 5m x 5m x 5m
    room_layout_bbox_size = th.tensor([5.0, 5.0, 5.0], dtype=th.float32, device=x_pred.device)
    obj_bbox_dict_list = []
    for i in range(len(obj_bbox_lst)):
        # print(f'predict object bbox feature: {obj_bbox_lst[i]}')
        obj_bbox_dict = {}

        # recover class label
        class_label_prob = obj_bbox_lst[i][:centroid_idx]
        # print(f'class_label_prob: {class_label_prob}')
        class_label_prob = th.where(class_label_prob > 0.5, 1, 0)
        if th.all(class_label_prob == 0):
            print(f'object {i} has no class label')
        class_label = class_labels_lst[class_label_prob.argmax()]
        if class_label == 'empty':
            print(f'object {i} is empty')
            continue
        obj_bbox_dict['class'] = class_label

        # recover centroid
        centroid = obj_bbox_lst[i][centroid_idx:size_idx]
        centroid = centroid * room_layout_bbox_size
        obj_bbox_dict['center'] = centroid.tolist()
        # recover size
        size = obj_bbox_lst[i][size_idx:angle_idx]
        size = (size + 1) * 0.5
        size = size * room_layout_bbox_size
        obj_bbox_dict['size'] = size.tolist()
        # recover angle
        angle = obj_bbox_lst[i][angle_idx:]
        angle_0 = np.arccos(angle[0])
        # angle_1 = np.arcsin(angle[1])
        print(f' object {class_label} centroid: {centroid} size: {size} angle: {angle_0}')
        obj_bbox_dict['angles'] = [0, 0, angle_0]
        obj_bbox_dict_list.append(obj_bbox_dict)

        if (phy_viol_weight > 0):
            violation_loss, end_points = physical_violation_loss_cube(end_points)
        else:
            violation_loss = th.tensor(0)


def pred_3d_iou_loss(x, condition, means, log_scales):
    """
    Compute the 3D IoU of a Gaussian distribution of 3D objects.

    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    B, C, _ = x.shape
    assert condition.shape[0] == B
    x_start_gt = x[:, C - 1, :]
    # get predicted sample for x[0]
    x_start_pred = th.distributions.Normal(means, th.exp(log_scales))
    x_start_pred = x_start_pred.sample()
    # only use x[0][3,:] to calculate iou among objects
    x_start_pred = x_start_pred[:, C - 1, :]
    x_start_pred = recover_predict_3d_bbox(x_start_pred, condition)
    # calculate iou
