import os
import sys
import numpy as np
from PIL import Image
from shapely.geometry import LineString
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.signal import correlate2d
from scipy.ndimage import shift, map_coordinates
from scipy.ndimage.filters import maximum_filter
from shapely.geometry import Polygon

from typing import Any, List, Dict, Tuple
# import imghdr
import json

import torch
import torch.utils.data as data
from . import panostretch

# room types
ROOM_TYPE_DICT = {'living room': 0, 'kitchen':1, 'bedroom':2, 'bathroom':3, 'balcony':4, 'corridor':5,
                  'dining room':6, 'study':7, 'studio':8, 'store room':9, 'garden':10, 'laundry room':11,
                    'office':12, 'basement':13, 'garage':14, 'undefined':15}
# ROOM_CLASS_LST = [10, 8, 3, 2, 0, 4, 5, 14, 13, 12, 7, 9, 11, 1, 6, 15]

def find_occlusion(coor):
    # equirectangular coordinates to sperical image coordinates
    img_x = coor[:, 0]
    img_y = coor[:, 1]
    u = panostretch.coorx2u(img_x)
    v = panostretch.coory2v(img_y)
    # spherical camera coordinates, assume z=-50
    x, y = panostretch.uv2xy(u, v, z=-50)
    occlusion = []
    for i in range(len(x)):
        raycast = LineString([(0, 0), (x[i], y[i])])
        other_layout = []
        for j in range(i + 1, len(x)):
            other_layout.append((x[j], y[j]))
        for j in range(0, i):
            other_layout.append((x[j], y[j]))
        other_layout = LineString(other_layout)
        occlusion.append(raycast.intersects(other_layout))
    return np.array(occlusion)


def sort_xy_filter_unique(xs, ys, y_small_first=True):
    xs, ys = np.array(xs), np.array(ys)
    idx_sort = np.argsort(xs + ys / ys.max() * (int(y_small_first) * 2 - 1))
    xs, ys = xs[idx_sort], ys[idx_sort]
    _, idx_unique = np.unique(xs, return_index=True)
    xs, ys = xs[idx_unique], ys[idx_unique]
    assert np.all(np.diff(xs) > 0)
    return xs, ys


def corners_to_1d_boundary(corner_lst, H, W):
    bound_ceil_x_lst, bound_ceil_y_lst = [], []
    bound_floor_x_lst, bound_floor_y_lst = [], []
    num_corners = len(corner_lst)

    # connect ceiling-wall boundary
    for i in range(num_corners // 2):
        xys = panostretch.pano_connect_points(corner_lst[i * 2], corner_lst[(i * 2 + 2) % num_corners], z=-50, w=W, h=H)
        bound_ceil_x_lst.extend(xys[:, 0])
        bound_ceil_y_lst.extend(xys[:, 1])

    # connect floor-wall boundary
    for i in range(num_corners // 2):
        xys = panostretch.pano_connect_points(corner_lst[i * 2 + 1],
                                              corner_lst[(i * 2 + 3) % num_corners],
                                              z=50,
                                              w=W,
                                              h=H)
        bound_floor_x_lst.extend(xys[:, 0])
        bound_floor_y_lst.extend(xys[:, 1])
    bound_ceil_x_lst, bound_ceil_y_lst = sort_xy_filter_unique(bound_ceil_x_lst, bound_ceil_y_lst, y_small_first=True)
    bound_floor_x_lst, bound_floor_y_lst = sort_xy_filter_unique(bound_floor_x_lst,
                                                                 bound_floor_y_lst,
                                                                 y_small_first=False)
    # ceiling boundary and floor boundary
    boundary_lst = np.zeros((2, W))
    boundary_lst[0] = np.interp(x=np.arange(W), xp=bound_ceil_x_lst, fp=bound_ceil_y_lst, period=W)
    boundary_lst[1] = np.interp(x=np.arange(W), xp=bound_floor_x_lst, fp=bound_floor_y_lst, period=W)
    # scale to [-pi/2, pi/2]
    boundary_lst = ((boundary_lst + 0.5) / H - 0.5) * np.pi
    return boundary_lst


def layout_2_depth(cor_id, h, w, floor_height=1.6, return_mask=False):
    # Convert corners to per-column boundary first
    # Up -pi/2,  Down pi/2
    ceiling_bound_lst, floor_bound_lst = corners_to_1d_boundary(cor_id, h, w)
    ceiling_bound_lst = ceiling_bound_lst[None, :]  # [1, w]
    floor_bound_lst = floor_bound_lst[None, :]  # [1, w]
    assert (ceiling_bound_lst > 0).sum() == 0
    assert (floor_bound_lst < 0).sum() == 0

    # Per-pixel v coordinate (vertical angle)
    v_lst = ((np.arange(h) + 0.5) / h - 0.5) * np.pi
    v_lst = np.repeat(v_lst[:, None], w, axis=1)  # [h, w]

    # Floor-plane to depth
    # floor_h = 1.6
    floor_d = np.abs(floor_height / np.sin(v_lst))

    # wall to camera distance on horizontal plane at cross camera center
    cs = floor_height / np.tan(floor_bound_lst)

    # Ceiling-plane to depth
    ceil_height = np.abs(cs * np.tan(ceiling_bound_lst))  # [1, w]
    ceil_d = np.abs(ceil_height / np.sin(v_lst))  # [h, w]

    # Wall to depth
    wall_d = np.abs(cs / np.cos(v_lst))  # [h, w]

    # Recover layout depth
    floor_mask = (v_lst > floor_bound_lst)
    ceil_mask = (v_lst < ceiling_bound_lst)
    wall_mask = (~floor_mask) & (~ceil_mask)
    depth = np.zeros([h, w], np.float32)  # [h, w]
    depth[floor_mask] = floor_d[floor_mask]
    depth[ceil_mask] = ceil_d[ceil_mask]
    depth[wall_mask] = wall_d[wall_mask]

    assert (depth == 0).sum() == 0
    if return_mask:
        return depth, floor_mask, ceil_mask, wall_mask
    return depth


def np_coorx2u(coorx, coorW=1024):
    return ((coorx + 0.5) / coorW - 0.5) * 2 * np.pi


def np_coory2v(coory, coorH=512):
    return -((coory + 0.5) / coorH - 0.5) * np.pi


def np_coor2xy(coor, z=50, coorW=1024, coorH=512, floorW=1024, floorH=512):
    '''
    coor: N x 2, index of array in (col, row) format
    '''
    coor = np.array(coor)
    u = np_coorx2u(coor[:, 0], coorW)
    v = np_coory2v(coor[:, 1], coorH)
    c = z / np.tan(v)
    x = c * np.sin(u) + floorW / 2 - 0.5
    y = -c * np.cos(u) + floorH / 2 - 0.5
    return np.hstack([x[:, None], y[:, None]])

def np_x_u_solve_y(x, u, floorW=1024, floorH=512):
    c = (x - floorW / 2 + 0.5) / np.sin(u)
    return -c * np.cos(u) + floorH / 2 - 0.5


def np_y_u_solve_x(y, u, floorW=1024, floorH=512):
    c = -(y - floorH / 2 + 0.5) / np.cos(u)
    return c * np.sin(u) + floorW / 2 - 0.5

def np_xy2coor(xy, z=50, coorW=1024, coorH=512, floorW=1024, floorH=512):
    '''
    xy: N x 2
    '''
    x = xy[:, 0] - floorW / 2 + 0.5
    y = xy[:, 1] - floorH / 2 + 0.5

    u = np.arctan2(x, -y)
    v = np.arctan(z / np.sqrt(x**2 + y**2))

    coorx = (u / (2 * np.pi) + 0.5) * coorW - 0.5
    coory = (-v / np.pi + 0.5) * coorH - 0.5

    return np.hstack([coorx[:, None], coory[:, None]])

def vote(vec, tol):
    vec = np.sort(vec)
    n = np.arange(len(vec))[::-1]
    n = n[:, None] - n[None, :] + 1.0
    l = squareform(pdist(vec[:, None], 'minkowski', p=1) + 1e-9)

    invalid = (n < len(vec) * 0.4) | (l > tol)
    if (~invalid).sum() == 0 or len(vec) < tol:
        best_fit = np.median(vec)
        p_score = 0
    else:
        l[invalid] = 1e5
        n[invalid] = -1
        score = n
        max_idx = score.argmax()
        max_row = max_idx // len(vec)
        max_col = max_idx % len(vec)
        assert max_col > max_row
        best_fit = vec[max_row:max_col+1].mean()
        p_score = (max_col - max_row + 1) / len(vec)

    l1_score = np.abs(vec - best_fit).mean()

    return best_fit, p_score, l1_score

def get_gpid(coorx, coorW):
    gpid = np.zeros(coorW)
    gpid[np.round(coorx).astype(int)] = 1
    gpid = np.cumsum(gpid).astype(int)
    gpid[gpid == gpid[-1]] = 0
    return gpid

def gen_ww_cuboid(xy, gpid, tol):
    xy_cor = []
    assert len(np.unique(gpid)) == 4

    # For each part seperated by wall-wall peak, voting for a wall
    for j in range(4):
        now_x = xy[gpid == j, 0]
        now_y = xy[gpid == j, 1]
        new_x, x_score, x_l1 = vote(now_x, tol)
        new_y, y_score, y_l1 = vote(now_y, tol)
        if (x_score, -x_l1) > (y_score, -y_l1):
            xy_cor.append({'type': 0, 'val': new_x, 'score': x_score})
        else:
            xy_cor.append({'type': 1, 'val': new_y, 'score': y_score})

    # Sanity fallback
    scores = [0, 0]
    for j in range(4):
        if xy_cor[j]['type'] == 0:
            scores[j % 2] += xy_cor[j]['score']
        else:
            scores[j % 2] -= xy_cor[j]['score']
    if scores[0] > scores[1]:
        xy_cor[0]['type'] = 0
        xy_cor[1]['type'] = 1
        xy_cor[2]['type'] = 0
        xy_cor[3]['type'] = 1
    else:
        xy_cor[0]['type'] = 1
        xy_cor[1]['type'] = 0
        xy_cor[2]['type'] = 1
        xy_cor[3]['type'] = 0

    return xy_cor


def gen_ww_general(init_coorx, xy, gpid, tol):
    xy_cor = []
    assert len(init_coorx) == len(np.unique(gpid))

    # Candidate for each part seperated by wall-wall boundary
    for j in range(len(init_coorx)):
        now_x = xy[gpid == j, 0]
        now_y = xy[gpid == j, 1]
        # print(f'now_x: {now_x}')
        # print(f'now_y: {now_y}')
        new_x, x_score, x_l1 = vote(now_x, tol)
        new_y, y_score, y_l1 = vote(now_y, tol)
        u0 = np_coorx2u(init_coorx[(j - 1 + len(init_coorx)) % len(init_coorx)])
        u1 = np_coorx2u(init_coorx[j])
        if (x_score, -x_l1) > (y_score, -y_l1):
            xy_cor.append({'type': 0, 'val': new_x, 'score': x_score, 'action': 'ori', 'gpid': j, 'u0': u0, 'u1': u1, 'tbd': True})
        else:
            xy_cor.append({'type': 1, 'val': new_y, 'score': y_score, 'action': 'ori', 'gpid': j, 'u0': u0, 'u1': u1, 'tbd': True})

    # Construct wall from highest score to lowest
    while True:
        # Finding undetermined wall with highest score
        tbd = -1
        for i in range(len(xy_cor)):
            if xy_cor[i]['tbd'] and (tbd == -1 or xy_cor[i]['score'] > xy_cor[tbd]['score']):
                tbd = i
        if tbd == -1:
            break

        # This wall is determined
        xy_cor[tbd]['tbd'] = False
        p_idx = (tbd - 1 + len(xy_cor)) % len(xy_cor)
        n_idx = (tbd + 1) % len(xy_cor)

        num_tbd_neighbor = xy_cor[p_idx]['tbd'] + xy_cor[n_idx]['tbd']

        # Two adjacency walls are not determined yet => not special case
        if num_tbd_neighbor == 2:
            continue

        # Only one of adjacency two walls is determine => add now or later special case
        if num_tbd_neighbor == 1:
            if (not xy_cor[p_idx]['tbd'] and xy_cor[p_idx]['type'] == xy_cor[tbd]['type']) or\
                    (not xy_cor[n_idx]['tbd'] and xy_cor[n_idx]['type'] == xy_cor[tbd]['type']):
                # Current wall is different from one determined adjacency wall
                if xy_cor[tbd]['score'] >= -1:
                    # Later special case, add current to tbd
                    xy_cor[tbd]['tbd'] = True
                    xy_cor[tbd]['score'] -= 100
                else:
                    # Fallback: forced change the current wall or infinite loop
                    if not xy_cor[p_idx]['tbd']:
                        insert_at = tbd
                        if xy_cor[p_idx]['type'] == 0:
                            new_val = np_x_u_solve_y(xy_cor[p_idx]['val'], xy_cor[p_idx]['u1'])
                            new_type = 1
                        else:
                            new_val = np_y_u_solve_x(xy_cor[p_idx]['val'], xy_cor[p_idx]['u1'])
                            new_type = 0
                    else:
                        insert_at = n_idx
                        if xy_cor[n_idx]['type'] == 0:
                            new_val = np_x_u_solve_y(xy_cor[n_idx]['val'], xy_cor[n_idx]['u0'])
                            new_type = 1
                        else:
                            new_val = np_y_u_solve_x(xy_cor[n_idx]['val'], xy_cor[n_idx]['u0'])
                            new_type = 0
                    new_add = {'type': new_type, 'val': new_val, 'score': 0, 'action': 'forced infer', 'gpid': -1, 'u0': -1, 'u1': -1, 'tbd': False}
                    xy_cor.insert(insert_at, new_add)
            continue

        # Below checking special case
        if xy_cor[p_idx]['type'] == xy_cor[n_idx]['type']:
            # Two adjacency walls are same type, current wall should be differen type
            if xy_cor[tbd]['type'] == xy_cor[p_idx]['type']:
                # Fallback: three walls with same type => forced change the middle wall
                xy_cor[tbd]['type'] = (xy_cor[tbd]['type'] + 1) % 2
                xy_cor[tbd]['action'] = 'forced change'
                xy_cor[tbd]['val'] = xy[gpid == xy_cor[tbd]['gpid'], xy_cor[tbd]['type']].mean()
        else:
            # Two adjacency walls are different type => add one
            tp0 = xy_cor[n_idx]['type']
            tp1 = xy_cor[p_idx]['type']
            if xy_cor[p_idx]['type'] == 0:
                val0 = np_x_u_solve_y(xy_cor[p_idx]['val'], xy_cor[p_idx]['u1'])
                val1 = np_y_u_solve_x(xy_cor[n_idx]['val'], xy_cor[n_idx]['u0'])
            else:
                val0 = np_y_u_solve_x(xy_cor[p_idx]['val'], xy_cor[p_idx]['u1'])
                val1 = np_x_u_solve_y(xy_cor[n_idx]['val'], xy_cor[n_idx]['u0'])
            new_add = [
                {'type': tp0, 'val': val0, 'score': 0, 'action': 'forced infer', 'gpid': -1, 'u0': -1, 'u1': -1, 'tbd': False},
                {'type': tp1, 'val': val1, 'score': 0, 'action': 'forced infer', 'gpid': -1, 'u0': -1, 'u1': -1, 'tbd': False},
            ]
            xy_cor = xy_cor[:tbd] + new_add + xy_cor[tbd+1:]

    return xy_cor

def gen_ww(init_coorx_lst:np.array, coory_ceil_lst: np.array, z_ceil=50, coorW=1024, coorH=512, floorW=1024, floorH=512, tol=3, force_cuboid=True):
    """generate wall-wall from corner_x pixel coordinates and ceiling pixel coordinates

    Args:
        init_coorx_lst (np.array): corners' x pixel coordinates from prediction, 1d
        coory_ceil_lst (np.array): ceiling boundary y pixel coordinates from prediction, 1d
        z_ceil (int, optional): _description_. Defaults to 50.
        coorW (int, optional): _description_. Defaults to 1024.
        coorH (int, optional): _description_. Defaults to 512.
        floorW (int, optional): _description_. Defaults to 1024.
        floorH (int, optional): _description_. Defaults to 512.
        tol (int, optional): _description_. Defaults to 3.
        force_cuboid (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    # get corners' indices in width axis
    gpid = get_gpid(init_coorx_lst, coorW)
    print(f'gpid: {gpid}')
    # get ceiling pixel coordinates
    coor_ceil_2d_lst = np.hstack([np.arange(coorW)[:, None], coory_ceil_lst[:, None]])
    # pixel coordinates to unit sphere coordinates
    xy = np_coor2xy(coor_ceil_2d_lst, z_ceil, coorW, coorH, floorW, floorH)
    print(f'xy: {xy}')

    # Generate wall-wall
    if force_cuboid:
        xy_cor = gen_ww_cuboid(xy, gpid, tol)
    else:
        xy_cor = gen_ww_general(init_coorx_lst, xy, gpid, tol)

    # Ceiling view to normal view
    cor = []
    for j in range(len(xy_cor)):
        next_j = (j + 1) % len(xy_cor)
        if xy_cor[j]['type'] == 1:
            cor.append((xy_cor[next_j]['val'], xy_cor[j]['val']))
        else:
            cor.append((xy_cor[j]['val'], xy_cor[next_j]['val']))
    cor = np_xy2coor(np.array(cor), z_ceil, coorW, coorH, floorW, floorH)
    cor = np.roll(cor, -2 * cor[::2, 0].argmin(), axis=0)

    return cor, xy_cor

def infer_coory(coory0, h, z0=50, coorH=512):
    v0 = np_coory2v(coory0, coorH)
    c0 = z0 / np.tan(v0)
    z1 = z0 + h
    v1 = np.arctan2(z1, c0)
    return (-v1 / np.pi + 0.5) * coorH - 0.5


def find_N_peaks(signal, filter_size=29, min_v=0.05, N=None):
    """find N peaks in signal vector

    Args:
        signal (np.array): input vector
        filter_size (int, optional): _description_. Defaults to 29.
        min_v (float, optional): _description_. Defaults to 0.05.
        N (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    max_v = maximum_filter(signal, size=filter_size, mode='wrap')
    pk_loc = np.where(max_v == signal)[0]
    pk_loc = pk_loc[signal[pk_loc] > min_v]
    if N is not None:
        order = np.argsort(-signal[pk_loc])
        pk_loc = pk_loc[order[:N]]
        pk_loc = pk_loc[np.argsort(pk_loc)]
    return pk_loc, signal[pk_loc]

def mean_percentile(vec, p1=25, p2=75):
    vmin = np.percentile(a=vec, q=p1)
    vmax = np.percentile(a=vec, q=p2)
    return vec[(vmin <= vec) & (vec <= vmax)].mean()

def refine_boundary_by_fix_floor(coor_y_ceil, coor_y_floor, z_floor=50, coorH=512):
    '''
    Refine coor_y_ceil by coor_y_floor
    coor_y_floor are assumed on given height z_floor
    '''
    v_ceil = np_coory2v(coor_y_ceil, coorH)
    v_floor = np_coory2v(coor_y_floor, coorH)

    c0 = z_floor / np.tan(v_ceil)
    z1 = c0 * np.tan(v_floor)
    z1_mean = mean_percentile(z1)
    v1_refine = np.arctan2(z1_mean, c0)
    coory1_refine = (-v1_refine / np.pi + 0.5) * coorH - 0.5
    return coory1_refine, z1_mean

    # c_floor = z_floor / np.tan(v_floor)
    # z_ceil = c_floor * np.tan(v_ceil)
    # z_ceil_mean = mean_percentile(z_ceil)
    # v_ceil_refine = np.arctan2(z_ceil_mean, c_floor)
    # coor_y_ceil_refined = (-v_ceil_refine / np.pi + 0.5) * coorH - 0.5
    # return coor_y_ceil_refined, z_ceil_mean

def get_mesh_from_corners(corners_lst: np.ndarray, H: int, W: int, camera_position: np.array,
                            rgb_img: np.array, b_ignore_floor: bool=False, b_ignore_ceiling: bool=True,
                            b_ignore_wall: bool=False,
                            b_in_world_frame: bool=True) -> Tuple:
    """ generate layout mesh from equirectangular image and corners

    Args:
        corners_lst (np.ndarray): 2d corners in equirectangular image
        H (int): _description_
        W (int): _description_
        camera_position (np.array): _description_
        rgb_img (np.array): _description_
        b_ignore_floor (bool, optional): _description_. Defaults to False.
        b_ignore_ceiling (bool, optional): _description_. Defaults to True.
        b_ignore_wall (bool, optional): _description_. Defaults to False.
        b_in_world_frame (bool, optional): generate mesh in world frame or camera frame. Defaults to False.

    Returns:
        Tuple: _description_
    """

    # Convert corners to layout
    depth_img, floor_mask, ceil_mask, wall_mask = layout_2_depth(corners_lst,
                                                                    H,
                                                                    W,
                                                                    floor_height=camera_position[2],
                                                                    return_mask=True)
    coorx, coory = np.meshgrid(np.arange(W), np.arange(H))
    us = np_coorx2u(coorx, W)
    vs = np_coory2v(coory, H)
    zs = depth_img * np.sin(vs)
    cs = depth_img * np.cos(vs)
    xs = cs * np.sin(us)
    # we align y axis to the panorama image,
    # if we need to flip the y axis, the ys need to be flipped
    ys = cs * np.cos(us)

    # Aggregate mask
    mask = np.ones_like(floor_mask)
    if b_ignore_floor:
        mask &= ~floor_mask
    if b_ignore_ceiling:
        mask &= ~ceil_mask
    if b_ignore_wall:
        mask &= ~wall_mask

    # Prepare ply's points and faces
    xyzrgb = np.concatenate([xs[..., None], ys[..., None], zs[..., None], rgb_img], -1)
    # convert points from camera frame to world frame
    if b_in_world_frame:
        xyzrgb[:, :, :3] = xyzrgb[:, :, :3] + camera_position

    xyzrgb = np.concatenate([xyzrgb, xyzrgb[:, [0]]], 1)
    # print(f' mask: {mask.shape}')
    mask = np.concatenate([mask, mask[:, [0]]], 1)
    # print(f'concatenated mask: {mask.shape}')
    lo_tri_template = np.array([[0, 0, 0], [0, 1, 0], [0, 1, 1]])
    up_tri_template = np.array([[0, 0, 0], [0, 1, 1], [0, 0, 1]])
    ma_tri_template = np.array([[0, 0, 0], [0, 1, 1], [0, 1, 0]])
    lo_mask = (correlate2d(mask, lo_tri_template, mode='same') == 3)
    up_mask = (correlate2d(mask, up_tri_template, mode='same') == 3)
    ma_mask = (correlate2d(mask, ma_tri_template, mode='same') == 3) & (~lo_mask) & (~up_mask)
    ref_mask = (
        lo_mask | (correlate2d(lo_mask, np.flip(lo_tri_template, (0,1)), mode='same') > 0) |\
        up_mask | (correlate2d(up_mask, np.flip(up_tri_template, (0,1)), mode='same') > 0) |\
        ma_mask | (correlate2d(ma_mask, np.flip(ma_tri_template, (0,1)), mode='same') > 0)
    )
    points = xyzrgb[ref_mask]

    ref_id = np.full(ref_mask.shape, -1, np.int32)
    ref_id[ref_mask] = np.arange(ref_mask.sum())
    faces_lo_tri = np.stack([
        ref_id[lo_mask],
        ref_id[shift(lo_mask, [1, 0], cval=False, order=0)],
        ref_id[shift(lo_mask, [1, 1], cval=False, order=0)],
    ], 1)
    faces_up_tri = np.stack([
        ref_id[up_mask],
        ref_id[shift(up_mask, [1, 1], cval=False, order=0)],
        ref_id[shift(up_mask, [0, 1], cval=False, order=0)],
    ], 1)
    faces_ma_tri = np.stack([
        ref_id[ma_mask],
        ref_id[shift(ma_mask, [1, 0], cval=False, order=0)],
        ref_id[shift(ma_mask, [0, 1], cval=False, order=0)],
    ], 1)
    faces = np.concatenate([faces_lo_tri, faces_up_tri, faces_ma_tri])

    return (points, faces)

def get_simple_mesh_from_corners(corners_lst: np.ndarray, H: int, W: int, camera_position: np.array,
                            rgb_img: np.array, b_ignore_floor: bool=False, b_ignore_ceiling: bool=True,
                            b_ignore_wall: bool=False,
                            b_in_world_frame: bool=True) -> Tuple:
    """ generate layout mesh from equirectangular image and corners

    Args:
        corners_lst (np.ndarray): 2d corners in equirectangular image
        H (int): _description_
        W (int): _description_
        camera_position (np.array): _description_
        rgb_img (np.array): _description_
        b_ignore_floor (bool, optional): _description_. Defaults to False.
        b_ignore_ceiling (bool, optional): _description_. Defaults to True.
        b_ignore_wall (bool, optional): _description_. Defaults to False.
        b_in_world_frame (bool, optional): generate mesh in world frame or camera frame. Defaults to False.

    Returns:
        Tuple: _description_
    """

    # Convert corners to layout
    depth_img, floor_mask, ceil_mask, wall_mask = layout_2_depth(corners_lst,
                                                                    H,
                                                                    W,
                                                                    floor_height=camera_position[2],
                                                                    return_mask=True)
    coorx, coory = np.meshgrid(np.arange(W), np.arange(H))
    us = np_coorx2u(coorx, W)
    vs = np_coory2v(coory, H)
    zs = depth_img * np.sin(vs)
    cs = depth_img * np.cos(vs)
    xs = cs * np.sin(us)
    # we align y axis to the panorama image,
    # if we need to flip the y axis, the ys need to be flipped
    ys = cs * np.cos(us)

    print(f'xs: {xs.shape}, ys: {ys.shape}, zs: {zs.shape}')
    corners_lst = corners_lst.astype(np.int32).reshape(-1, 2)
    print(f'corners_lst: {corners_lst.shape}')
    corners_xyz = np.concatenate([xs[corners_lst[:, 1], corners_lst[:, 0], None], ys[corners_lst[:, 1], corners_lst[:, 0], None], 
                                  zs[corners_lst[:, 1], corners_lst[:, 0], None]], -1)
    return corners_xyz


class PanoCorBoundDataset(data.Dataset):
    '''
    dataset for layout: PanoCoordinatesBoundary
    '''

    def __init__(
            self,
            root_dir,
            flip=False,
            rotate=False,
            gamma=False,
            p_base=0.96,
            max_stretch=2.0,
            normcor=False,
            return_corners=False,
            return_path=False,
            #  support parallel training
            shard=0,
            num_shards=1):
        self.img_dir = os.path.join(root_dir, 'img')
        self.cor_dir = os.path.join(root_dir, 'label_cor')
        self.cam_pos_dir = os.path.join(root_dir, 'cam_pos')
        self.room_type_dir = os.path.join(root_dir, 'room_type')
        # object bbox folder
        self.bbox_3d_dir = os.path.join(root_dir, 'bbox_3d')

        # total image file names and text file names
        self.img_fnames = sorted(
            [fname for fname in os.listdir(self.img_dir) if fname.endswith('.jpg') or fname.endswith('.png')])
        self.txt_fnames = ['%s.txt' % fname[:-4] for fname in self.img_fnames]
        self.json_fnames = ['%s.json' % fname[:-4] for fname in self.img_fnames]
        #  image file names and text file names on local_rank machine
        self.local_img_fnames = self.img_fnames[shard::num_shards]
        self.local_txt_fnames = self.txt_fnames[shard::num_shards]
        self.local_json_fnames = self.json_fnames[shard::num_shards]

        self.flip = flip
        self.rotate = rotate
        self.gamma = gamma
        self.p_base = p_base
        self.max_stretch = max_stretch
        self.normcor = normcor
        self.return_corners = return_corners
        self.return_path = return_path
        self.local_classes = None

        # The direction of all camera is always along the negative y-axis.
        self.cam_R = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], np.float32)

        self._check_dataset()

    def _check_dataset(self):
        for fname in self.txt_fnames:
            assert os.path.isfile(os.path.join(self.cor_dir, fname)), '%s not found' % os.path.join(self.cor_dir, fname)

    def __len__(self):
        # return len(self.img_fnames)
        return len(self.local_img_fnames)

    def __getitem__(self, idx: int) -> List:
        """retrieve layout data

        Args:
            idx (int): panorama/room idx

        Returns:
            List: [Image, [boundary_x:1x1024, boundary_y:1x1024], boundary_wall_probability:1x-024]
        """
        # Read image
        img_path = os.path.join(self.img_dir, self.local_img_fnames[idx])
        img = Image.open(img_path).convert('RGB')
        img = np.array(img, np.float32) / 255.
        H, W = img.shape[:2]

        # read camera position file
        cam_pos_lst = []
        cam_pos_filepath = os.path.join(self.cam_pos_dir, self.local_txt_fnames[idx])
        with open(cam_pos_filepath) as f:
            cam_pos_lst = np.array([line.strip().split() for line in f if line.strip()], np.float32)
        assert len(cam_pos_lst) == 1, cam_pos_filepath

        # read room type file
        room_type = None
        room_type_filepath = os.path.join(self.room_type_dir, self.local_txt_fnames[idx])
        with open(room_type_filepath) as f:
            room_type = f.readline().strip()
            assert room_type in ROOM_TYPE_DICT.keys(), room_type_filepath
            room_type = ROOM_TYPE_DICT[room_type]

        # read object bbox file
        object_bbox_filepath = os.path.join(self.bbox_3d_dir, self.local_json_fnames[idx])
        object_bbox_lst = []
        with open(object_bbox_filepath) as f:
            object_bbox_dicts = json.load(f)
            object_bbox_dicts = object_bbox_dicts['objects']
        for obj_bbox in object_bbox_dicts:
            bbox_class_label = obj_bbox['name'].lower()
            bbox_centroid = np.array(obj_bbox['centroid'], np.float32)
            bbox_size = np.array(obj_bbox['size'], np.float32)
            # only use Z angle
            bbox_angle = np.array(obj_bbox['angles'], np.float32)[-1]
            object_bbox_lst.append([bbox_class_label, bbox_centroid, bbox_size, bbox_angle])

        # Read ground truth corners
        with open(os.path.join(self.cor_dir, self.local_txt_fnames[idx])) as f:
            corners_lst = np.array([line.strip().split() for line in f if line.strip()], np.float32)

            # Corner with minimum x should at the beginning
            corners_lst = np.roll(corners_lst[:, :2], -2 * np.argmin(corners_lst[::2, 0]), 0)

            # Detect occlusion
            occlusion = find_occlusion(corners_lst[::2].copy()).repeat(2)
            # corners correspondenses' x coordinate should be identical
            assert (np.abs(corners_lst[0::2, 0] - corners_lst[1::2, 0]) > W / 100).sum() == 0, img_path
            # corners correspondenses' y coordinate should be y_floor < y_ceiling
            assert (corners_lst[0::2, 1] > corners_lst[1::2, 1]).sum() == 0, img_path

        # Prepare 1d ceiling-wall/floor-wall boundary
        boundary_lst = corners_to_1d_boundary(corners_lst, H, W)

        # Random flip
        if self.flip and np.random.randint(2) == 0:
            img = np.flip(img, axis=1)
            boundary_lst = np.flip(boundary_lst, axis=1)
            corners_lst[:, 0] = img.shape[1] - 1 - corners_lst[:, 0]

        # Random horizontal rotate
        if self.rotate:
            delta_x = np.random.randint(img.shape[1])
            img = np.roll(img, delta_x, axis=1)
            boundary_lst = np.roll(boundary_lst, delta_x, axis=1)
            corners_lst[:, 0] = (corners_lst[:, 0] + delta_x) % img.shape[1]

        # Random gamma augmentation
        if self.gamma:
            p = np.random.uniform(1, 2)
            if np.random.randint(2) == 0:
                p = 1 / p
            img = img**p

        # Prepare 1d wall-wall probability
        corner_x_lst = corners_lst[~occlusion, 0]
        dist_o = cdist(corner_x_lst.reshape(-1, 1), np.arange(W).reshape(-1, 1), metric='euclidean', p=1)
        dist_r = cdist(corner_x_lst.reshape(-1, 1), np.arange(W).reshape(-1, 1) + W, metric='euclidean', p=1)
        dist_l = cdist(corner_x_lst.reshape(-1, 1), np.arange(W).reshape(-1, 1) - W, metric='euclidean', p=1)
        dist = np.min([dist_o, dist_r, dist_l], axis=0)
        nearest_dist = dist.min(0)
        corner_y_prob_lst = (self.p_base**nearest_dist).reshape(1, -1)

        # # Convert all data to tensor
        # x = torch.FloatTensor(img.transpose([2, 0, 1]).copy())
        # boundary_lst = torch.FloatTensor(boundary_lst.copy())
        # corner_y_prob_lst = torch.FloatTensor(corner_y_prob_lst.copy())

        # # Check whether additional output are requested
        # out_lst = [x, boundary_lst, corner_y_prob_lst]
        # if self.return_corners:
        #     out_lst = np.append(out_lst, corners_lst)
        # if self.return_path:
        #     out_lst = np.append(out_lst, img_path)
        # return out_lst
        # normalize to [-1, 1]
        boundary_lst = boundary_lst.copy().astype(np.float32)
        boundary_lst = boundary_lst /(0.5*np.pi)
        # normalize to [-1, 1]
        corner_y_prob_lst = corner_y_prob_lst.copy().astype(np.float32)
        corner_y_prob_lst = corner_y_prob_lst * 2 -1
        out_lst = np.append(boundary_lst, corner_y_prob_lst, axis=0)

        class_dict = {}
        if room_type is not None:
            class_dict["y"] = np.array(room_type, dtype=np.int64)
        return out_lst, class_dict
     
    def get_gt_layout_mesh(self,
                        idx: int,
                        b_ignore_floor: bool = False,
                        b_ignore_ceiling: bool = True,
                        b_ignore_wall: bool = False) -> Tuple:
        # Read image
        img_path = os.path.join(self.img_dir, self.img_fnames[idx])
        equirect_img = np.array(Image.open(img_path))
        if equirect_img.shape[2] == 4:
            equirect_img = equirect_img[:, :, :3]
        print(f'equirect_img.shape: {equirect_img.shape}')
        H, W = equirect_img.shape[:2]

        # read camera position file
        cam_pos_lst = []
        cam_pos_filepath = os.path.join(self.cam_pos_dir, self.txt_fnames[idx])
        with open(cam_pos_filepath) as f:
            cam_pos_lst = np.array([line.strip().split() for line in f if line.strip()], np.float32)
        assert len(cam_pos_lst) == 1, cam_pos_filepath
        # convert the unit into meter
        cam_pos_lst = cam_pos_lst[0] * 0.001
        print(f'cam_pos_lst: {cam_pos_lst}')

        # Read ground truth corners
        corners_lst = []
        with open(os.path.join(self.cor_dir, self.txt_fnames[idx])) as f:
            corners_lst = np.array([line.strip().split() for line in f if line.strip()], np.float32)

            # Corner with minimum x should at the beginning
            corners_lst = np.roll(corners_lst[:, :2], -2 * np.argmin(corners_lst[::2, 0]), 0)

        # Prepare 1d wall-wall probability
        # unoccluded_corner_lst = corners_lst[~occlusion]
        # corner_x_lst = corners_lst[~occlusion, 0]
        # dist_o = cdist(corner_x_lst.reshape(-1, 1), np.arange(W).reshape(-1, 1), metric='euclidean', p=1)
        # dist_r = cdist(corner_x_lst.reshape(-1, 1), np.arange(W).reshape(-1, 1) + W, metric='euclidean', p=1)
        # dist_l = cdist(corner_x_lst.reshape(-1, 1), np.arange(W).reshape(-1, 1) - W, metric='euclidean', p=1)
        # dist = np.min([dist_o, dist_r, dist_l], axis=0)
        # nearest_dist = dist.min(0)
        # corner_y_prob_lst = (self.p_base**nearest_dist).reshape(1, -1)

        points, faces= get_mesh_from_corners(corners_lst, H, W, camera_position=cam_pos_lst, rgb_img=equirect_img,
                                    b_ignore_floor=b_ignore_floor, b_ignore_ceiling=b_ignore_ceiling, b_ignore_wall=b_ignore_wall)

        return (points, faces, corners_lst, cam_pos_lst)

    def get_layout_mesh_from_prediction(self, bound_ceil_floor_lst:np.array, wall_prob_lst:np.array, b_force_raw:bool=False, 
                                        b_force_cuboid:bool=False) -> Tuple:
        # random choose a camera position
        # idx = np.random.randint(len(self.local_img_fnames))
        idx = 2
        # Read image
        img_path = os.path.join(self.img_dir, self.local_img_fnames[idx])
        equirect_img = np.array(Image.open(img_path))
        if equirect_img.shape[2] == 4:
            equirect_img = equirect_img[:, :, :3]
        # print(f'equirect_img.shape: {equirect_img.shape}')
        H, W = equirect_img.shape[:2]

        # read camera position file
        cam_pos_lst = []
        cam_pos_filepath = os.path.join(self.cam_pos_dir, self.local_txt_fnames[idx])
        with open(cam_pos_filepath) as f:
            cam_pos_lst = np.array([line.strip().split() for line in f if line.strip()], np.float32)
        assert len(cam_pos_lst) == 1, cam_pos_filepath
        # convert the unit into meter
        cam_pos_lst = cam_pos_lst[0] * 0.001
        # print(f'cam_pos_lst: {cam_pos_lst}')

        # convert uv coords to pixel coords
        y_boundary_lst = (bound_ceil_floor_lst / np.pi + 0.5) * H - 0.5
        y_boundary_lst[0] = np.clip(y_boundary_lst[0], 1, H/2-1)
        y_boundary_lst[1] = np.clip(y_boundary_lst[1], H/2+1, H-2)
        corner_prob_lst = wall_prob_lst

        # Init floor height
        z_ceil = 50
        # z_floor = - cam_pos_lst[2]
        # calculate floor height
        _, z_floor = refine_boundary_by_fix_floor(*y_boundary_lst, z_ceil)
        print(f'z_floor: {z_floor}')

        if b_force_raw:
            # Do not run post-processing, export raw polygon (1024*2 vertices) instead.
            # [TODO] Current post-processing lead to bad results on complex layout.
            # celing pixel coords
            cor = np.stack([np.arange(1024), y_boundary_lst[0]], 1)

        else:
            # Detech wall-wall peaks
            min_prob = 0 if b_force_cuboid else 0.05
            filter_size = int(round(W * 0.05 / 2))
            wall_num = 4 if b_force_cuboid else None
            # get corners' x coords from wall probablities
            corner_x_lst, _ = find_N_peaks(corner_prob_lst, filter_size=filter_size, min_v=min_prob, N=wall_num)
            print(f'corner_x_lst: {corner_x_lst.shape}')

            # Generate wall-walls
            cor, xy_cor = gen_ww(corner_x_lst, y_boundary_lst[0], z_ceil, tol=abs(0.16 * z_floor / 1.6), force_cuboid=b_force_cuboid)
            if not b_force_cuboid:
                # Check valid (for fear self-intersection)
                xy2d = np.zeros((len(xy_cor), 2), np.float32)
                for i in range(len(xy_cor)):
                    xy2d[i, xy_cor[i]['type']] = xy_cor[i]['val']
                    xy2d[i, xy_cor[i-1]['type']] = xy_cor[i-1]['val']
                if not Polygon(xy2d).is_valid:
                    print(
                        'Fail to generate valid general layout!! '
                        'Generate cuboid as default.', file=sys.stderr)
                    corner_x_lst, _ = find_N_peaks(corner_prob_lst, filter_size=filter_size, min_v=0, N=4)
                    cor, xy_cor = gen_ww(corner_x_lst, y_boundary_lst[0], z_ceil, tol=abs(0.16 * z_floor / 1.6), force_cuboid=True)

        # Expand ceiling pixel coords with floor
        coord_floor = infer_coory(cor[:, 1], z_floor - z_ceil, z_ceil)[:, None]
        # pixel coords cor: [x, y_ceil, y_floor]
        cor = np.hstack([cor, coord_floor])

        # Collect corner coords in equirectangular image
        corners_lst = np.zeros((len(cor)*2, 2), np.float32)
        for j in range(len(cor)):
            corners_lst[j*2] = cor[j, 0], cor[j, 1]
            corners_lst[j*2 + 1] = cor[j, 0], cor[j, 2]
        print(f'corners_lst: {corners_lst.shape}')
        # equirect_img = np.random.randint(0, 255, size=(H, W, 3), dtype=np.uint8)
        points, faces= get_mesh_from_corners(corners_lst, H, W, camera_position=cam_pos_lst, rgb_img=equirect_img,
                                    b_ignore_floor=False, b_ignore_ceiling=True, b_ignore_wall=False)
        return (points, faces, corners_lst, cam_pos_lst)
        