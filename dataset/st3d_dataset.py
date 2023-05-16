import os
import numpy as np
from PIL import Image
from shapely.geometry import LineString
from scipy.spatial.distance import cdist
from scipy.signal import correlate2d
from scipy.ndimage import shift
from typing import Any, List, Dict, Tuple

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

        # total image file names and text file names
        self.img_fnames = sorted(
            [fname for fname in os.listdir(self.img_dir) if fname.endswith('.jpg') or fname.endswith('.png')])
        self.txt_fnames = ['%s.txt' % fname[:-4] for fname in self.img_fnames]
        #  image file names and text file names on local_rank machine
        self.local_img_fnames = self.img_fnames[shard::num_shards]
        self.local_txt_fnames = self.txt_fnames[shard::num_shards]

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
        # img_path = os.path.join(self.img_dir, self.img_fnames[idx])
        img_path = os.path.join(self.img_dir, self.local_img_fnames[idx])
        img = np.array(Image.open(img_path), np.float32)[..., :3] / 255.
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

        # Convert all data to tensor
        # x = torch.FloatTensor(img.transpose([2, 0, 1]).copy())
        # boundary_lst = torch.FloatTensor(boundary_lst.copy())
        # corner_y_prob_lst = torch.FloatTensor(corner_y_prob_lst.copy())

        # Check whether additional output are requested
        # out_lst = [x, boundary_lst, corner_y_prob_lst]
        # if self.return_corners:
        #     out_lst = np.append(out_lst, corners_lst)
        # if self.return_path:
        #     out_lst = np.append(out_lst, img_path)
        #     out_lst.append(img_path)
        
        boundary_lst = boundary_lst.copy().astype(np.float32)
        corner_y_prob_lst = corner_y_prob_lst.copy().astype(np.float32)
        out_lst = np.append(boundary_lst, corner_y_prob_lst, axis=0)

        class_dict = {}
        if room_type is not None:
            class_dict["y"] = np.array(room_type, dtype=np.int64)

        # out_lst = out_lst[np.newaxis,:,:]
        return out_lst, class_dict

    def get_layout_mesh(self,
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

        # Convert corners to layout
        depth_img, floor_mask, ceil_mask, wall_mask = layout_2_depth(corners_lst,
                                                                     H,
                                                                     W,
                                                                     floor_height=cam_pos_lst[2],
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
        xyzrgb = np.concatenate([xs[..., None], ys[..., None], zs[..., None], equirect_img], -1)
        # convert points from camera frame to world frame
        xyzrgb[:, :, :3] = xyzrgb[:, :, :3] + cam_pos_lst
        xyzrgb = np.concatenate([xyzrgb, xyzrgb[:, [0]]], 1)
        print(f' mask: {mask.shape}')
        mask = np.concatenate([mask, mask[:, [0]]], 1)
        print(f'concatenated mask: {mask.shape}')
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

        return (points, faces, corners_lst, cam_pos_lst)
