import os
import argparse

import json
import numpy as np
from tqdm import tqdm
import imghdr
import shutil
import cv2
import trimesh
import copy
import multiprocessing as mp
from collections import Counter, OrderedDict

import open3d as o3d
from panda3d.core import Triangulator

from typing import List, Tuple, Dict, Any, Union

from misc.utils import matrix_to_euler_angles, euler_angle_to_matrix
from misc.equirect_projection import vis_objs3d
from dataset.metadata import (INVALID_SCENES_LST, INVALID_ROOMS_LST, OBJECT_LABEL_IDS, COLOR_TO_LABEL,
                              ST3D_LIVINGROOM_MIN_LEN, ST3D_BEDROOM_MIN_LEN, ST3D_DININGROOM_MIN_LEN)
from dataset.metadata import ROOM_WALLS_LARGER_THAN_10

from dataset.st3d_dataset import get_mesh_from_corners, np_coorx2u, np_coory2v

from visualize_mesh import verify_normal, create_spatial_quad_polygen
from visualize_3d import convert_lines_to_vertices
'''
Assume datas is extracted by `misc/structured3d_extract_zip.py`.
That is to said, assuming following structure:
- {in_root}/scene_xxxxx
    - rgb/
        - *png
    - layout/
        - *txt


The reorganized structure as follow:
- {out_train_root}
    - img/             # rgb panoramas
        - scene_xxxxx_*png
    - label_cor/       # 2D layout coordinates
        - scene_xxxxx_*txt
    - quad_walls/   # 3D quad walls 
        - scene_xxxxx_*txt
    - cam_pos/          # camera position
        - scene_xxxxx_*txt
    - room_type/        # room type
        - scene_xxxxx_*txt
    - bbox_3d/          # 3D bbox of objects
        - scene_xxxxx_*json 

- {out_valid_root} ...
- {out_test_root} ...
'''
ALL_SCENE = ['scene_%05d' % i for i in range(0, 3500)]
TRAIN_SCENE = ['scene_%05d' % i for i in range(0, 3000)]
VALID_SCENE = ['scene_%05d' % i for i in range(3000, 3250)]
TEST_SCENE = ['scene_%05d' % i for i in range(3250, 3500)]

ST3D_BEDROOM_FURNITURES_SET = set()
ST3D_LIVINGROOM_FURNITURES_SET = set()
ST3D_DININGROOM_FURNITURES_SET = set()


class ST3DQuadRoomLayout(List):

    class QuadWall(object):

        def __init__(self, wall_corners: np.ndarray, wall_normal: np.ndarray, wall_center: np.ndarray, wall_sx: float,
                     wall_sy: float):
            """ Construct a wall as quad polygen

            Args:
                wall_corners (np.ndarray): 4 corners' coordinate of wall
                wall_normal (np.ndarray): normalized plane normal of wall
                wall_center (np.ndarray): center coordinate of wall
                wall_sx (float): wall size in x axis
                wall_sy (float): wall size in y axis
            """
            self.wall_corners = wall_corners
            self.wall_normal = wall_normal
            self.wall_center = wall_center
            self.wall_sx = wall_sx
            self.wall_sy = wall_sy

        @property
        def wall_corners(self):
            return self.wall_corners

        @property
        def wall_normal(self):
            return self.wall_normal

        @property
        def wall_center(self):
            return self.wall_center

        @property
        def wall_sx(self):
            return self.wall_sx

        @property
        def wall_sy(self):
            return self.wall_sy

    def __init__(self, quad_wall_lst: List[QuadWall]):
        super().__init__()
        self.quad_wall_lst = quad_wall_lst

    def __getitem__(self, id):
        return self.quad_wall_lst[id]

    def append(self, quadwall: QuadWall):
        return self.quad_wall_lst.append(quadwall)

    def __len__(self):
        return len(self.quad_wall_lst)

    def __iter__(self):
        return iter(self.quad_wall_lst)

    def __setitem__(self, id, quadwall: QuadWall):
        self.quad_wall_lst[id] = quadwall

    def __delitem__(self, id):
        del self.quad_wall_lst[id]


def vis_scene_mesh(room_layout_mesh: trimesh.Trimesh,
                   obj_bbox_lst: List[Dict],
                   room_layout_bbox=None) -> trimesh.Trimesh:

    def create_oriented_bbox(scene_bbox: List[Dict]) -> trimesh.Trimesh:
        """Export oriented (around Z axis) scene bbox to meshes
        Args:
            scene_bbox: (N x 7 numpy array): xyz pos of center and 3 lengths (dx,dy,dz)
                and heading angle around Z axis.
                Y forward, X right, Z upward. heading angle of positive X is 0,
                heading angle of positive Y is 90 degrees.
            out_filename: (string) filename
        """

        def heading2rotmat(heading_angle_rad):
            rotmat = np.eye(3)
            cosval = np.cos(heading_angle_rad)
            sinval = np.sin(heading_angle_rad)
            rotmat[0:2, 0:2] = np.array([[cosval, -sinval], [sinval, cosval]])
            return rotmat

        def convert_oriented_box_to_trimesh_fmt(box):
            box_center = box['center']
            box_lengths = box['size']
            transform_matrix = np.eye(4)
            transform_matrix[0:3, 3] = box_center
            # only use z angle, rad
            transform_matrix[0:3, 0:3] = heading2rotmat(box['angles'][-1])
            box_trimesh_fmt = trimesh.creation.box(box_lengths, transform_matrix)
            color = list(COLOR_TO_LABEL.keys())[list(COLOR_TO_LABEL.values()).index(box['class'])]
            box_trimesh_fmt.visual.face_colors = np.random.uniform(0, 1, (len(box_trimesh_fmt.faces), 3))
            # for facet in box_trimesh_fmt.facets:
            #     box_trimesh_fmt.visual.face_colors[facet] = [color[0], color[1], color[2], 255]
            return box_trimesh_fmt

        scene = trimesh.scene.Scene()
        for box in scene_bbox:
            scene.add_geometry(convert_oriented_box_to_trimesh_fmt(box))

        mesh_list = trimesh.util.concatenate(scene.dump())
        return mesh_list

    v_object_meshes = create_oriented_bbox(obj_bbox_lst)
    if room_layout_bbox is not None:
        scene_mesh = trimesh.util.concatenate([room_layout_mesh, v_object_meshes, room_layout_bbox])
    elif room_layout_mesh is not None:
        scene_mesh = trimesh.util.concatenate([room_layout_mesh, v_object_meshes])
    else:
        scene_mesh = trimesh.util.concatenate([v_object_meshes])
    return scene_mesh


def vis_color_pointcloud(rgb_img_filepath, depth_img_filepath, saved_color_pcl_filepath):

    def get_unit_spherical_map():
        h = 512
        w = 1024

        coorx, coory = np.meshgrid(np.arange(w), np.arange(h))
        us = np_coorx2u(coorx, w)
        vs = np_coory2v(coory, h)

        X = np.expand_dims(np.cos(vs) * np.sin(us), 2)
        Y = np.expand_dims(np.sin(vs), 2)
        Z = np.expand_dims(np.cos(vs) * np.cos(us), 2)
        unit_map = np.concatenate([X, Z, Y], axis=2)

        return unit_map

    def display_inlier_outlier(cloud, ind):
        inlier_cloud = cloud.select_by_index(ind)
        outlier_cloud = cloud.select_by_index(ind, invert=True)

        print("Showing outliers (red) and inliers (gray): ")
        outlier_cloud.paint_uniform_color([1, 0, 0])
        inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
        o3d.visualization.draw([inlier_cloud, outlier_cloud])

    assert os.path.exists(rgb_img_filepath), 'rgb panorama doesnt exist!!!'
    assert os.path.exists(depth_img_filepath), 'depth panorama doesnt exist!!!'

    raw_depth_img = cv2.imread(depth_img_filepath, cv2.IMREAD_UNCHANGED)
    if len(raw_depth_img.shape) == 3:
        raw_depth_img = cv2.cvtColor(raw_depth_img, cv2.COLOR_BGR2GRAY)
    depth_img = np.asarray(raw_depth_img)
    if np.isnan(depth_img.any()) or len(depth_img[depth_img > 0]) == 0:
        print('empyt depth image')
        exit(-1)

    raw_rgb_img = cv2.imread(rgb_img_filepath, cv2.IMREAD_UNCHANGED)
    rgb_img = cv2.cvtColor(raw_rgb_img, cv2.COLOR_BGR2RGB)
    if rgb_img.shape[2] == 4:
        rgb_img = rgb_img[:, :, :3]
    if np.isnan(rgb_img.any()) or len(rgb_img[rgb_img > 0]) == 0:
        print('empyt rgb image')
        exit(-1)
    color = np.clip(rgb_img, 0.0, 255.0) / 255.0
    # print(f'raw_rgb shape: {rgb_img.shape} color shape: {color.shape}, ')

    depth_img = np.expand_dims((depth_img / 1000.0), axis=2)
    pointcloud = depth_img * get_unit_spherical_map()

    o3d_pointcloud = o3d.geometry.PointCloud()
    o3d_pointcloud.points = o3d.utility.Vector3dVector(pointcloud.reshape(-1, 3))
    o3d_pointcloud.colors = o3d.utility.Vector3dVector(color.reshape(-1, 3))
    # remove outliers
    # cl, ind = o3d_pointcloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    # display_inlier_outlier(o3d_pointcloud, ind)
    o3d.io.write_point_cloud(saved_color_pcl_filepath, o3d_pointcloud)
    return o3d_pointcloud


def normalize_bbox_size(object_bbox_dict_lst: List[Dict]):
    """ normalize bbox size to [0, 1] """
    assert len(object_bbox_dict_lst) > 0, 'object_bbox_dict_lst is empty!!!'
    object_bbox_dict_lst = copy.deepcopy(object_bbox_dict_lst)
    object_bbox_dict_lst = sorted(object_bbox_dict_lst, key=lambda x: np.linalg.norm(x['size']))
    min_bbox_size = np.array(object_bbox_dict_lst[0]['size'])
    max_bbox_size = np.array(object_bbox_dict_lst[-1]['size'])
    for object_bbox_dict in object_bbox_dict_lst:
        object_bbox_dict['size'] = ((np.array(object_bbox_dict['size'] - min_bbox_size) / max_bbox_size) * 2 -
                                    1).tolist()
    return object_bbox_dict_lst


def parse_bbox_in_room(room_folderpath: str, room_layout_mesh, quad_walls_dict: Dict[str, List] = None):
    """ parse object bounding box in room

    Args:
        room_folderpath (str): room folder path
        room_layout_mesh (_type_): room layout mesh derived from 2d layout
        quad_walls_dict (Dict[str, List]): room layout as quad walls.

    Returns:
        _type_: _description_
    """

    room_bbox_3d_path = os.path.join(room_folderpath, 'full', 'bbox_3d.json')
    rgb_img_path = os.path.join(room_folderpath, 'full', 'rgb_rawlight.png')
    instance_img_path = os.path.join(room_folderpath, 'full', 'instance.png')
    camera_pos_path = os.path.join(room_folderpath, 'camera_xyz.txt')

    # parse room bbox
    with open(room_bbox_3d_path, 'r') as file:
        room_anno_3d_dict = json.load(file)

    id2index = dict()
    for index, object in enumerate(room_anno_3d_dict):
        id2index[object.get('ID')] = index

    rgb_img = cv2.imread(rgb_img_path, cv2.IMREAD_UNCHANGED)
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    instance_img = cv2.imread(instance_img_path, cv2.IMREAD_UNCHANGED)
    cam_position = np.loadtxt(camera_pos_path)

    layout_bbox_min = trimesh.bounds.corners(room_layout_mesh.bounding_box_oriented.bounds).min(axis=0)
    layout_bbox_max = trimesh.bounds.corners(room_layout_mesh.bounding_box_oriented.bounds).max(axis=0)
    layout_bbox_size = layout_bbox_max - layout_bbox_min
    layout_bbox = room_layout_mesh.bounding_box_oriented

    def check_bbox_in_room(bbox: Dict, room_layout_mesh: trimesh.Trimesh, layout_bbox_min: np.array,
                           layout_bbox_max: np.array):
        bbox_center = np.array([bbox['center']])
        margin_dist = 0.5
        if bbox_center[:, 0] < layout_bbox_min[0] or bbox_center[:, 0] > layout_bbox_max[0] or \
            bbox_center[:, 1] < layout_bbox_min[1] or bbox_center[:, 1] > layout_bbox_max[1] or \
            bbox_center[:, 2] < layout_bbox_min[2] or bbox_center[:, 2] > layout_bbox_max[2]:
            cloest_pts, distance, faces_idx = room_layout_mesh.nearest.on_surface(bbox_center)
            # print('%s distance %f to room ' % (bbox['class'], distance))
            if distance < margin_dist and bbox['class'] in ['door', 'window', 'picture', 'curtain']:
                return True
            return False
        else:
            return True

    # object bboxs in camera frame
    obj_bbox_lst = []
    # normalized object bboxs
    obj_bbox_normal_lst = []
    # skip background
    for index in np.unique(instance_img)[:-1]:
        # for each instance in current image
        # we remove some incorrect objeect labels manually
        if index not in id2index.keys():
            continue
        bbox = room_anno_3d_dict[id2index[index]]

        if bbox['label'] not in OBJECT_LABEL_IDS.keys():
            continue

        basis = np.array(bbox['basis'])
        coeffs = np.array(bbox['coeffs'])
        centroid = np.array(bbox['centroid'])

        obj_bbox_dict = {}
        obj_bbox_dict['rotations'] = basis.tolist()
        obj_bbox_dict['centroid'] = list(centroid)
        obj_bbox_dict['dimensions'] = list(coeffs)
        obj_bbox_dict['class'] = bbox['label']

        rotation_euler_angles_rad = matrix_to_euler_angles(basis)
        obj_bbox_dict['angles'] = rotation_euler_angles_rad.tolist()
        bbox_center = (centroid - cam_position) * 0.001
        obj_bbox_dict['center'] = bbox_center.tolist()
        bbox_size = coeffs * 0.001 * 2
        obj_bbox_dict['size'] = bbox_size.tolist()
        if check_bbox_in_room(obj_bbox_dict, room_layout_mesh, layout_bbox_min, layout_bbox_max):
            obj_bbox_lst.append(obj_bbox_dict)

            obj_bbox_normal_dict = {}
            obj_bbox_normal_dict['class'] = bbox['label']
            obj_bbox_normal_dict['angles'] = [
                np.cos(rotation_euler_angles_rad[2]),
                np.sin(rotation_euler_angles_rad[2])
            ]
            # here we normalize bbox center and size w.r.t. room_layout bbox
            # normalize bbox center to [-1, 1]
            obj_bbox_normal_dict['center'] = (bbox_center / layout_bbox_size).tolist()
            # normalize bbox size to [-1, 1]
            obj_bbox_normal_dict['size'] = ((bbox_size / layout_bbox_size) * 2 - 1).tolist()
            obj_bbox_normal_lst.append(obj_bbox_normal_dict)

    if len(obj_bbox_lst) < ST3D_LIVINGROOM_MIN_LEN:
        return None, None, None, None

    # normalize bbox size to [-1, 1]
    # obj_bbox_normal_lst = normalize_bbox_size(obj_bbox_normal_lst)

    # visualize quad wall if exists
    if quad_walls_dict is not None:
        wall_lst = []
        for quad_wall in quad_walls_dict['walls']:
            wall_dict = {}

            wall_dict['center'] = quad_wall['center']
            wall_dict['size'] = [quad_wall['width'], 0.05, quad_wall['height']]
            normal = quad_wall['normal']
            # print(f'wall normal: {normal}')
            # The direction of all camera is always along the negative y-axis.
            cos_angle = np.array(normal).dot(np.array([0, -1, 0]))
            angle = np.arccos(cos_angle)
            if abs(cos_angle) < 1e-6:
                angle = np.pi / 2 if normal[0] > 0 else -np.pi / 2

            wall_dict['angles'] = [0, 0, angle]
            rotation_matrix = euler_angle_to_matrix(wall_dict['angles'])
            recovered_normal = rotation_matrix.dot(np.array([0, -1, 0]))
            # print(f'recovered normal: {recovered_normal}')
            # print(f' recovered normal is {np.allclose(np.array(normal), recovered_normal, atol=1e-3)}')
            wall_dict['class'] = 'wall'
            wall_lst.append(wall_dict)
        obj_bbox_lst.extend(wall_lst)

    anno_img = vis_objs3d(image=rgb_img,
                          v_bbox3d=obj_bbox_lst,
                          camera_position=cam_position,
                          b_show_axes=False,
                          b_show_centroid=False,
                          b_show_bbox3d=True,
                          b_show_info=True,
                          thickness=2)

    scene_mesh = vis_scene_mesh(room_layout_mesh, obj_bbox_lst, room_layout_bbox=None)

    obj_bbox_dicts = {}
    obj_bbox_dicts['objects'] = obj_bbox_lst
    obj_bbox_normal_dicts = {}
    obj_bbox_normal_dicts['objects'] = obj_bbox_normal_lst
    return obj_bbox_dicts, obj_bbox_normal_dicts, anno_img, scene_mesh


def parse_room_layout(img_filepath: str, cam_pos_filepath: str, layout_coor_filepath: str) -> trimesh.Trimesh:
    # Read image
    equirect_img = cv2.imread(img_filepath, cv2.IMREAD_UNCHANGED)
    equirect_img = cv2.cvtColor(equirect_img, cv2.COLOR_BGR2RGB)
    if equirect_img.shape[2] == 4:
        equirect_img = equirect_img[:, :, :3]
    H, W = equirect_img.shape[:2]

    # read camera position file
    cam_pos_lst = []
    with open(cam_pos_filepath) as f:
        cam_pos_lst = np.array([line.strip().split() for line in f if line.strip()], np.float32)
    assert len(cam_pos_lst) == 1, cam_pos_filepath
    # convert the unit into meter
    cam_pos_lst = cam_pos_lst[0] * 0.001

    # Read ground truth corners
    corners_lst = []
    with open(layout_coor_filepath, 'r') as f:
        corners_lst = np.array([line.strip().split() for line in f if line.strip()], np.float32)

        # Corner with minimum x should at the beginning
        corners_lst = np.roll(corners_lst[:, :2], -2 * np.argmin(corners_lst[::2, 0]), 0)

    points, faces = get_mesh_from_corners(corners_lst,
                                          H,
                                          W,
                                          camera_position=cam_pos_lst,
                                          rgb_img=equirect_img,
                                          b_ignore_floor=False,
                                          b_ignore_ceiling=False,
                                          b_ignore_wall=False,
                                          b_in_world_frame=False)
    # print(f'points.shape: {points.shape}, faces.shape: {faces.shape}')
    # downsample the mesh
    raw_mesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(points[:, :3]),
                                         triangles=o3d.utility.Vector3iVector(faces))
    simplified_mesh = raw_mesh.simplify_vertex_clustering(voxel_size=0.2,
                                                          contraction=o3d.geometry.SimplificationContraction.Average)
    simplified_vertices = simplified_mesh.vertices
    simplified_faces = np.array(simplified_mesh.triangles)
    room_layout_mesh = trimesh.Trimesh(vertices=simplified_vertices, faces=simplified_faces, process=True)
    return room_layout_mesh


def create_layout_mesh(vertices, vertices_floor, delta_height, camera_center):
    # create mesh for 3D floorplan visualization
    triangles = []

    # the number of vertical walls
    num_walls = len(vertices)

    # 1. vertical wall (always rectangle)
    num_vertices = 0
    for i in range(len(vertices)):
        # hardcode triangles for each vertical wall
        triangle = np.array([[0, 2, 1], [2, 0, 3]])
        triangles.append(triangle + num_vertices)
        num_vertices += 4

    # 2. floor and ceiling
    # Since the floor and ceiling may not be a rectangle, triangulate the polygon first.
    tri = Triangulator()
    for i in range(len(vertices_floor)):
        tri.add_vertex(vertices_floor[i, 0], vertices_floor[i, 1])

    for i in range(len(vertices_floor)):
        tri.add_polygon_vertex(i)

    tri.triangulate()

    # polygon triangulation
    triangle = []
    for i in range(tri.getNumTriangles()):
        triangle.append([tri.get_triangle_v0(i), tri.get_triangle_v1(i), tri.get_triangle_v2(i)])
    triangle = np.array(triangle)

    # add triangles for floor and ceiling
    triangles.append(triangle + num_vertices)
    num_vertices += len(np.unique(triangle))
    triangles.append(triangle + num_vertices)

    # 3. Merge wall, floor, and ceiling
    vertices.append(vertices_floor)
    vertices.append(vertices_floor + delta_height)
    vertices = np.concatenate(vertices, axis=0)

    triangles = np.concatenate(triangles, axis=0)

    # 4. create mesh
    vertices_in_cam = (vertices - camera_center) * 0.001 if camera_center is not None else vertices * 0.001
    mesh = trimesh.Trimesh(vertices=vertices_in_cam, faces=triangles, process=False)
    mesh = mesh.subdivide_to_size(0.02, max_iter=1000)
    return mesh


def parse_wall_corners(scene_annos: dict, room_id: str, camera_position_filepath: str,
                       room_layout_mesh: trimesh.Trimesh):

    # read camera position file
    cam_pos_lst = []
    with open(camera_position_filepath) as f:
        cam_pos_lst = np.array([line.strip().split() for line in f if line.strip()], np.float32)
    assert len(cam_pos_lst) == 1, camera_position_filepath
    cam_position = cam_pos_lst[0]

    # parse corners
    junctions = np.array([item['coordinate'] for item in scene_annos['junctions']])
    lines_holes = []
    for semantic in scene_annos['semantics']:
        if semantic['type'] in ['window', 'door']:
            for planeID in semantic['planeID']:
                lines_holes.extend(np.where(np.array(scene_annos['planeLineMatrix'][planeID]))[0].tolist())

    lines_holes = np.unique(lines_holes)
    _, vertices_holes = np.where(np.array(scene_annos['lineJunctionMatrix'])[lines_holes])
    vertices_holes = np.unique(vertices_holes)

    # parse annotations
    walls = dict()
    walls_normal = dict()
    for semantic in scene_annos['semantics']:
        if semantic['ID'] != int(room_id):
            continue

        # find junctions of ceiling and floor
        for planeID in semantic['planeID']:
            plane_anno = scene_annos['planes'][planeID]

            if plane_anno['type'] != 'wall':
                lineIDs = np.where(np.array(scene_annos['planeLineMatrix'][planeID]))[0]
                lineIDs = np.setdiff1d(lineIDs, lines_holes)
                junction_pairs = [
                    np.where(np.array(scene_annos['lineJunctionMatrix'][lineID]))[0].tolist() for lineID in lineIDs
                ]
                wall = convert_lines_to_vertices(junction_pairs)
                walls[plane_anno['type']] = wall[0]

        # save normal of the vertical walls
        for planeID in semantic['planeID']:
            plane_anno = scene_annos['planes'][planeID]

            if plane_anno['type'] == 'wall':
                lineIDs = np.where(np.array(scene_annos['planeLineMatrix'][planeID]))[0]
                lineIDs = np.setdiff1d(lineIDs, lines_holes)
                junction_pairs = [
                    np.where(np.array(scene_annos['lineJunctionMatrix'][lineID]))[0].tolist() for lineID in lineIDs
                ]
                wall = convert_lines_to_vertices(junction_pairs)
                walls_normal[tuple(np.intersect1d(wall, walls['floor']))] = plane_anno['normal']

    # we assume that zs of floor equals 0, then the wall height is from the ceiling
    wall_height = np.mean(junctions[walls['ceiling']], axis=0)[-1]
    delta_height = np.array([0, 0, wall_height])

    # list of corner index
    wall_floor = walls['floor']

    # wall
    quad_wall_dict, quad_wall_normalized_dict = {}, {}
    quad_wall_lst, quad_wall_normalized_lst = [], []
    quad_corners_lst = []
    for i, j in zip(wall_floor, np.roll(wall_floor, shift=-1)):
        corner_i, corner_j = junctions[i], junctions[j]
        plane_normal = walls_normal[tuple(sorted([i, j]))]
        flip = verify_normal(corner_i, corner_j, delta_height, plane_normal)

        if flip:
            corner_j, corner_i = corner_i, corner_j

        # 3D coordinate for each wall
        quad_corners = np.array([corner_i, corner_i + delta_height, corner_j + delta_height, corner_j])
        # print(f'plane normal: {plane_normal}')

        wall_center = np.mean(quad_corners, axis=0)
        # wall center in camera frame, unit: meter
        wall_center_in_cam = (wall_center - cam_position) * 0.001
        wall_normal = np.array(plane_normal)
        # The direction of all camera is always along the positive y-axis.
        cos_angle = np.array(wall_normal).dot(np.array([0, -1, 0]))
        angle = np.arccos(cos_angle)
        if abs(cos_angle) < 1e-6:
            angle = np.pi / 2 if wall_normal[0] > 0 else -np.pi / 2
        wall_width = np.linalg.norm(corner_i - corner_j) * 0.001
        wall_height = np.linalg.norm(delta_height) * 0.001

        # wall = ST3DQuadRoomLayout.QuadWall(quad_corners, wall_center_in_cam, wall_normal, wall_width, wall_height)
        wall_dict = {}
        wall_dict['ID'] = len(quad_wall_lst)
        wall_dict['center'] = wall_center_in_cam.tolist()
        wall_dict['normal'] = wall_normal.tolist()
        wall_dict['angles'] = [np.cos(angle), np.sin(angle)]
        wall_dict['width'] = wall_width.tolist()
        wall_dict['height'] = wall_height.tolist()
        wall_dict['corners'] = (quad_corners * 0.001).tolist()
        quad_wall_lst.append(wall_dict)

    corner_floor = junctions[wall_floor]
    # room_layout_mesh = create_layout_mesh(quad_corners_lst, corner_floor, delta_height, cam_position)
    # compute the bounding box of the layout
    layout_bbox_min = trimesh.bounds.corners(room_layout_mesh.bounding_box_oriented.bounds).min(axis=0)
    layout_bbox_max = trimesh.bounds.corners(room_layout_mesh.bounding_box_oriented.bounds).max(axis=0)
    layout_bbox_size = layout_bbox_max - layout_bbox_min
    for wall_dict in quad_wall_lst:
        wall_center_in_cam = np.array(wall_dict['center'])
        wall_normal = np.array(wall_dict['normal'])
        wall_width = float(wall_dict['width'])
        wall_height = float(wall_dict['height'])
        wall_angles = wall_dict['angles']
        # if layout_bbox_size is not None:
        wall_normalized_dict = {}
        wall_normalized_dict['ID'] = wall_dict['ID']
        wall_normalized_dict['center'] = (wall_center_in_cam / layout_bbox_size).tolist()
        wall_normalized_dict['normal'] = wall_normal.tolist()
        wall_normalized_dict['angles'] = wall_angles
        wall_normalized_dict['width'] = (wall_width / max(layout_bbox_size[0], layout_bbox_size[1])).tolist()
        wall_normalized_dict['height'] = (wall_height / layout_bbox_size[2]).tolist()
        quad_wall_normalized_lst.append(wall_normalized_dict)

    quad_wall_dict['walls'] = quad_wall_lst
    quad_wall_normalized_dict['walls'] = quad_wall_normalized_lst
    return quad_wall_dict, quad_wall_normalized_dict, layout_bbox_size


def prepare_dataset(raw_dataset_dir, target_room_type, scene_ids, out_dir, b_save_debug_files=False):

    furniture_counts = []

    room_layout_size_lst = []
    for scene_id in tqdm(scene_ids):

        if scene_id in INVALID_SCENES_LST:
            continue

        room_type_lst = None
        # parse scene annotation
        scene_anno_3d_filepath = os.path.join(raw_dataset_dir, scene_id, 'annotation_3d.json')
        if not os.path.isfile(scene_anno_3d_filepath):
            INVALID_SCENES_LST.append(scene_id)
            continue
        else:
            scene_anno_3d_dict = json.load(open(scene_anno_3d_filepath, 'r'))
            room_type_lst = scene_anno_3d_dict['semantics']

        # print(f'Processing scene: {scene_id}')
        scene_dir = os.path.join(raw_dataset_dir, scene_id, '2D_rendering')
        for room_id in np.sort(os.listdir(scene_dir)):

            room_str = '%s_%s' % (scene_id, room_id)
            if room_str in INVALID_ROOMS_LST:
                continue

            room_type_str = 'undefined'
            if room_type_lst is not None:
                for rt in room_type_lst:
                    if rt['ID'] == int(room_id):
                        room_type_str = rt['type']
                        break
            if room_type_str != target_room_type:
                # print(f'room_type_str: {room_type_str}, target_room_type: {target_room_type}')
                continue

            room_path = os.path.join(scene_dir, room_id, "panorama")
            source_img_path = os.path.join(room_path, "full", "rgb_rawlight.png")
            source_cor_path = os.path.join(room_path, "layout.txt")
            source_cam_pos_path = os.path.join(room_path, "camera_xyz.txt")

            # parse room layout
            room_layout_mesh = parse_room_layout(source_img_path, source_cam_pos_path, source_cor_path)
            quad_walls_dict, quad_walls_normalized_dict, room_layout_size = parse_wall_corners(
                scene_anno_3d_dict, room_id, source_cam_pos_path, room_layout_mesh)
            # skip wall number < 4
            if len(quad_walls_dict['walls']) < 4:
                print(f'bad scene {room_str} walls number < 4')
                INVALID_ROOMS_LST.append(room_str)
                continue
            if len(quad_walls_normalized_dict['walls']) > 10:
                print(f'bad scene {room_str} walls number > 10')
                ROOM_WALLS_LARGER_THAN_10.append(room_str)
                continue

            # parse 3d bbox of objects in the room
            obj_bbox_3d_dict, obj_bbox_3d_normalized_dict, debug_bbox_img, debug_bbox_trimesh = parse_bbox_in_room(
                room_path, room_layout_mesh, quad_walls_dict)
            if obj_bbox_3d_dict is None:
                print(f'bad scene {room_str}')
                INVALID_ROOMS_LST.append(room_str)
                continue

            out_img_dir = os.path.join(out_dir, room_type_str, 'img')
            out_cord_dir = os.path.join(out_dir, room_type_str, 'label_cor')
            out_cam_pos_dir = os.path.join(out_dir, room_type_str, 'cam_pos')
            out_room_type_dir = os.path.join(out_dir, room_type_str, 'room_type')
            out_bbox_3d_dir = os.path.join(out_dir, room_type_str, 'bbox_3d')
            out_quad_wall_dir = os.path.join(out_dir, room_type_str, 'quad_walls')
            os.makedirs(out_img_dir, exist_ok=True)
            os.makedirs(out_cord_dir, exist_ok=True)
            os.makedirs(out_cam_pos_dir, exist_ok=True)
            os.makedirs(out_room_type_dir, exist_ok=True)
            os.makedirs(out_bbox_3d_dir, exist_ok=True)
            os.makedirs(out_quad_wall_dir, exist_ok=True)
            target_img_path = os.path.join(out_img_dir, '%s_%s.png' % (scene_id, room_id))
            target_cor_path = os.path.join(out_cord_dir, '%s_%s.txt' % (scene_id, room_id))
            target_cam_pos_path = os.path.join(out_cam_pos_dir, '%s_%s.txt' % (scene_id, room_id))
            target_room_type_path = os.path.join(out_room_type_dir, '%s_%s.txt' % (scene_id, room_id))
            target_bbox_3d_path = os.path.join(out_bbox_3d_dir, '%s_%s.json' % (scene_id, room_id))
            target_bbox_3d_normal_path = os.path.join(out_bbox_3d_dir, '%s_%s_normalized.json' % (scene_id, room_id))
            target_bbox_3d_vis_path = os.path.join(out_bbox_3d_dir, '%s_%s.png' % (scene_id, room_id))
            target_bbox_3d_mesh_path = os.path.join(out_bbox_3d_dir, '%s_%s.ply' % (scene_id, room_id))
            target_quad_wall_path = os.path.join(out_quad_wall_dir, '%s_%s.json' % (scene_id, room_id))
            target_quad_wall_normalized_path = os.path.join(out_quad_wall_dir,
                                                            '%s_%s_normalized.json' % (scene_id, room_id))

            # skip rooms without bed
            if target_room_type == 'bedroom':
                room_furniture_types = set([box['class'] for box in obj_bbox_3d_dict['objects']])
                if 'bed' not in room_furniture_types:
                    INVALID_ROOMS_LST.append(room_str)
                    print(f'bad scene {room_str} without bed')
                    continue

            # skip rooms with corrupted files
            if not os.path.isfile(source_img_path) or not os.path.isfile(source_cor_path) \
            or not os.path.isfile(source_cam_pos_path) or imghdr.what(source_img_path) is None:
                INVALID_ROOMS_LST.append(room_str)
                print(f'bad scene {room_str} with corrupted files')
                continue
            else:
                shutil.copyfile(source_img_path, target_img_path)
                shutil.copyfile(source_cor_path, target_cor_path)
                shutil.copyfile(source_cam_pos_path, target_cam_pos_path)
                # write room type
                with open(target_room_type_path, 'w') as f:
                    f.write(room_type_str)
                # write 3d bbox
                with open(target_bbox_3d_path, 'w') as f:
                    json.dump(obj_bbox_3d_dict, f, indent=4)
                # write normalized 3d bbox
                with open(target_bbox_3d_normal_path, 'w') as f:
                    json.dump(obj_bbox_3d_normalized_dict, f, indent=4)
                # write quad walls
                with open(target_quad_wall_path, 'w') as f:
                    json.dump(quad_walls_dict, f, indent=4)
                # write normalized quad walls
                with open(target_quad_wall_normalized_path, 'w') as f:
                    json.dump(quad_walls_normalized_dict, f, indent=4)

                if b_save_debug_files:
                    # visualize debug bbox img
                    cv2.imwrite(target_bbox_3d_vis_path, debug_bbox_img)
                    # print(f'save visualization for object bbox annotation of {room_id_str}')
                    debug_bbox_trimesh.export(target_bbox_3d_mesh_path)

            # furniture statistics
            if target_room_type == 'bedroom':
                room_furniture_types = set([box['class'] for box in obj_bbox_3d_dict['objects']])
                ST3D_BEDROOM_FURNITURES_SET.update(room_furniture_types)
            elif target_room_type == 'living room':
                room_furniture_types = set([box['class'] for box in obj_bbox_3d_dict['objects']])
                ST3D_LIVINGROOM_FURNITURES_SET.update(room_furniture_types)
            elif target_room_type == 'dining room':
                room_furniture_types = set([box['class'] for box in obj_bbox_3d_dict['objects']])
                ST3D_DININGROOM_FURNITURES_SET.update(room_furniture_types)

            room_layout_size_lst.append(room_layout_size)
            furniture_counts.append([box['class'] for box in obj_bbox_3d_dict['objects']])

    furniture_counts = Counter(sum(furniture_counts, []))
    furniture_counts = OrderedDict(sorted(furniture_counts.items(), key=lambda x: -x[1]))
    print(f"furniture_counts: \n {furniture_counts}")
    print(f'mean room_layout size : {np.mean(np.array(room_layout_size_lst), axis=0)}')


def parse_args():
    parser = argparse.ArgumentParser(description="Structured3D 2D Layout Visualization")
    parser.add_argument("--dataset_path",
                        default="/data/dataset/Structured3D/Structured3D/",
                        help="raw dataset path",
                        metavar="DIR")
    parser.add_argument("--room_type",
                        default="st3d_livingroom",
                        choices=["st3d_bedroom", "st3d_livingroom", "st3d_diningroom", "st3d_study"],
                        help="structured3d room type")
    parser.add_argument('--out_all_path', default='/mnt/nas_3dv/hdd1/datasets/Structured3d/preprocessed/all_raw_light')
    parser.add_argument('--out_train_path',
                        default='/mnt/nas_3dv/hdd1/datasets/Structured3d/preprocessed/st3d_train_full_raw_light')
    parser.add_argument('--out_valid_path',
                        default='/mnt/nas_3dv/hdd1/datasets/Structured3d/preprocessed/st3d_valid_full_raw_light')
    parser.add_argument('--out_test_path',
                        default='/mnt/nas_3dv/hdd1/datasets/Structured3d/preprocessed/st3d_test_full_raw_light')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.room_type == 'st3d_bedroom':
        room_type_str = 'bedroom'
    elif args.room_type == 'st3d_livingroom':
        room_type_str = 'living room'
    elif args.room_type == 'st3d_diningroom':
        room_type_str = 'dining room'
    elif args.room_type == 'st3d_study':
        room_type_str = 'study'

    # args_lst = [(args.dataset_path, room_type_str, scene_id, args.out_all_path) for scene_id in ALL_SCENE]
    # with mp.Pool() as pool:
    #     pool.imap(prepare_dataset, args_lst)

    prepare_dataset(args.dataset_path, room_type_str, ALL_SCENE, args.out_all_path, b_save_debug_files=True)
    # prepare_dataset(args.dataset_path, TRAIN_SCENE, args.out_train_path)
    # prepare_dataset(args.dataset_path, VALID_SCENE, args.out_valid_path)
    # prepare_dataset(args.dataset_path, TEST_SCENE, args.out_test_path)
    print('*' * 20 + ' invalid rooms ids: ' + '*' * 20)
    print(INVALID_ROOMS_LST)

    print('*' * 20 + ' room walls  size > 10: ' + '*' * 20)
    print(ROOM_WALLS_LARGER_THAN_10)

    if args.room_type == 'st3d_bedroom':
        print('*' * 20 + ' bedroom furniture types: ' + '*' * 20)
        print(ST3D_BEDROOM_FURNITURES_SET)
    elif args.room_type == 'st3d_livingroom':
        print('*' * 20 + ' livingroom furniture types: ' + '*' * 20)
        print(ST3D_LIVINGROOM_FURNITURES_SET)
    elif args.room_type == 'st3d_diningroom':
        print('*' * 20 + ' st3d_diningroom furniture types: ' + '*' * 20)
        print(ST3D_DININGROOM_FURNITURES_SET)


if __name__ == "__main__":
    main()
