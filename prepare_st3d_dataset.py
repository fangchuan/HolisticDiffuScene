import os
import argparse

import json
import numpy as np
from tqdm import tqdm
import imghdr
import shutil
import cv2
import trimesh
import open3d as o3d

from typing import List, Tuple, Dict, Any, Union

from misc.utils import matrix_to_euler_angles
from misc.equirect_projection import vis_objs3d
from dataset.metadata import INVALID_SCENES_LST, INVALID_ROOMS_LST, OBJECT_LABEL_IDS
from dataset.st3d_dataset import get_mesh_from_corners
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
    - img/
        - scene_xxxxx_*png
    - label_cor/
        - scene_xxxxx_*txt
    - cam_pos/
        - scene_xxxxx_*txt
    - room_type/
        - scene_xxxxx_*txt
    - bbox_3d/
        - scene_xxxxx_*json 

- {out_valid_root} ...
- {out_test_root} ...
'''
ALL_SCENE = ['scene_%05d' % i for i in range(0, 3500)]
TRAIN_SCENE = ['scene_%05d' % i for i in range(0, 3000)]
VALID_SCENE = ['scene_%05d' % i for i in range(3000, 3250)]
TEST_SCENE = ['scene_%05d' % i for i in range(3250, 3500)]

def vis_scene_mesh(room_layout_mesh:trimesh.Trimesh, obj_bbox_lst:List[Dict], room_layout_bbox=None) -> trimesh.Trimesh:
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
            return box_trimesh_fmt.subdivide()

        scene = trimesh.scene.Scene()
        for box in scene_bbox:
            scene.add_geometry(convert_oriented_box_to_trimesh_fmt(box))

        mesh_list = trimesh.util.concatenate(scene.dump())
        return mesh_list

    v_object_meshes = create_oriented_bbox(obj_bbox_lst)
    if room_layout_bbox is not None:
        scene_mesh = trimesh.util.concatenate([room_layout_mesh, v_object_meshes, room_layout_bbox])
    else:
        scene_mesh = trimesh.util.concatenate([room_layout_mesh, v_object_meshes])
    return scene_mesh

    
def parse_bbox_in_room(room_folderpath:str, room_layout_mesh):
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
    layout_bbox = room_layout_mesh.bounding_box_oriented
    
    def check_bbox_in_room(bbox:Dict, room_layout_mesh:trimesh.Trimesh, layout_bbox_min:np.array, layout_bbox_max:np.array):
        bbox_center = np.array([bbox['center']])
        margin_dist = 0.5
        if bbox_center[:, 0] < layout_bbox_min[0] or bbox_center[:, 0] > layout_bbox_max[0] or \
            bbox_center[:, 1] < layout_bbox_min[1] or bbox_center[:, 1] > layout_bbox_max[1] or \
            bbox_center[:, 2] < layout_bbox_min[2] or bbox_center[:, 2] > layout_bbox_max[2]:
            cloest_pts, distance, faces_idx = room_layout_mesh.nearest.on_surface(bbox_center)
            # print('%s distance %f to room ' % (bbox['name'], distance) )
            if distance < margin_dist and bbox['name'] in ['door', 'window', 'picture', 'curtain']:
                return True
            return False
        else:
            return True

    obj_bbox_lst = []
    # skip background
    for index in np.unique(instance_img)[:-1]:
        # for each instance in current image
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
        obj_bbox_dict['name'] = bbox['label']

        obj_bbox_dict['angles'] = matrix_to_euler_angles(basis).tolist()
        bbox_center = (centroid - cam_position) * 0.001
        obj_bbox_dict['center'] = bbox_center.tolist()
        bbox_size = coeffs * 0.001 * 2
        obj_bbox_dict['size'] = bbox_size.tolist()
        if check_bbox_in_room(obj_bbox_dict, room_layout_mesh, layout_bbox_min, layout_bbox_max):
            obj_bbox_lst.append(obj_bbox_dict)

    anno_img = vis_objs3d(image=rgb_img, v_bbox3d=obj_bbox_lst, camera_position=cam_position, 
                          b_show_axes=False, b_show_centroid=False, b_show_bbox3d=True, b_show_info=True, thickness=2)
    
    scene_mesh = vis_scene_mesh(room_layout_mesh, obj_bbox_lst, room_layout_bbox=None)

    obj_bbox_dicts = {}
    obj_bbox_dicts['objects'] = obj_bbox_lst
    return obj_bbox_dicts, anno_img, scene_mesh

def parse_room_layout(img_filepath:str, cam_pos_filepath:str, layout_coor_filepath:str, room_id_str:str, scene_anno_3d_dict:Dict):
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


    points, faces= get_mesh_from_corners(corners_lst, H, W, camera_position=cam_pos_lst, rgb_img=equirect_img,
                                b_ignore_floor=False, b_ignore_ceiling=False, b_ignore_wall=False, b_in_world_frame=False)
    # print(f'points.shape: {points.shape}, faces.shape: {faces.shape}')
    # downsample the mesh
    raw_mesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(points[:,:3]), triangles=o3d.utility.Vector3iVector(faces))
    simplified_mesh = raw_mesh.simplify_vertex_clustering(voxel_size=0.1, contraction=o3d.geometry.SimplificationContraction.Average)
    simplified_vertices = simplified_mesh.vertices
    simplified_faces = np.array(simplified_mesh.triangles)
    room_layout_mesh = trimesh.Trimesh(vertices=simplified_vertices, faces=simplified_faces, process=True)
    return room_layout_mesh
        
def prepare_dataset(raw_dataset_dir, scene_ids, out_dir):

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

            room_path = os.path.join(scene_dir, room_id, "panorama")
            source_img_path = os.path.join(room_path, "full", "rgb_rawlight.png")
            source_cor_path = os.path.join(room_path, "layout.txt")
            source_cam_pos_path = os.path.join(room_path, "camera_xyz.txt")
            room_bbox_3d_path = os.path.join(room_path, 'full', 'bbox_3d.json')

            # parse room layout
            room_layout_mesh = parse_room_layout(source_img_path, source_cam_pos_path, source_cor_path, room_id, scene_anno_3d_dict)
            # parse 3d bbox of objects in the room
            obj_bbox_3d_dict, debug_bbox_img, debug_bbox_trimesh = parse_bbox_in_room(room_path, room_layout_mesh)
            
            out_img_dir = os.path.join(out_dir, room_type_str, 'img')
            out_cord_dir = os.path.join(out_dir, room_type_str, 'label_cor')
            out_cam_pos_dir = os.path.join(out_dir, room_type_str, 'cam_pos')
            out_room_type_dir = os.path.join(out_dir, room_type_str, 'room_type')
            out_bbox_3d_dir = os.path.join(out_dir, room_type_str, 'bbox_3d')
            os.makedirs(out_img_dir, exist_ok=True)
            os.makedirs(out_cord_dir, exist_ok=True)
            os.makedirs(out_cam_pos_dir, exist_ok=True)
            os.makedirs(out_room_type_dir, exist_ok=True)
            os.makedirs(out_bbox_3d_dir, exist_ok=True)
            target_img_path = os.path.join(out_img_dir, '%s_%s.png' % (scene_id, room_id))
            target_cor_path = os.path.join(out_cord_dir, '%s_%s.txt' % (scene_id, room_id))
            target_cam_pos_path = os.path.join(out_cam_pos_dir, '%s_%s.txt' % (scene_id, room_id))
            target_room_type_path = os.path.join(out_room_type_dir, '%s_%s.txt' % (scene_id, room_id))
            target_bbox_3d_path = os.path.join(out_bbox_3d_dir, '%s_%s.json' % (scene_id, room_id))
            target_bbox_3d_vis_path = os.path.join(out_bbox_3d_dir, '%s_%s.png' % (scene_id, room_id))
            target_bbox_3d_mesh_path = os.path.join(out_bbox_3d_dir, '%s_%s.ply' % (scene_id, room_id))

            if not os.path.isfile(source_img_path) or not os.path.isfile(source_cor_path) \
            or not os.path.isfile(source_cam_pos_path) or imghdr.what(source_img_path) is None:
                INVALID_ROOMS_LST.append(room_str)
                print(f'bad scene {room_str}')
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

                # visualize debug bbox img
                cv2.imwrite(target_bbox_3d_vis_path, debug_bbox_img)
                # print(f'save visualization for object bbox annotation of {room_id_str}')
                debug_bbox_trimesh.export(target_bbox_3d_mesh_path)


    print('*' * 10 + ' invalid rooms ids: ' + '*' * 10)
    print(INVALID_ROOMS_LST)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Structured3D 2D Layout Visualization")
    parser.add_argument("--dataset_path",
                        default="/data/dataset/Structured3D/Structured3D/",
                        help="raw dataset path",
                        metavar="DIR")
    parser.add_argument(
        '--out_all_path',
        default=
        '/mnt/nas_3dv/hdd1/datasets/Structured3d/preprocessed/all_raw_light')
    parser.add_argument(
        '--out_train_path',
        default=
        '/mnt/nas_3dv/hdd1/datasets/Structured3d/preprocessed/st3d_train_full_raw_light')
    parser.add_argument(
        '--out_valid_path',
        default=
        '/mnt/nas_3dv/hdd1/datasets/Structured3d/preprocessed/st3d_valid_full_raw_light')
    parser.add_argument(
        '--out_test_path',
        default=
        '/mnt/nas_3dv/hdd1/datasets/Structured3d/preprocessed/st3d_test_full_raw_light')
    return parser.parse_args()


def main():
    args = parse_args()

    prepare_dataset(args.dataset_path, ALL_SCENE, args.out_all_path)
    # prepare_dataset(args.dataset_path, TRAIN_SCENE, args.out_train_path)
    # prepare_dataset(args.dataset_path, VALID_SCENE, args.out_valid_path)
    # prepare_dataset(args.dataset_path, TEST_SCENE, args.out_test_path)


if __name__ == "__main__":
    main()
