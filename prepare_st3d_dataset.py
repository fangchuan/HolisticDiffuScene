import os
import argparse

import json
import numpy as np
from tqdm import tqdm
import imghdr
import shutil
import cv2

from misc.utils import matrix_to_euler_angles
from misc.equirect_projection import vis_objs3d
from dataset.metadata import INVALID_SCENES_LST, INVALID_ROOMS_LST, OBJECT_LABEL_IDS
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


def parse_bbox_in_room(room_folderpath:str):
    room_bbox_3d_path = os.path.join(room_folderpath, 'full', 'bbox_3d.json')
    rgb_img_path = os.path.join(room_folderpath, 'full', 'rgb_rawlight.png')
    instance_img_path = os.path.join(room_folderpath, 'full', 'instance.png')
    camera_pos_path = os.path.join(room_folderpath, 'camera_xyz.txt')

    with open(room_bbox_3d_path, 'r') as file:
        room_anno_3d_dict = json.load(file)

    id2index = dict()
    for index, object in enumerate(room_anno_3d_dict):
        id2index[object.get('ID')] = index

    rgb_img = cv2.imread(rgb_img_path, cv2.IMREAD_UNCHANGED)
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    instance_img = cv2.imread(instance_img_path, cv2.IMREAD_UNCHANGED)
    cam_position = np.loadtxt(camera_pos_path)

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

        # obj_bbox_dict['angles'] = R.from_matrix(basis).as_euler('xyz', degrees=False).tolist()
        obj_bbox_dict['angles'] = matrix_to_euler_angles(basis).tolist()
        obj_bbox_dict['center'] = list((centroid - cam_position) * 0.001)
        obj_bbox_dict['size'] = list(coeffs * 0.001 * 2)
        obj_bbox_lst.append(obj_bbox_dict)

    anno_img = vis_objs3d(image=rgb_img, v_bbox3d=obj_bbox_lst, camera_position=cam_position, 
                          b_show_axes=False, b_show_centroid=False, b_show_bbox3d=True, b_show_info=True, thickness=2)
    obj_bbox_dicts = {}
    obj_bbox_dicts['objects'] = obj_bbox_lst
    return obj_bbox_dicts, anno_img

def prepare_dataset(raw_dataset_dir, scene_ids, out_dir):

    for scene_id in tqdm(scene_ids):
        if scene_id in INVALID_SCENES_LST:
            continue
            
        room_type_lst = None
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
            # parse 3d bbox of objects in the room
            obj_bbox_3d_dict, debug_bbox_img = parse_bbox_in_room(room_path)
            
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
        '/mnt/nas_3dv/hdd0/dataset/Structured3d/preprocessed/all_raw_light')
    parser.add_argument(
        '--out_train_path',
        default=
        '/mnt/nas_3dv/hdd0/dataset/Structured3d/preprocessed/st3d_train_full_raw_light')
    parser.add_argument(
        '--out_valid_path',
        default=
        '/mnt/nas_3dv/hdd0/dataset/Structured3d/preprocessed/st3d_valid_full_raw_light')
    parser.add_argument(
        '--out_test_path',
        default=
        '/mnt/nas_3dv/hdd0/dataset/Structured3d/preprocessed/st3d_test_full_raw_light')
    return parser.parse_args()


def main():
    args = parse_args()

    prepare_dataset(args.dataset_path, ALL_SCENE, args.out_all_path)
    # prepare_dataset(args.dataset_path, TRAIN_SCENE, args.out_train_path)
    # prepare_dataset(args.dataset_path, VALID_SCENE, args.out_valid_path)
    # prepare_dataset(args.dataset_path, TEST_SCENE, args.out_test_path)


if __name__ == "__main__":
    main()
