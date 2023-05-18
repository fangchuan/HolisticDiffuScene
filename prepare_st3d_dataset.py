import os
import argparse

import json
import numpy as np
from tqdm import tqdm
import imghdr

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
        - scene_xxxxx_*png (softlink)
    - label_cor/
        - scene_xxxxx_*txt (softlink)
    - cam_pos/
        - scene_xxxxx_*txt (softlink)

- {out_valid_root} ...
- {out_test_root} ...
'''
TRAIN_SCENE = ['scene_%05d' % i for i in range(0, 3000)]
VALID_SCENE = ['scene_%05d' % i for i in range(3000, 3250)]
TEST_SCENE = ['scene_%05d' % i for i in range(3250, 3500)]

INVALID_SCENES_LST = [
    'scene_01155', 'scene_01714', 'scene_01816', 'scene_03398', 'scene_01192',
    'scene_01852'
]

INVALID_ROOMS_LST = [
    'scene_00212_494', 'scene_00403_11105', 'scene_00411_918447',
    'scene_00810_1817', 'scene_01209_4566', 'scene_01410_180208',
    'scene_01815_1429', 'scene_01815_237', 'scene_02011_2003',
    'scene_02212_1058346', 'scene_02212_175980', 'scene_02411_26703128',
    'scene_02608_345', 'scene_00014_1083', 'scene_00014_947',
    'scene_00212_377', 'scene_01209_5767', 'scene_01210_277',
    'scene_01608_270286', 'scene_01608_270288', 'scene_02411_706561',
    'scene_02609_1419', 'scene_00609_159'
]


def prepare_dataset(raw_dataset_dir, scene_ids, out_dir):
    out_img_dir = os.path.join(out_dir, 'img')
    out_cord_dir = os.path.join(out_dir, 'label_cor')
    out_cam_pos_dir = os.path.join(out_dir, 'cam_pos')
    out_room_type_dir = os.path.join(out_dir, 'room_type')
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_cord_dir, exist_ok=True)
    os.makedirs(out_cam_pos_dir, exist_ok=True)
    os.makedirs(out_room_type_dir, exist_ok=True)

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
            # print(source_img_path)
            # print(source_cor_path)
            target_img_path = os.path.join(out_img_dir, '%s_%s.png' % (scene_id, room_id))
            target_cor_path = os.path.join(out_cord_dir, '%s_%s.txt' % (scene_id, room_id))
            target_cam_pos_path = os.path.join(out_cam_pos_dir, '%s_%s.txt' % (scene_id, room_id))
            target_room_type_path = os.path.join(out_room_type_dir, '%s_%s.txt' % (scene_id, room_id))
            # assert os.path.isfile(source_img_path)
            # assert os.path.isfile(source_cor_path)
            if not os.path.isfile(source_img_path) or not os.path.isfile(source_cor_path) \
            or not os.path.isfile(source_cam_pos_path) or imghdr.what(source_img_path) is None:
                INVALID_ROOMS_LST.append(room_str)
                print(f'bad scene {room_str}')
                continue
            else:
                os.symlink(source_img_path, target_img_path)
                os.symlink(source_cor_path, target_cor_path)
                os.symlink(source_cam_pos_path, target_cam_pos_path)
                # write room type
                with open(target_room_type_path, 'w') as f:
                    f.write(room_type_str)

    print('*' * 10 + ' invalid rooms ids: ' + '*' * 10)
    print(INVALID_ROOMS_LST)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Structured3D 2D Layout Visualization")
    parser.add_argument("--dataset_path",
                        required=True,
                        default="/data/dataset/Structured3D/Structured3D/",
                        help="raw dataset path",
                        metavar="DIR")
    parser.add_argument(
        "--bbox_path",
        required=True,
        default="/data/dataset/Structured3D/Structured3D_bbox/Structured3D/",
        help="3d bbox folder path")
    parser.add_argument(
        '--out_train_path',
        default=
        '/data/dataset/Structured3D/preprocessed/st3d_train_full_raw_light')
    parser.add_argument(
        '--out_valid_path',
        default=
        '/data/dataset/Structured3D/preprocessed/st3d_valid_full_raw_light')
    parser.add_argument(
        '--out_test_path',
        default=
        '/data/dataset/Structured3D/preprocessed/st3d_test_full_raw_light')
    return parser.parse_args()


def main():
    args = parse_args()

    prepare_dataset(args.dataset_path, TRAIN_SCENE, args.out_train_path)
    prepare_dataset(args.dataset_path, VALID_SCENE, args.out_valid_path)
    prepare_dataset(args.dataset_path, TEST_SCENE, args.out_test_path)


if __name__ == "__main__":
    main()
