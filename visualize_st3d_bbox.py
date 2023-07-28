import os
import argparse
import json
import cv2

from misc.equirect_projection import vis_objs3d


def visualize_bbox_annotations(rgb_imgs_folderpath, labels_folderpath):

    annotated_room_ids_lst = [
        room.split('.')[0]
        for room in os.listdir(labels_folderpath)
        if room.endswith('.json') and room.startswith('scene_')
    ]
    annotated_room_ids_lst = sorted(annotated_room_ids_lst)

    for room_id in annotated_room_ids_lst:
        rgb_img_path = os.path.join(rgb_imgs_folderpath, room_id + '.png')
        label_file_path = os.path.join(labels_folderpath, room_id + '.json')

        if not os.path.isfile(label_file_path):
            print(' room annotstion file not found: ', label_file_path)
            continue
        if not os.path.isfile(rgb_img_path):
            print(' rgb image file not found: ', rgb_img_path)
            continue

        # read rgb image
        rgb_img = cv2.imread(rgb_img_path, cv2.IMREAD_UNCHANGED)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

        # read object annotations file
        with open(label_file_path, 'r') as file:
            room_annos = json.load(file)
            room_annos = room_annos['objects']

        output_vis_folderpath = os.path.join(os.path.dirname(labels_folderpath), 'debug_annotations')
        if not os.path.exists(output_vis_folderpath):
            os.makedirs(output_vis_folderpath, exist_ok=True)

        object_annos_dict_lst = []
        for obj_anno in room_annos:
            obj_annos_dict = {}
            class_label = obj_anno['name']
            obj_annos_dict['class'] = class_label
            angle_x = float(obj_anno['rotations']['x'])
            angle_y = float(obj_anno['rotations']['y'])
            angle_z = float(obj_anno['rotations']['z'])
            obj_annos_dict['angles'] = [angle_x, angle_y, angle_z]

            size_x = float(obj_anno['dimensions']['length'])
            size_y = float(obj_anno['dimensions']['width'])
            size_z = float(obj_anno['dimensions']['height'])
            obj_annos_dict['size'] = [size_x, size_y, size_z]

            center_x = float(obj_anno['centroid']['x'])
            center_y = float(obj_anno['centroid']['y'])
            center_z = float(obj_anno['centroid']['z'])
            obj_annos_dict['center'] = [center_x, center_y, center_z]

            object_annos_dict_lst.append(obj_annos_dict)

        # save debug image
        debug_img = vis_objs3d(image=rgb_img,
                               v_bbox3d=object_annos_dict_lst,
                               camera_position=None,
                               b_show_axes=False,
                               b_show_centroid=False,
                               b_show_bbox3d=True,
                               b_show_info=True)
        output_img_filepath = os.path.join(output_vis_folderpath, room_id + '_bbox.png')
        #     print(f'save visualization for object bbox annotation of {room_id_str}')
        cv2.imwrite(output_img_filepath, debug_img)


def parse_args():
    parser = argparse.ArgumentParser(description="Structured3D 3D Bounding Box Visualization")
    parser.add_argument("--rgbs_path",
                        default="/data/dataset/Structured3D/preprocessed/annotations/livingroom/rgb",
                        help="raw dataset path",
                        metavar="DIR")
    parser.add_argument("--labels_path",
                        default="/data/dataset/Structured3D/preprocessed/annotations/livingroom/labels",
                        help="debug folder path for object bbox annotations",
                        metavar="DIR")
    return parser.parse_args()


def main():
    args = parse_args()
    rgb_imgs_folderpath = args.rgbs_path
    labels_folderpath = args.labels_path

    if not os.path.exists(rgb_imgs_folderpath) or not os.path.isdir(labels_folderpath):
        raise ValueError("rgb images folder does not exist!")

    visualize_bbox_annotations(rgb_imgs_folderpath, labels_folderpath)


if __name__ == "__main__":
    main()