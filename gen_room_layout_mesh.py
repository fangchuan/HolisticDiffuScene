import os
import cv2
import argparse

import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib as matlib

from shapely.geometry import Polygon
from descartes.patch import PolygonPatch
from tqdm import tqdm
from PIL import Image
import open3d as o3d

# from misc.panorama import draw_boundary_from_cor_id
# from misc.colors import colormap_255
from dataset.metadata import ST3D_BEDROOM_FURNITURE, ST3D_LIVINGROOM_FURNITURE, ST3D_DININGROOM_FURNITURE
from dataset.st3d_dataset import PanoCorBoundDataset, np_coor2xy, np_coor2xy, ROOM_TYPE_DICT
from prepare_st3d_dataset import vis_scene_mesh
from misc.equirect_projection import vis_objs3d


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir',
                        default='/mnt/nas_3dv/hdd1/datasets/Structured3d/preprocessed/all_quad_walls/bedroom/')
    parser.add_argument(
        '--samples_filepath',
        default=
        '/home/hkust/fangchuan/codes/Structured3D/sample_results/openai-2023-07-05-16-35-56-396382/samples_10x23x32.npz'
    )
    parser.add_argument('--room_type', default='bedroom', type=str, help='generated room type')
    parser.add_argument('--ith', default=0, type=int, help='Pick a data id to visualize.'
                        '-1 for visualize all data')
    parser.add_argument('--flip', action='store_true', help='whether to random flip')
    parser.add_argument('--rotate', action='store_true', help='whether to random horizon rotation')
    parser.add_argument('--gamma', action='store_true', help='whether to random luminance change')
    parser.add_argument('--vis_layout_mesh', action='store_true', help='whether to visualize layout mesh')
    parser.add_argument('--vis_layout_wireframe', action='store_true', help='whether to visualize wireframe of layout')
    parser.add_argument('--out_dir', default='sample_dataset_visualization')
    return parser.parse_args()


def visualize_a_data(img, bound_y_lst, corner_y_lst):
    img = (img.numpy().transpose([1, 2, 0]) * 255).astype(np.uint8)
    img_H, img_W = img.shape[:2]
    bound_y_lst = bound_y_lst.numpy()
    # scale to image pixel coordinates
    bound_y_lst = ((bound_y_lst / np.pi + 0.5) * img_H).round().astype(int)
    corner_y_lst = corner_y_lst.numpy()

    git_corner_img = np.zeros((30, 1024, 3), np.uint8)
    git_corner_img[:] = corner_y_lst[0][None, :, None] * 255
    padding_img = np.zeros((3, 1024, 3), np.uint8) + 255

    img_with_boundary = (img.copy() * 0.5).astype(np.uint8)
    y1 = np.round(bound_y_lst[0]).astype(int)
    y2 = np.round(bound_y_lst[1]).astype(int)
    y1 = np.vstack([np.arange(1024), y1]).T.reshape(-1, 1, 2)
    y2 = np.vstack([np.arange(1024), y2]).T.reshape(-1, 1, 2)
    img_with_boundary[bound_y_lst[0], np.arange(len(bound_y_lst[0])), 1] = 255
    img_with_boundary[bound_y_lst[1], np.arange(len(bound_y_lst[1])), 1] = 255

    return np.concatenate([git_corner_img, padding_img, img_with_boundary], 0)


def visualize_synth_data_4_1024(bound_y_lst, corner_y_lst, obj_bbox_lst, cam_position):
    img = np.zeros((512, 1024, 3), np.uint8)
    img_H, img_W = img.shape[:2]
    # scale to image pixel coordinates
    bound_y_lst = ((bound_y_lst / np.pi + 0.5) * img_H).round().astype(int)

    git_corner_img = np.zeros((30, 1024, 3), np.uint8)
    git_corner_img[:] = corner_y_lst[None, :, None] * 255
    padding_img = np.zeros((3, 1024, 3), np.uint8) + 255

    img_with_boundary = (img.copy() * 0.5).astype(np.uint8)
    # draw boundary lines green
    img_with_boundary[bound_y_lst[0], np.arange(len(bound_y_lst[0])), 1] = 255
    img_with_boundary[bound_y_lst[1], np.arange(len(bound_y_lst[1])), 1] = 255

    ret_img = np.concatenate([git_corner_img, padding_img, img_with_boundary], 0)
    ret_img = vis_objs3d(ret_img,
                         v_bbox3d=obj_bbox_lst,
                         camera_position=cam_position,
                         b_show_axes=False,
                         b_show_centroid=False,
                         b_show_bbox3d=True,
                         b_show_info=True)
    return ret_img


# save layout mesh
def save_layout_mesh(save_filepath: str, points: np.array, faces: np.array):
    ply_header = '\n'.join([
        'ply',
        'format ascii 1.0',
        f'element vertex {len(points):d}',
        'property float x',
        'property float y',
        'property float z',
        'property uchar red',
        'property uchar green',
        'property uchar blue',
        f'element face {len(faces):d}',
        'property list uchar int vertex_indices',
        'end_header',
    ])
    with open(save_filepath, 'w') as f:
        f.write(ply_header)
        f.write('\n')
        for x, y, z, r, g, b in points:
            f.write(f'{x:.2f} {y:.2f} {z:.2f} {r:.0f} {g:.0f} {b:.0f}\n')
        for i, j, k in faces:
            f.write(f'3 {i:d} {j:d} {k:d}\n')


def recover_quad_wall_layout_mesh(room_type: str, quad_wall_lst: np.ndarray, object_bbox_lst: np.ndarray):
    # room_layout_mesh = trimesh.Trimesh(vertices=points[:, :3], faces=faces)
    # room_layout_bbox_min = trimesh.bounds.corners(room_layout_mesh.bounding_box_oriented.bounds).min(axis=0)
    # room_layout_bbox_max = trimesh.bounds.corners(room_layout_mesh.bounding_box_oriented.bounds).max(axis=0)
    # room_layout_bbox_size = room_layout_bbox_max - room_layout_bbox_min
    # print(f'room_layout_bbox_size: {room_layout_bbox_size}')
    room_layout_bbox_size = np.array([1.0, 1.0, 1.0])

    if room_type == 'bedroom':
        class_labels_lst = (ST3D_BEDROOM_FURNITURE)
    elif room_type == 'living_room':
        class_labels_lst = (ST3D_LIVINGROOM_FURNITURE)
    elif room_type == 'dining_room':
        class_labels_lst = (ST3D_DININGROOM_FURNITURE)
    else:
        raise NotImplementedError

    print(f' room_type: {room_type}, class_labels_lst: {len(class_labels_lst)}')
    class_idx = 0
    centroid_idx = len(class_labels_lst)
    size_idx = 3 + centroid_idx
    angle_idx = 3 + size_idx

    # recover quad wall bbox of room layout
    quad_wall_dict_list = []
    for i in range(len(quad_wall_lst)):
        quad_wall_dict = {}
        # recover class label
        class_label_prob = quad_wall_lst[i][:centroid_idx]
        # print(f'class_label_prob: {class_label_prob}')
        class_label_prob = np.where(class_label_prob > 0.5, 1, 0)
        class_label = class_labels_lst[class_label_prob.argmax()]
        if class_label == 'empty':
            print(f'wall {i} is empty')
            continue
        quad_wall_dict['class'] = class_label
        wall_center = quad_wall_lst[i][centroid_idx:centroid_idx + 3] * room_layout_bbox_size
        quad_wall_dict['center'] = wall_center.tolist()
        wall_size = quad_wall_lst[i][size_idx:size_idx + 3]
        wall_normal = quad_wall_lst[i][angle_idx:angle_idx + 2]
        # rearrange_func = lambda normal, size: np.array([normal[0], normal[1], size[1]]), np.array([size[0], size[2]])
        # wall_normal, wall_size = rearrange_func(wall_normal, wall_size)
        wall_normal = [wall_normal[0], wall_normal[1], wall_size[1]]
        wall_size = [wall_size[0], wall_size[2]]
        wall_size = [
            wall_size[0] * max(room_layout_bbox_size[0], room_layout_bbox_size[1]), 0.01,
            wall_size[1] * room_layout_bbox_size[2]
        ]
        # The direction of all camera is always along the negative y-axis.
        cos_angle = np.array(wall_normal).dot(np.array([0, -1, 0]))
        angle = np.arccos(cos_angle)
        if abs(cos_angle) < 1e-3:
            angle = np.pi / 2 if wall_normal[0] > 0 else -np.pi / 2
        quad_wall_dict['size'] = wall_size
        quad_wall_dict['angles'] = [0, 0, angle]
        print(f' wall {class_label} centroid: {wall_center} size: {wall_size} noraml: {wall_normal}')
        quad_wall_dict_list.append(quad_wall_dict)

    # recover object bbox
    obj_bbox_dict_list = []
    for i in range(len(object_bbox_lst)):
        # print(f'predict object bbox feature: {object_bbox_lst[i]}')
        obj_bbox_dict = {}

        # recover class label
        class_label_prob = object_bbox_lst[i][:centroid_idx]
        # print(f'class_label_prob: {class_label_prob}')
        class_label_prob = np.where(class_label_prob > 0.5, 1, 0)
        if len(class_label_prob) == 0:
            print(f'object {i} has no class label')
        class_label = class_labels_lst[class_label_prob.argmax()]
        if class_label == 'empty':
            print(f'object {i} is empty')
            continue
        obj_bbox_dict['class'] = class_label

        # recover centroid
        centroid = object_bbox_lst[i][centroid_idx:size_idx]
        centroid = centroid * room_layout_bbox_size
        obj_bbox_dict['center'] = centroid.tolist()
        # recover size
        size = object_bbox_lst[i][size_idx:angle_idx]
        size = (size + 1) * 0.5
        size = size * room_layout_bbox_size
        obj_bbox_dict['size'] = size.tolist()
        # recover angle
        angle = object_bbox_lst[i][angle_idx:]
        angle_0 = np.arccos(angle[0])
        angle_1 = np.arcsin(angle[1])
        obj_bbox_dict['angles'] = [0, 0, angle_0]
        print(f' object {class_label} centroid: {centroid} size: {size} angle: {angle_0}')
        obj_bbox_dict_list.append(obj_bbox_dict)

    quad_wall_dict_list.extend(obj_bbox_dict_list)
    return quad_wall_dict_list


if __name__ == "__main__":

    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print('args:')
    for key, val in vars(args).items():
        print('    {:16} {}'.format(key, val))

    dataset = PanoCorBoundDataset(root_dir=args.root_dir,
                                  flip=args.flip,
                                  rotate=args.rotate,
                                  gamma=args.gamma,
                                  return_path=True)

    # Showing some information about dataset
    print('len(dataset): {}'.format(len(dataset)))
    # img, boundary_lst, wall_y_prob_lst, img_filepath = dataset[0]
    # print('image : ', img.size())
    # print('ceiling-wall and floor-wall boundary veto: ', boundary_lst.size())
    # print('wall-wall probability vector: ', wall_y_prob_lst.size())

    b_vis_mesh_from_corners = False
    b_vis_mesh_from_diffusion = True

    # load sample results: Bx3x1024
    sample_result_lst = np.load(args.samples_filepath)

    # for idx, data_items in enumerate(tqdm(dataset)):
    #     if idx != args.ith:
    #         continue
    #     else:
    #         img, boundary_lst, wall_prob_lst, img_filepath = data_items
    #         img_fname = os.path.split(img_filepath)[-1]
    #         out_img = visualize_a_data(img, boundary_lst, wall_prob_lst)
    #         save_img_filepath = os.path.join(args.out_dir, img_fname)
    #         Image.fromarray(out_img).save(save_img_filepath)

    #         # save results ply
    #         boundary_lst = boundary_lst.numpy()
    #         wall_prob_lst = wall_prob_lst.numpy()[0]
    #         if b_vis_mesh_from_corners:
    #             layout_ply_points, layout_ply_faces, layout_corner_lst, cam_pos_lst = dataset.get_gt_layout_mesh(idx)
    #         elif b_vis_mesh_from_diffusion:
    #             layout_ply_points, layout_ply_faces, layout_corner_lst, cam_pos_lst = dataset.get_layout_mesh_from_prediction(bound_ceil_floor_lst=boundary_lst,
    #                                                                                                                           wall_prob_lst=wall_prob_lst)
    #         print('layout_ply_points: ', layout_ply_points.shape)
    #         print('layout_ply_faces: ', layout_ply_faces.shape)
    #         print('layout_corner_lst: ', layout_corner_lst.shape)
    #         print('cam_pos_lst: ', cam_pos_lst)
    #         ply_fname = img_fname.replace(img_fname[-4:] , '.ply')
    #         save_ply_filepath = os.path.join(args.out_dir, ply_fname)
    #         save_layout_mesh(save_ply_filepath, layout_ply_points, layout_ply_faces)

    #         if args.vis_layout_mesh:
    #             mesh = o3d.geometry.TriangleMesh()
    #             mesh.vertices = o3d.utility.Vector3dVector(layout_ply_points[:, :3])
    #             mesh.vertex_colors = o3d.utility.Vector3dVector(layout_ply_points[:, 3:] / 255.)
    #             mesh.triangles = o3d.utility.Vector3iVector(layout_ply_faces)
    #             draw_geometries = [mesh]

    #             # Show wireframe
    #             if args.vis_layout_wireframe:
    #                 # Convert cor_id to 3d xyz
    #                 N = len(layout_corner_lst) // 2
    #                 floor_height = cam_pos_lst[2]
    #                 floor_xy = np_coor2xy(layout_corner_lst[1::2], floor_height, img.shape[1], img.shape[0], floorW=1, floorH=1)
    #                 c = np.sqrt((floor_xy**2).sum(1))
    #                 v = np_coor2xy(layout_corner_lst[0::2, 1], img.shape[0])
    #                 ceil_z = (c * np.tan(v)).mean()

    #                 # Prepare wireframe in open3d
    #                 assert N == len(floor_xy)
    #                 wf_points = [[x, y, floor_height] for x, y in floor_xy] +\
    #                             [[x, y, ceil_z] for x, y in floor_xy]
    #                 wf_lines = [[i, (i+1)%N] for i in range(N)] +\
    #                         [[i+N, (i+1)%N+N] for i in range(N)] +\
    #                         [[i, i+N] for i in range(N)]
    #                 wf_colors = [[1, 0, 0] for i in range(len(wf_lines))]
    #                 wf_line_set = o3d.geometry.LineSet()
    #                 wf_line_set.points = o3d.utility.Vector3dVector(wf_points)
    #                 wf_line_set.lines = o3d.utility.Vector2iVector(wf_lines)
    #                 wf_line_set.colors = o3d.utility.Vector3dVector(wf_colors)
    #                 draw_geometries.append(wf_line_set)

    #             o3d.visualization.draw_geometries(draw_geometries, mesh_show_back_face=True)

    #         break

    for idx in range(len(sample_result_lst['arr_0'])):
        scene_sample_result = sample_result_lst['arr_0'][idx]
        # print(f'scene_sample_result: {scene_sample_result.shape}')
        scene_sample_label = sample_result_lst['arr_1'][idx]
        scene_sample_label = [key for key in ROOM_TYPE_DICT.keys() if ROOM_TYPE_DICT[key] == scene_sample_label][0]
        scene_sample_label = scene_sample_label.replace(' ', '_')
        print(f'scene_sample_label: {scene_sample_label}')

        # # convert sample into real range
        # boundary_lst = scene_sample_result[:2, :] * 0.5 * np.pi
        # wall_prob_lst = (scene_sample_result[2, :] + 1) * 0.5
        # quad walls
        quad_wall_lst = scene_sample_result[:10, :]

        def recover_real_range(quad_wall_lst):
            ret = []
            for quad_wall in quad_wall_lst:
                ret_quad_wall = np.zeros_like(quad_wall)
                # class label
                ret_quad_wall[:24] = quad_wall[:24]
                # translation
                ret_quad_wall[24:24 + 3] = quad_wall[24:24 + 3]
                # normal
                ret_quad_wall[24 + 3:24 + 3 + 3] = quad_wall[24 + 3:24 + 3 + 3]
                # size
                ret_quad_wall[24 + 3 + 3:24 + 3 + 3 + 2] = quad_wall[24 + 3 + 3:24 + 3 + 3 + 2]
                ret.append(ret_quad_wall)
            return np.array(ret)

        quad_wall_lst = recover_real_range(quad_wall_lst)
        print(f'quad_wall_lst: {quad_wall_lst.shape}')
        # if args.room_type == 'bedroom':
        #     obj_feat_num, obj_feat_dim = 13, 32
        # elif args.room_type == 'living_room':
        #     obj_feat_num, obj_feat_dim = 24, 32
        # elif args.room_type == 'dining_room':
        #     obj_feat_num, obj_feat_dim = 24, 32

        # obj_bbox_lst = scene_sample_result[3, :(obj_feat_num * obj_feat_dim)].reshape((obj_feat_num, obj_feat_dim))
        obj_bbox_lst = scene_sample_result[10:, :]

        def recover_object_bbox_real_range(obj_bbox_lst):
            ret = []
            for bbox in obj_bbox_lst:
                ret_obj_bbox = np.zeros_like(bbox)
                # class label
                ret_obj_bbox[:24] = bbox[:24]
                # translation
                ret_obj_bbox[24:24 + 3] = bbox[24:24 + 3]
                # size
                ret_obj_bbox[24 + 3:24 + 3 + 3] = bbox[24 + 3:24 + 3 + 3]
                # angles
                ret_obj_bbox[24 + 3 + 3:24 + 3 + 3 + 2] = bbox[24 + 3 + 3:24 + 3 + 3 + 2]
                ret.append(ret_obj_bbox)
            return np.array(ret)

        obj_bbox_lst = recover_object_bbox_real_range(obj_bbox_lst)
        print(f'obj_bbox_lst: {obj_bbox_lst.shape}')

        if b_vis_mesh_from_corners:
            layout_ply_points, layout_ply_faces, layout_corner_lst, cam_pos_lst = dataset.get_gt_layout_mesh(idx)
        elif b_vis_mesh_from_diffusion:
            # layout_ply_points, layout_ply_faces, layout_corner_lst, cam_pos_lst, room_layout_mesh, obj_bbox_dict_lst = dataset.get_predicted_layout_mesh(
            #     room_type=args.room_type,
            #     bound_ceil_floor_lst=boundary_lst,
            #     wall_prob_lst=wall_prob_lst,
            #     obj_bbox_lst=obj_bbox_lst,
            #     b_force_raw=False)
            obj_bbox_dict_lst = recover_quad_wall_layout_mesh(args.room_type,
                                                              quad_wall_lst=quad_wall_lst,
                                                              object_bbox_lst=obj_bbox_lst)
        # print('layout_ply_points: ', layout_ply_points.shape)
        # print('layout_ply_faces: ', layout_ply_faces.shape)
        # print('layout_corner_lst: ', layout_corner_lst.shape)
        # print('cam_pos_lst: ', cam_pos_lst)

        # save synthetic boundaries as image
        img_fname = f'{scene_sample_label}_{idx}.png'
        # out_img = visualize_synth_data_4_1024(bound_y_lst=boundary_lst,
        #                                       corner_y_lst=wall_prob_lst,
        #                                       obj_bbox_lst=obj_bbox_dict_lst,
        #                                       cam_position=cam_pos_lst)
        out_img = np.zeros((512, 1024, 3), np.uint8)
        cam_position = np.zeros((3,), np.float32)
        out_img = vis_objs3d(out_img,
                             v_bbox3d=obj_bbox_dict_lst,
                             camera_position=cam_position,
                             b_show_axes=False,
                             b_show_centroid=False,
                             b_show_bbox3d=True,
                             b_show_info=True)
        save_img_filepath = os.path.join(args.out_dir, img_fname)
        Image.fromarray(out_img).save(save_img_filepath)

        # save synthetic object and room_layout as ply
        ply_fname = f'{scene_sample_label}_{idx}.ply'
        save_ply_filepath = os.path.join(args.out_dir, ply_fname)
        # save_layout_mesh(save_ply_filepath, layout_ply_points, layout_ply_faces)
        scene_mesh = vis_scene_mesh(room_layout_mesh=None, obj_bbox_lst=obj_bbox_dict_lst, room_layout_bbox=None)
        scene_mesh.export(save_ply_filepath)
