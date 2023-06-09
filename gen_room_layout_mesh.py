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

from dataset.st3d_dataset import PanoCorBoundDataset, np_coor2xy, np_coor2xy, ROOM_TYPE_DICT



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='/data/dataset/Structured3D/preprocessed/st3d_train_full_raw_light/')
    parser.add_argument('--samples_filepath', default='/home/hkust/fangchuan/codes/Structured3D/sample_results/samples_10x3x1024.npz')
    parser.add_argument('--ith', default=0, type=int,
                        help='Pick a data id to visualize.'
                             '-1 for visualize all data')
    parser.add_argument('--flip', action='store_true',
                        help='whether to random flip')
    parser.add_argument('--rotate', action='store_true',
                        help='whether to random horizon rotation')
    parser.add_argument('--gamma', action='store_true',
                        help='whether to random luminance change')
    parser.add_argument('--vis_layout_mesh', action='store_true',
                        help='whether to visualize layout mesh')
    parser.add_argument('--vis_layout_wireframe', action='store_true',
                        help='whether to visualize wireframe of layout')
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

def visualize_synth_data(bound_y_lst, corner_y_lst):
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

    return np.concatenate([git_corner_img, padding_img, img_with_boundary], 0)

# save layout mesh 
def save_layout_mesh(save_filepath:str, points:np.array, faces:np.array):
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



if __name__ == "__main__":

    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print('args:')
    for key, val in vars(args).items():
        print('    {:16} {}'.format(key, val))

    dataset = PanoCorBoundDataset(
        root_dir=args.root_dir,
        flip=args.flip, rotate=args.rotate, gamma=args.gamma, 
        return_path=True)

    # Showing some information about dataset
    print('len(dataset): {}'.format(len(dataset)))
    img, boundary_lst, wall_y_prob_lst, img_filepath = dataset[0]
    print('image : ', img.size())
    print('ceiling-wall and floor-wall boundary veto: ', boundary_lst.size())
    print('wall-wall probability vector: ', wall_y_prob_lst.size())

    b_vis_mesh_from_corners = False
    b_vis_mesh_from_boundary_lst = True
    
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
    #         elif b_vis_mesh_from_boundary_lst:
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
        scene_sample_label = sample_result_lst['arr_1'][idx]
        scene_sample_label = [key for key in ROOM_TYPE_DICT.keys() if ROOM_TYPE_DICT[key] == scene_sample_label][0]
        scene_sample_label = scene_sample_label.replace(' ', '_')
        print(f'scene_sample_label: {scene_sample_label}')

        # save results ply
        boundary_lst = scene_sample_result[:2, :]
        wall_prob_lst = scene_sample_result[2, :]

        # save synthetic boundaries as image
        img_fname = f'{scene_sample_label}_{idx}.png'
        out_img = visualize_synth_data(bound_y_lst=boundary_lst, corner_y_lst=wall_prob_lst)
        save_img_filepath = os.path.join(args.out_dir, img_fname)
        Image.fromarray(out_img).save(save_img_filepath)

        if b_vis_mesh_from_corners:
            layout_ply_points, layout_ply_faces, layout_corner_lst, cam_pos_lst = dataset.get_gt_layout_mesh(idx)
        elif b_vis_mesh_from_boundary_lst:
            layout_ply_points, layout_ply_faces, layout_corner_lst, cam_pos_lst = dataset.get_layout_mesh_from_prediction(bound_ceil_floor_lst=boundary_lst, 
                                                                                                                            wall_prob_lst=wall_prob_lst, b_force_raw=True)
        print('layout_ply_points: ', layout_ply_points.shape)
        print('layout_ply_faces: ', layout_ply_faces.shape)
        print('layout_corner_lst: ', layout_corner_lst.shape)
        print('cam_pos_lst: ', cam_pos_lst)
        ply_fname = f'{scene_sample_label}_{idx}.ply'
        save_ply_filepath = os.path.join(args.out_dir, ply_fname)
        save_layout_mesh(save_ply_filepath, layout_ply_points, layout_ply_faces)