import cv2
import numpy as np
from sklearn.preprocessing import normalize
from scipy.spatial.transform import Rotation

def interpolate_line(p1, p2, num=30):
    t = np.expand_dims(np.linspace(0, 1, num=num, dtype=np.float32), 1)
    points = p1 * (1 - t) + t * p2
    return points

def cam3d2rad(cam3d):
    """
    Transform 3D points in camera coordinate to longitude and latitude.

    Parameters
    ----------
    cam3d: n x 3 numpy array or bdb3d dict

    Returns
    -------
    n x 2 numpy array of longitude and latitude in radiation
    first rotate left-right, then rotate up-down
    longitude: (left) -pi -- 0 --> +pi (right)
    latitude: (up) -pi/2 -- 0 --> +pi/2 (down)
    """
    backend, atan2 = (np, np.arctan2)
    lon = atan2(cam3d[..., 0], cam3d[..., 1])
    # lat = backend.arcsin(cam3d[..., 1] / backend.linalg.norm(cam3d, axis=-1))
    lat = backend.arccos(
        cam3d[..., 2] / backend.linalg.norm(cam3d, axis=-1)) - np.pi/2
    return backend.stack([lon, lat], -1)


def camrad2pix(camrad):
    """
    Transform longitude and latitude of a point to panorama pixel coordinate.

    Parameters
    ----------
    camrad: n x 2 numpy array

    Returns
    -------
    n x 2 numpy array of xy coordinate in pixel
    x: (left) 0 --> (width - 1) (right)
    y: (up) 0 --> (height - 1) (down)
    """
    # if 'K' in self.camera:
    #     raise NotImplementedError
    # if isinstance(camrad, torch.Tensor):
    #     campix = torch.empty_like(camrad, dtype=torch.float32)
    # else:
    campix = np.empty_like(camrad, dtype=np.float32)
    width, height = IMG_WIDTH, IMG_HEIGHT
    # if isinstance(camrad, torch.Tensor):
    #     width, height = [x.view([-1] + [1] * (camrad.dim() - 2))
    #                      for x in (width, height)]
    campix[..., 0] = camrad[..., 0] * width / (2. * np.pi) + width / 2. + 0.5
    campix[..., 1] = camrad[..., 1] * height / np.pi + height / 2. + 0.5
    return campix


def cam3d2pix(cam3d):
    """
    Transform 3D points from camera coordinate to pixel coordinate.

    Parameters
    ----------
    cam3d: n x 3 numpy array or bdb3d dict

    Returns
    -------
    for 3D points: n x 2 numpy array of xy in pixel.
    x: (left) 0 --> width - 1 (right)
    y: (up) 0 --> height - 1 (down)
    """
    # if isinstance(cam3d, dict):
    #     campix = self.world2campix(self.cam3d2world(cam3d))
    # else:
    #     if 'K' in self.camera:
    #         campix = self.transform(self.camera['K'], cam3d)
    #     else:
    campix = camrad2pix(cam3d2rad(cam3d))
    return campix


def obj2frame(point, bdb3d):
    """
    Transform 3D points or Trimesh from normalized object coordinate frame to coordinate frame bdb3d is in.
    object: x-left, y-back, z-up (defined by iGibson)
    world: right-hand coordinate of iGibson (z-up)

    Parameters
    ----------
    point: n x 3 numpy array or Trimesh
    bdb3d: dict, self['objs'][id]['bdb3d']

    Returns
    -------
    n x 3 numpy array or Trimesh
    """
    # if isinstance(obj, trimesh.Trimesh):
    #     obj = obj.copy()
    #     normalized_vertices = normalize_to_unit_square(obj.vertices, keep_ratio=False)[0]
    #     obj_vertices = normalized_vertices / 2
    #     obj.vertices = IGTransform.obj2frame(obj_vertices, bdb3d)
    #     return obj
    # if isinstance(obj, torch.Tensor):
    #     size = bdb3d['size'].unsqueeze(-2)
    #     centroid = bdb3d['centroid'].unsqueeze(-2)
    #     return (bdb3d['basis'] @ (obj * size).transpose(-1, -2)).transpose(-1, -2) + centroid
    # else:
    rotation = Rotation.from_euler(
        'zyx', [bdb3d['rotations']['z'], bdb3d['rotations']['y'], bdb3d['rotations']['x']], degrees=True).as_matrix()
    centroid = np.array(
        [-bdb3d['centroid']['x'], bdb3d['centroid']['y'], bdb3d['centroid']['z']])
    sizes = np.array([bdb3d['dimensions']['length'],
                      bdb3d['dimensions']['width'], bdb3d['dimensions']['height']])
    return (rotation @ (point * sizes).T).T + centroid


def bdb3d_corners(bdb3d: (dict, np.ndarray)):
    """
    Get ordered corners of given 3D bounding box dict or disordered corners

    Parameters
    ----------
    bdb3d: 3D bounding box dict

    Returns
    -------
    8 x 3 numpy array of bounding box corner points in the following order:
    right-forward-down
    left-forward-down
    right-back-down
    left-back-down
    right-forward-up
    left-forward-up
    right-back-up
    left-back-up
    """
    if isinstance(bdb3d, np.ndarray):
        centroid = np.mean(bdb3d, axis=0)
        z = bdb3d[:, -1]
        surfaces = []
        for surface in (bdb3d[z < centroid[-1]], bdb3d[z >= centroid[-1]]):
            surface_2d = surface[:, :2]
            center_2d = centroid[:2]
            vecters = surface_2d - center_2d
            angles = np.arctan2(vecters[:, 0], vecters[:, 1])
            orders = np.argsort(-angles)
            surfaces.append(surface[orders][(0, 1, 3, 2), :])
        corners = np.concatenate(surfaces)
    else:
        corners = np.unpackbits(np.arange(8, dtype=np.uint8)[..., np.newaxis],
                                axis=1, bitorder='little', count=-5).astype(np.float32)
        corners = corners - 0.5
        # if isinstance(bdb3d['size'], torch.Tensor):
        #     corners = torch.from_numpy(corners).to(bdb3d['size'].device)
        corners = obj2frame(corners, bdb3d)
    return corners



def wrapped_line(image, p1, p2, colour, thickness, lineType=cv2.LINE_AA):
    if p1[0] > p2[0]:
        p1, p2 = p2, p1

    _p1 = np.array(p1)
    _p2 = np.array(p2)

    dist1 = np.linalg.norm(_p1 - _p2)

    p1b = np.array([p1[0]+image.shape[1], p1[1]])
    p2b = np.array([p2[0]-image.shape[1], p2[1]])

    dist2 = np.linalg.norm(_p1 - p2b)

    if dist1 < dist2:
        cv2.line(image, p1, p2, colour, thickness, lineType=lineType)
    else:
        cv2.line(image, p1, tuple(p2b), colour, thickness, lineType=lineType)
        cv2.line(image, tuple(p1b), p2, colour, thickness, lineType=lineType)



# visualize 3dbbox on panorama
def vis_objs3d(image, v_bbox3d, b_show_axes=False, b_show_centroid=False, b_show_bbox3d=True, b_show_info=False, thickness=2):

    def draw_line3d(image, p1, p2, color, thickness, quality=30, frame='world'):
        color = (np.ones(3, dtype=np.uint8) * color).tolist()
        if frame == 'world':
            print('input points must be in camera frame')
        elif frame != 'cam3d':
            raise NotImplementedError
        points = interpolate_line(p1, p2, quality)
        normal_points = normalize(points)
        pix = np.round(cam3d2pix(normal_points)).astype(np.int32)
        for t in range(quality - 1):
            p1, p2 = pix[t], pix[t + 1]
            wrapped_line(image, tuple(p1), tuple(p2), color,
                         thickness, lineType=cv2.LINE_AA)

    def draw_objaxes(image, centroid, sizes, rotation, thickness=2):

        for axis in np.eye(3, dtype=np.float32):
            endpoint = rotation @ ((axis / 2) * sizes) + centroid
            color = axis * 255
            draw_line3d(image, centroid, endpoint,
                        color, thickness, frame='cam3d')

    def draw_centroid(image, centroid, color, thickness=2):
        color = (np.ones(3, dtype=np.uint8) * color).tolist()
        normal_centroid = centroid/np.linalg.norm(centroid)
        center = cam3d2pix(normal_centroid)
        cv2.circle(image, tuple(center.astype(np.int32).tolist()),
                   5, color, thickness=thickness, lineType=cv2.LINE_AA)

    def draw_bdb3d(image, bdb3d, color, thickness=2):
        corners = bdb3d_corners(bdb3d)
        corners_box = corners.reshape(2, 2, 2, 3)
        for k in [0, 1]:
            for l in [0, 1]:
                for idx1, idx2 in [((0, k, l), (1, k, l)), ((k, 0, l), (k, 1, l)), ((k, l, 0), (k, l, 1))]:
                    draw_line3d(
                        image, corners_box[idx1], corners_box[idx2], color, thickness=thickness, frame='cam3d')
        for idx1, idx2 in [(0, 5), (1, 4)]:
            draw_line3d(image, corners[idx1], corners[idx2],
                        color, thickness=thickness, frame='cam3d')

    def draw_objinfo(image, bdb3d_centeroid, obj_cls_name, color):
        color = [255 - c for c in color]
        normal_centroid = bdb3d_centeroid/np.linalg.norm(bdb3d_centeroid)
        bdb3d_pix = cam3d2pix(normal_centroid)
        bottom_left = bdb3d_pix.astype(np.int32)
        bottom_left[1] -= 16
        cv2.putText(image, obj_cls_name, tuple(bottom_left.tolist()),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    image = image.copy()
    dis = [np.linalg.norm([-o['centroid']['x'], o['centroid']
                           ['y'], o['centroid']['z']]) for o in v_bbox3d]
    i_objs = sorted(range(len(dis)), key=lambda k: dis[k])
    for i_obj in reversed(i_objs):
        bdb3d = v_bbox3d[i_obj]
        obj_label = bdb3d['name'][0:bdb3d['name'].rfind('_')]

        obj_cls_id = ReplicaXRDatasetConfig().type2class[obj_label]
        color = (replicapano_colorbox[obj_cls_id]
                 * 255).astype(np.uint8).tolist()
        centroid = np.array(
            [-bdb3d['centroid']['x'], bdb3d['centroid']['y'], bdb3d['centroid']['z']])
        sizes = np.array([bdb3d['dimensions']['length'],
                          bdb3d['dimensions']['width'], bdb3d['dimensions']['height']])
        bdb3d['rotations']['z'] *= -1
        rotation = Rotation.from_euler(
            'zyx', [bdb3d['rotations']['z'], bdb3d['rotations']['y'], bdb3d['rotations']['x']], degrees=True).as_matrix()

        if b_show_axes:
            draw_objaxes(image, centroid, sizes, rotation, thickness=thickness)
        if b_show_centroid:
            draw_centroid(image, centroid,
                          color, thickness=thickness)
        if b_show_bbox3d:
            draw_bdb3d(image, bdb3d, color, thickness=thickness)
        if b_show_info:
            draw_objinfo(image, centroid, obj_label, color)
    return image