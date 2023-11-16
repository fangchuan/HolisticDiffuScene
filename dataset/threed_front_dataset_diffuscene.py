import os
import sys
import numpy as np
from PIL import Image

from typing import Any, List, Dict, Tuple
# import imghdr
import json

import torch.utils.data as data

from .threed_front.metadata import THREED_FRONT_BEDROOM_WO_DOOR_WINDOW_WALL_FURNITURE, THREED_FRONT_BEDROOM_FURNITURE_CNTS, \
    THREED_FRONT_LIVINGROOM_WO_DOOR_WINDOW_WALL_FURNITURE, THREED_FRONT_DININGROOM_FURNITURE_WO_DOOR_WINDOW_WALL, THREED_FRONT_LIVINGROOM_FURNITURE_CNTS, \
    THREED_FRONT_BEDROOM_MIN_FURNITURE_NUM, THREED_FRONT_BEDROOM_MAX_FURNITURE_NUM, \
    THREED_FRONT_LIVINGROOM_MIN_FURNITURE_NUM, THREED_FRONT_LIVINGROOM_MAX_FURNITURE_NUM, \
    THREED_FRONT_BEDROOM_MAX_WALL_NUM, THREED_FRONT_LIVINGROOM_MAX_WALL_NUM

# room types
ROOM_TYPE_DICT = {
    'living room': 0,
    'kitchen': 1,
    'bedroom': 2,
    'bathroom': 3,
    'balcony': 4,
    'corridor': 5,
    'dining room': 6,
    'study': 7,
    'studio': 8,
    'store room': 9,
    'garden': 10,
    'laundry room': 11,
    'office': 12,
    'basement': 13,
    'garage': 14,
    'undefined': 15
}

from .threed_front.utils import CSVSplitsBuilder


def padding_and_reshape_bbox(room_type: int, bbox_lst: np.array, bbox_dim: int) -> List:
    """ Implement the padding for the quad wall boxes.
    Args:
        room_type (int): The room type.
        bbox_lst (np.array): The quadwall bounding box list.
        bbox_dim (int): The dimension of the quadwallbounding box.
    Returns:
        _type_: _description_
    """
    L = len(bbox_lst)
    if room_type == ROOM_TYPE_DICT['bedroom']:
        class_num = len(THREED_FRONT_BEDROOM_WO_DOOR_WINDOW_WALL_FURNITURE)
        max_len = THREED_FRONT_BEDROOM_MAX_FURNITURE_NUM
    elif room_type == ROOM_TYPE_DICT['living room']:
        class_num = len(THREED_FRONT_LIVINGROOM_WO_DOOR_WINDOW_WALL_FURNITURE)
        max_len = THREED_FRONT_LIVINGROOM_MAX_FURNITURE_NUM
    elif room_type == ROOM_TYPE_DICT['dining room']:
        class_num = len(THREED_FRONT_DININGROOM_FURNITURE_WO_DOOR_WINDOW_WALL)
        max_len = THREED_FRONT_LIVINGROOM_MAX_FURNITURE_NUM

    assert L <= max_len, 'The length of the wall bbox list should be less than 10.'

    # Pad the end label in the end of each sequence, and convert the class labels to -1, 1
    empty_label = np.eye(class_num)[-1] * 2 - 1
    padding = np.concatenate([empty_label, np.zeros(bbox_dim - class_num, dtype=np.float32)], axis=0)
    bbox_lst = np.vstack([bbox_lst, np.tile(padding, [max_len - L, 1])])

    return bbox_lst


class RotationAugmentation(object):

    def __init__(self, dataset, min_rad=0.174533, max_rad=5.06145, fixed=False):
        super().__init__(dataset)
        self._min_rad = min_rad
        self._max_rad = max_rad
        self._fixed = fixed

    @staticmethod
    def rotation_matrix_around_y(theta):
        R = np.zeros((3, 3))
        R[0, 0] = np.cos(theta)
        R[0, 2] = -np.sin(theta)
        R[2, 0] = np.sin(theta)
        R[2, 2] = np.cos(theta)
        R[1, 1] = 1.
        return R

    @property
    def rot_angle(self):
        if np.random.rand() < 0.5:
            return np.random.uniform(self._min_rad, self._max_rad)
        else:
            return 0.0

    @property
    def fixed_rot_angle(self):
        if np.random.rand() < 0.25:
            return np.pi * 1.5
        elif np.random.rand() < 0.50:
            return np.pi
        elif np.random.rand() < 0.75:
            return np.pi * 0.5
        else:
            return 0.0

    def __getitem__(self, idx):
        # Get the rotation matrix for the current scene
        if self._fixed:
            rot_angle = self.fixed_rot_angle
        else:
            rot_angle = self.rot_angle
        R = RotationAugmentation.rotation_matrix_around_y(rot_angle)

        sample_params = self._dataset[idx]
        for k, v in sample_params.items():
            if k == "translations":
                sample_params[k] = v.dot(R)
            elif k == "angles":
                angle_min, angle_max = self.bounds["angles"]
                sample_params[k] = \
                    (v + rot_angle - angle_min) % (2 * np.pi) + angle_min
            elif k == "room_layout":
                # Fix the ordering of the channels because it was previously
                # changed
                img = np.transpose(v, (1, 2, 0))
                sample_params[k] = np.transpose(rotate(img, rot_angle * 180 / np.pi, reshape=False), (2, 0, 1))
        return sample_params


class ThreedFrontDataset(data.Dataset):
    '''
    dataset for ThreeD-Front
    '''

    def __init__(
            self,
            config: Dict[str, Any],
            room_type='bedroom',
            is_train=True,
            is_test=False,
            rot_augmentation=True,  #  max number of text_prompt sentences
            shard=0,  #  support parallel training
            num_shards=1):

        self._base_dir = config["data"]["dataset_directory"]
        self.config = config
        self.room_type = room_type
        self.is_train = is_train
        self.is_test = is_test
        self.rot_augmentation = rot_augmentation

        self._parse_train_stats(config["data"]["train_stats"])

        # Make the train/test/validation splits
        splits_builder = CSVSplitsBuilder(config["data"]["annotation_file"])
        split_scene_ids = splits_builder.get_splits(
            keep_splits=config["training"].get("splits", ["train", "val"])) if is_train \
                  else splits_builder.get_splits(keep_splits=config["validation"].get("splits", ["test"]))

        # rooms liist
        self._tags_lst = sorted([
            oi for oi in os.listdir(self._base_dir) if oi.split("_")[1] in split_scene_ids
            if os.path.exists(os.path.join(self._base_dir, oi))
        ])
        # bbox npz files
        self._path_to_rooms_lst = sorted([
            os.path.join(self._base_dir, pi, "boxes.npz")
            for pi in self._tags_lst
            if os.path.exists(os.path.join(self._base_dir, pi, "boxes.npz"))
        ])
        # text prompt files
        self._path_to_rooms_text_lst = sorted([
            os.path.join(self._base_dir, pi, "text_prompt.txt")
            for pi in self._tags_lst
            if os.path.exists(os.path.join(self._base_dir, pi, "text_prompt.txt"))
        ])
        assert len(self._tags_lst) == len(self._path_to_rooms_lst), "Number of scenes and boxes.npz files do not match"
        assert len(self._tags_lst) == len(
            self._path_to_rooms_text_lst), "Number of scenes and text_prompt.txt files do not match"

        rendered_scene = "rendered_scene_256.png"
        path_to_rendered_scene = os.path.join(self._base_dir, self._tags_lst[0], rendered_scene)
        if not os.path.isfile(path_to_rendered_scene):
            rendered_scene = "rendered_scene_256_no_lamps.png"

        self._path_to_renders = sorted([os.path.join(self._base_dir, pi, rendered_scene) for pi in self._tags_lst])

        self.local_tags_lst = self._tags_lst[shard:][::num_shards]
        self.local_path_to_rooms_lst = self._path_to_rooms_lst[shard:][::num_shards]
        self.local_path_to_renders = self._path_to_renders[shard:][::num_shards]
        self.local_path_to_rooms_text_lst = self._path_to_rooms_text_lst[shard:][::num_shards]

        # pre-load all data
        self.local_rooms_dict_lst = []
        self.local_rooms_text_lst = []
        self._preload_()

    def _get_room_layout(self, room_layout):
        # Resize the room_layout if needed
        img = Image.fromarray(room_layout[:, :, 0])
        img = img.resize(tuple(map(int, self.config["data"]["room_layout_size"].split(","))), resample=Image.BILINEAR)
        D = np.asarray(img).astype(np.float32) / np.float32(255)
        return D

    def get_room_params(self, i):
        D = np.load(self._path_to_rooms_lst[i])

        room = self._get_room_layout(D["room_layout"])
        room = np.transpose(room[:, :, None], (2, 0, 1))
        # room_text = self._get_room_text(str(D["scene_uid"]))
        room_text_emb = D["desc_emb"]
        print(f'room_text_emb: {room_text_emb.shape}')

        return {
            "room_layout": room,
            "room_text_emb": room_text_emb,
            "class_labels": D["class_labels"],
            "translations": D["translations"],
            "sizes": D["sizes"],
            "angles": D["angles"]
        }

    def __len__(self):
        return len(self.local_path_to_rooms_lst)

    def _parse_train_stats(self, train_stats):
        with open(os.path.join(self._base_dir, train_stats), "r") as f:
            train_stats = json.load(f)
        self._centroids = train_stats["bounds_translations"]
        self._centroids = (np.array(self._centroids[:3]), np.array(self._centroids[3:]))
        self._sizes = train_stats["bounds_sizes"]
        self._sizes = (np.array(self._sizes[:3]), np.array(self._sizes[3:]))
        self._angles = train_stats["bounds_angles"]
        self._angles = (np.array(self._angles[0]), np.array(self._angles[1]))

        self._class_labels = train_stats["class_labels"]
        self._object_types = train_stats["object_types"]
        self._class_frequencies = train_stats["class_frequencies"]
        self._class_order = train_stats["class_order"]
        self._count_furniture = train_stats["count_furniture"]

    def _preload_(self):
        """pre-load all data
        """
        for idx in range(len(self)):
            # load npz data
            npz_data = np.load(self.local_path_to_rooms_lst[idx])
            self.local_rooms_dict_lst.append(npz_data)
            # load text prompt file
            with open(self.local_path_to_rooms_text_lst[idx], "r") as f:
                text = f.read().strip()
                self.local_rooms_text_lst.append(text)

    @property
    def class_labels(self):
        return self._class_labels

    @property
    def object_types(self):
        return self._object_types

    @property
    def class_frequencies(self):
        return self._class_frequencies

    @property
    def class_order(self):
        return self._class_order

    @property
    def count_furniture(self):
        return self._count_furniture

    def _check_dataset(self):
        for scene_tag in self._tags_lst:
            assert os.path.isfile(os.path.join(self._base_dir, scene_tag, "boxes.npz")), \
                "boxes.npz not found in {}".format(os.path.join(self._base_dir, scene_tag))

    @staticmethod
    def scale(x, minimum, maximum):
        X = x.astype(np.float32)
        X = np.clip(X, minimum, maximum)
        X = ((X - minimum) / (maximum - minimum))
        X = 2 * X - 1
        return X

    @staticmethod
    def descale(x, minimum, maximum):
        x = (x + 1) / 2
        x = x * (maximum - minimum) + minimum
        return x

    def post_process(self, samples: np.ndarray):
        """ post process the samples in the room

        Args:
            samples (np.ndarray): sampled furniture and walls in the room
        """
        N, C = samples.shape
        center_bounds = self._centroids
        size_bounds = self._sizes

        center_dim = 3
        size_dim = 3
        angle_dim = 2
        class_label_dim = len(self._class_labels)

        new_samples = []
        for i in range(N):
            # descale class labels
            class_labels = samples[i, :class_label_dim]
            descaled_class_labels = (class_labels + 1) / 2
            class_label_prob = np.where(descaled_class_labels > 0.5, 1, 0)
            class_label = self.class_labels[class_label_prob.argmax()]
            if class_label == 'end':
                continue

            # descale center
            center = samples[i, class_label_dim:class_label_dim + center_dim]
            descaled_centers = self.descale(center, *center_bounds)

            # descale size
            size = samples[i, class_label_dim + center_dim:class_label_dim + center_dim + size_dim]
            descaled_sizes = self.descale(size, *size_bounds)

            # cvt cos,sin to angle
            cos_sin_angle = samples[i, class_label_dim + center_dim + size_dim:class_label_dim + center_dim + size_dim +
                                    angle_dim]
            angles = np.arctan2(cos_sin_angle[1:2], cos_sin_angle[0:1])

            # concatenate
            descaled_samples = np.concatenate([descaled_class_labels, descaled_centers, descaled_sizes, angles],
                                              axis=-1)
            new_samples.append(descaled_samples)
        return np.array(new_samples)

    def __getitem__(self, idx: int):
        """retrieve scene data

        Args:
            idx (int): panorama/room idx

        Returns:
            List: 
        """
        # D = np.load(self.local_path_to_rooms_lst[idx])
        D = self.local_rooms_dict_lst[idx]
        text_prompt = self.local_rooms_text_lst[idx]
        # print('text_prompt: ', text_prompt)

        # room_layout_img = self._get_room_layout(D["room_layout"])
        # room_layout_img = np.transpose(room_layout_img[:, :, None], (2, 0, 1))
        room_text_emb = D["desc_emb"].squeeze(0).astype(np.float32)
        # print(f'room_text_emb: {room_text_emb.shape}')

        # "room_text_emb": room_text_emb,
        # "class_labels": D["class_labels"],
        # "translations": D["translations"],
        # "sizes": D["sizes"],
        # "angles": D["angles"]
        bbox_onehot_class_labels = D["class_labels"]
        # print(f'bbox_onehot_class_labels: {bbox_onehot_class_labels.shape}')
        bbox_trans = D["translations"]
        # print(f'bbox_translations: {bbox_trans.shape}')
        bbox_sizes = D["sizes"]
        # print(f'bbox_sizes: {bbox_sizes.shape}')
        bbox_angles = D["angles"]
        # print(f'bbox_angles: {bbox_angles.shape}')

        # data augmentation
        if self.rot_augmentation:
            rot_angle = np.random.uniform(0.0, np.pi)
            R = RotationAugmentation.rotation_matrix_around_y(rot_angle)

            # rotate translations
            bbox_trans = bbox_trans.dot(R)

            # rotate angles
            angle_min, angle_max = self._angles
            bbox_angles = (bbox_angles + rot_angle - angle_min) % (2 * np.pi) + angle_min

        # encode angles as [cosin, sin]
        bbox_cos_sin_angles = np.concatenate([np.cos(bbox_angles), np.sin(bbox_angles)], axis=-1)
        # scale properties to -1 ~ 1
        scaled_class_labels = bbox_onehot_class_labels * 2 - 1
        scaled_trans = self.scale(bbox_trans, *self._centroids)
        scaled_size = self.scale(bbox_sizes, *self._sizes)

        # assert bbox_onehot_class_labels.shape[0] == bbox_trans.shape[0] == bbox_sizes.shape[
        #     0] == bbox_cos_sin_angles.shape[0]
        out_lst = np.concatenate([scaled_class_labels, scaled_trans, scaled_size, bbox_cos_sin_angles], axis=-1)
        # print(f'concatenated out_lst: {out_lst.shape}')

        # pad to max_length
        bbox_dim = out_lst.shape[-1]
        out_lst = padding_and_reshape_bbox(room_type=ROOM_TYPE_DICT[self.room_type],
                                           bbox_lst=out_lst,
                                           bbox_dim=bbox_dim)
        # print(f'padding out_lst: {out_lst.shape}')
        # print(f'out_lst: {out_lst}')
        out_lst = out_lst.transpose(1, 0)

        cond_dict = {}
        # if room_type is not None:
        #     cond_dict["y"] = np.array(room_type, dtype=np.int64)

        cond_dict["text"] = text_prompt
        cond_dict["context"] = room_text_emb
        if self.is_test:
            return out_lst, cond_dict, self.local_tags_lst[idx]
        else:
            return out_lst, cond_dict
