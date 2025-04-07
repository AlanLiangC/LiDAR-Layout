import os
import json
import numpy as np
from torch.utils.data import Dataset
from collections import defaultdict
from ..utils.aug_utils import get_lidar_transform, mask_points_by_range

import torch

class NUSC_CUBE_DATASET(Dataset):
    def __init__(self, data_root, split='train', dataset_config=None, aug_config=None, **kwargs):
        self.data_root = data_root
        self.split = split
        self.data_config = dataset_config
        self.point_cloud_range = dataset_config.point_cloud_range
        self.lidar_transform = get_lidar_transform(aug_config, split)
        self.prepare_data()

    def prepare_data(self):
        if self.split == 'train':
            with open(os.path.join(self.data_root, 'v1.0-trainval/v1.0-trainval/sample_data.json')) as f:
                sample_data = json.load(f)

            custom_path = 'v1.0-trainval'
            file_paths = [os.path.join(self.data_root, custom_path, x['filename']) 
                            for x in sample_data 
                            if 'sweeps/LIDAR_TOP' in x['filename']]
            self.data = sorted(file_paths)
        elif self.split == 'val':
            with open(os.path.join(self.data_root, 'v1.0-trainval/v1.0-mini/sample_data.json')) as f:
                sample_data = json.load(f)

            custom_path = 'v1.0-trainval'
            file_paths = [os.path.join(self.data_root, custom_path, x['filename']) 
                            for x in sample_data 
                            if 'samples/LIDAR_TOP' in x['filename']]
            self.data = sorted(file_paths)

    def __len__(self):
        return len(self.data)
    
    @staticmethod
    def load_lidar_sweep(path):
        scan = np.fromfile(path, dtype=np.float32).reshape((-1, 5))
        return scan[:, 0:3]

    def points2coords(self, points):
        pass

    def __getitem__(self, index):
        batch = dict()
        data_path = self.data[index]
        sweep = self.load_lidar_sweep(data_path)

        if self.lidar_transform:
            sweep, _ = self.lidar_transform(sweep, None)

        mask = mask_points_by_range(sweep, self.point_cloud_range)
        batch['points_for_cube'] = sweep[mask]
        return batch
    
    @staticmethod
    def collate_fn(batch_list, _unused=False):
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret = {}
        batch_size_ratio = 1

        for key, val in data_dict.items():
            try:
                if key in ['gt_boxes', 'layout']:
                    max_gt = max([len(x) for x in val])
                    batch_gt_boxes3d = np.zeros((batch_size, 13, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        if val[k].__len__() <= 13:
                            batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
                        else:
                            batch_gt_boxes3d[k] = val[k][:13]
                    ret[key] = batch_gt_boxes3d
                elif key in ['reproj']:
                    ret[key] = val
                elif key in ['points', 'voxel_coords']:
                    coors = []
                    if isinstance(val[0], list):
                        val =  [i for item in val for i in item]
                    for i, coor in enumerate(val):
                        coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                        coors.append(coor_pad)
                elif key in ['points_for_cube']:
                    coors = []
                    max_n_input_points = max([item.shape[0] for item in val])
                    for i, coor in enumerate(val):
                        coor_pad = np.pad(coor, ((0, max_n_input_points - coor.shape[0]), (0, 0)), mode='constant', constant_values=float("nan"))
                        coors.append(coor_pad)
                    ret[key] = np.stack(coors, axis=0)
                    # coors = []
                    # if isinstance(val[0], list):
                    #     val =  [i for item in val for i in item]
                    # for i, coor in enumerate(val):
                    #     coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                    #     coors.append(coor_pad)
                    # ret[key] = np.concatenate(coors, axis=0)
                else:
                    ret[key] = np.stack(val, axis=0)
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError
            
        for key, value in ret.items():
            try:
                ret[key] = torch.from_numpy(value).float()
            except:
                ret[key] = value

        return ret