import os
import torch
import torch.nn.functional as F
import json
import numba
import numpy as np
from collections import defaultdict
from .base import DatasetBase
from ..utils.lidar import LiDARUtility

@numba.jit(nopython=True, parallel=False)
def scatter(array, index, value):
    for (h, w), v in zip(index, value):
        array[h, w] = v
    return array

class NuScenesGen(DatasetBase):
    def __init__(self, project='spherical', **kwargs):
        super().__init__(**kwargs)
        self.project = project
        self.lidar_utils = LiDARUtility(
            self.img_size,
            depth_format='log_depth',
            min_depth=self.depth_range[0],
            max_depth=self.depth_range[1],
        )
        self.prepare_data()

    def prepare_data(self):
        if self.split == 'train':
            with open(os.path.join(self.data_root, 'v1.0-trainval/v1.0-trainval/sample_data.json')) as f:
                sample_data = json.load(f)
        else:
            with open(os.path.join(self.data_root, 'v1.0-trainval/v1.0-mini/sample_data.json')) as f:
                sample_data = json.load(f)

        custom_path = 'v1.0-trainval'
        file_paths = [os.path.join(self.data_root, custom_path, x['filename']) 
                           for x in sample_data 
                           if 'samples/LIDAR_TOP' in x['filename']]
        self.data = sorted(file_paths)

    def load_points_as_images(
        self,
        point_path: str,
    ):
        # load xyz & intensity and add depth & mask
        points = self.load_lidar_sweep(point_path)[:,:4]
        xyz = points[:, :3]  # xyz
        x = xyz[:, [0]]
        y = xyz[:, [1]]
        z = xyz[:, [2]]
        depth = np.linalg.norm(xyz, ord=2, axis=1, keepdims=True)
        mask = (depth >= self.depth_range[0]) & (depth <= self.depth_range[1])
        points = np.concatenate([points, depth, mask], axis=1)

        if self.project == 'scan_unfolding':
            # the i-th quadrant
            # suppose the points are ordered counterclockwise
            quads = np.zeros_like(x, dtype=np.int32)
            quads[(x >= 0) & (y >= 0)] = 0  # 1st
            quads[(x < 0) & (y >= 0)] = 1  # 2nd
            quads[(x < 0) & (y < 0)] = 2  # 3rd
            quads[(x >= 0) & (y < 0)] = 3  # 4th

            # split between the 3rd and 1st quadrants
            diff = np.roll(quads, shift=1, axis=0) - quads
            delim_inds, _ = np.where(diff == 3)  # number of lines
            inds = list(delim_inds) + [len(points)]  # add the last index

            # vertical grid
            grid_h = np.zeros_like(x, dtype=np.int32)
            cur_ring_idx = self.img_size[0] - 1  # ...0
            for i in reversed(range(len(delim_inds))):
                grid_h[inds[i] : inds[i + 1]] = cur_ring_idx
                if cur_ring_idx >= 0:
                    cur_ring_idx -= 1
                else:
                    break
        else:
            h_up, h_down = np.deg2rad(self.fov[0]), np.deg2rad(self.fov[1])
            elevation = np.arcsin(z / depth) + abs(h_down)
            grid_h = 1 - elevation / (h_up - h_down)
            grid_h = np.floor(grid_h * self.img_size[0]).clip(0, self.img_size[0] - 1).astype(np.int32)

        # horizontal grid
        azimuth = -np.arctan2(y, x)  # [-pi,pi]
        grid_w = (azimuth / np.pi + 1) / 2 % 1  # [0,1]
        grid_w = np.floor(grid_w * self.img_size[1]).clip(0, self.img_size[1] - 1).astype(np.int32)

        grid = np.concatenate((grid_h, grid_w), axis=1)

        # projection
        order = np.argsort(-depth.squeeze(1))
        proj_points = np.zeros((self.img_size[0], self.img_size[1], 4 + 2), dtype=points.dtype)
        proj_points = scatter(proj_points, grid[order], points[order])

        return proj_points.astype(np.float32)

    def load_lidar_sweep(self, path):
        scan = np.fromfile(path, dtype=np.float32).reshape((-1, 5))
        # get xyz & intensity & timestamp
        return scan
    
    def load_semantic_map(self, path, pcd):
        raise NotImplementedError

    def load_camera(self, path):
        raise NotImplementedError

    def get_pth_path(self, pts_path):
        return pts_path.replace('sweeps', 'sweeps_range').replace('.bin', '.pth')

    def process_remission(selef, range_feature):
        range_feature = np.clip(range_feature, 0, 1.0)
        range_feature = np.expand_dims(range_feature, axis = 0)
        return range_feature

    def __getitem__(self, idx):
        data_path = self.data[idx]
        # lidar point cloud
        xyzrdm = self.load_points_as_images(data_path)
        xyzrdm = xyzrdm.transpose(2, 0, 1)
        xyzrdm *= xyzrdm[[5]]
        return {
            "xyz": xyzrdm[:3],
            "reflectance": xyzrdm[[3]]/255,
            "depth": xyzrdm[[4]],
            "mask": xyzrdm[[5]],
        }
    
    def post_preprocess(self, batch):
        x = []
        x += [self.lidar_utils.convert_depth(batch["depth"])]
        x += [batch["reflectance"]]
        x = torch.cat(x, dim=1)
        x = self.lidar_utils.normalize(x)
        x = F.interpolate(
            x.to('cpu'),
            size=(self.img_size[0], self.img_size[1]),
            mode="nearest-exact",
        )
        batch['image'] = x
        return batch

    def collate_fn(self, batch_list, _unused=False):
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

        ret['batch_size'] = batch_size * batch_size_ratio
        ret = self.post_preprocess(ret)
        return ret