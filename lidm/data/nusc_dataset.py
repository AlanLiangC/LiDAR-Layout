import os
import json
import copy
import torch
import pickle
import numpy as np
from collections import defaultdict
from .base import DatasetBase
from ..utils.lidar_utils import pcd2range, box2coord2dx2, range2pcd

class nuScenesBase(DatasetBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataset_name = 'nuScenes'
        self.num_sem_cats = kwargs['dataset_config'].num_sem_cats + 1 # 17
        self.return_remission = (kwargs['dataset_config'].num_channels == 2)

    @staticmethod
    def load_lidar_sweep(path):
        scan = np.fromfile(path, dtype=np.float32).reshape((-1, 5))
        # get xyz & intensity
        return scan[:, 0:3]
    
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
        example = dict()
        data_path = self.data[idx]
        # lidar point cloud
        sweep = self.load_lidar_sweep(data_path)

        if self.lidar_transform:
            sweep, _ = self.lidar_transform(sweep, None)

        if self.condition_key == 'segmentation':
            # semantic maps
            proj_range, sem_map = self.load_semantic_map(data_path, sweep)
            example[self.condition_key] = sem_map
        else:
            proj_range, proj_feature = pcd2range(sweep[:,:3], self.img_size, self.fov, self.depth_range,remission=sweep[:,-1])
        proj_range, proj_mask = self.process_scan(proj_range)

        if self.return_remission:
            proj_feature = self.process_remission(proj_feature)
            proj_range = np.concatenate((proj_range, proj_feature), axis = 0)

        example['image'], example['mask'] = proj_range, proj_mask
        if self.return_pcd:
            reproj_sweep, _, _ = range2pcd(proj_range[0] * .5 + .5, self.fov, self.depth_range, self.depth_scale, self.log_scale)
            example['raw'] = sweep
            example['reproj'] = reproj_sweep.astype(np.float32)
        # image degradation
        if self.degradation_transform:
            degraded_proj_range = self.degradation_transform(proj_range)
            example['degraded_image'] = degraded_proj_range

        # cameras
        if self.condition_key == 'camera':
            cameras = self.load_camera(data_path)
            example[self.condition_key] = cameras
        return example

class nuScenesImageTrain(nuScenesBase):
    def __init__(self, **kwargs):
        super().__init__(split='train', **kwargs)

    def prepare_data(self):
        with open(os.path.join(self.data_root, 'v1.0-trainval/v1.0-trainval/sample_data.json')) as f:
            sample_data = json.load(f)

        custom_path = 'v1.0-trainval'
        file_paths = [os.path.join(self.data_root, custom_path, x['filename']) 
                           for x in sample_data 
                           if 'sweeps/LIDAR_TOP' in x['filename']]
        self.data = sorted(file_paths)

class nuScenesImageValidation(nuScenesBase):
    def __init__(self, **kwargs):
        super().__init__(split='val', **kwargs)

    def prepare_data(self):
        with open(os.path.join(self.data_root, 'v1.0-trainval/v1.0-mini/sample_data.json')) as f:
            sample_data = json.load(f)

        custom_path = 'v1.0-trainval'
        file_paths = [os.path.join(self.data_root, custom_path, x['filename']) 
                           for x in sample_data 
                           if 'sweeps/LIDAR_TOP' in x['filename']]
        self.data = sorted(file_paths)


class nuScenesLayoutBase(nuScenesBase):
    def __init__(self, **kwargs):
        self.info_path = kwargs['info_path']
        self.class_names = ['car','truck', 'construction_vehicle', 'bus', 'trailer', 'motorcycle', 'bicycle', 'pedestrian']
        self.max_layout = kwargs['max_layout']
        super().__init__(**kwargs)

    def prepare_data(self):
        with open(self.info_path, 'rb') as f:
            self.data = pickle.load(f)
        self.data = self.balanced_infos_resampling(self.data)

    def out_build_dataset(self, data_infos):
        self.data = data_infos

    def balanced_infos_resampling(self, infos):
        """
        Class-balanced sampling of nuScenes dataset from https://arxiv.org/abs/1908.09492
        """
        if self.class_names is None:
            return infos

        cls_infos = {name: [] for name in self.class_names}
        for info in infos:
            for name in set(info['gt_names']):
                if name in self.class_names:
                    cls_infos[name].append(info)

        duplicated_samples = sum([len(v) for _, v in cls_infos.items()])
        cls_dist = {k: len(v) / duplicated_samples for k, v in cls_infos.items()}

        sampled_infos = []

        frac = 1.0 / len(self.class_names)
        ratios = [frac / v for v in cls_dist.values()]

        for cur_cls_infos, ratio in zip(list(cls_infos.values()), ratios):
            sampled_infos += np.random.choice(
                cur_cls_infos, int(len(cur_cls_infos) * ratio)
            ).tolist()

        cls_infos_new = {name: [] for name in self.class_names}
        for info in sampled_infos:
            for name in set(info['gt_names']):
                if name in self.class_names:
                    cls_infos_new[name].append(info)

        return sampled_infos

    def get_lidar_with_sweeps(self, index):
        info = self.data[index]
        lidar_path = os.path.join(self.data_root, info['lidar_path'])
        points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])[:, :3]
        return points

    def scale_boxes(self, boxes_3d):
        new_boxes = np.zeros([boxes_3d.shape[0], 8])
        x_min, x_max = self.dataset_config.x_range
        y_min, y_max = self.dataset_config.y_range
        z_min, z_max = self.dataset_config.z_range

        boxes_3d[:,0] = (boxes_3d[:,0] - x_min) / (x_max - x_min)
        boxes_3d[:,1] = (boxes_3d[:,1] - y_min) / (y_max - y_min)
        boxes_3d[:,2] = (boxes_3d[:,2] - z_min) / (z_max - z_min)
        boxes_3d[:,3:6] = np.log(boxes_3d[:,3:6])
        new_boxes[:,:6] = boxes_3d[:,:6]
        new_boxes[:,6] = np.sin(boxes_3d[:,6])
        new_boxes[:,7] = np.cos(boxes_3d[:,6])
        return new_boxes
    
    def __getitem__(self, idx):
        input_dict = {}
        info = copy.deepcopy(self.data[idx])
        points = self.get_lidar_with_sweeps(idx)

        input_dict.update({
            'points': points,
            'gt_names': info['scene_graph']['keep_box_names'],
            'gt_boxes': info['scene_graph']['keep_box'],
        })

        if self.lidar_box_transform:
            input_dict = self.lidar_box_transform(input_dict)
        
        # range image
        proj_range, _ = pcd2range(input_dict['points'], self.img_size, self.fov, self.depth_range)
        proj_range, proj_mask = self.process_scan(proj_range)

        input_dict['image'] = proj_range
        input_dict['mask'] = proj_mask

        #  layout
        centers_coord_2d = box2coord2dx2(input_dict['gt_boxes'], self.fov, self.depth_range)
        gt_classes = np.array([self.class_names.index(n) + 1 for n in input_dict['gt_names']], dtype=np.int32)
        scaled_boxes = self.scale_boxes(input_dict['gt_boxes'])
        gt_boxes = np.concatenate((scaled_boxes, centers_coord_2d.reshape(-1,4), gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
        input_dict['layout'] = gt_boxes

        if self.return_pcd:
            reproj_sweep, _, _ = range2pcd(proj_range[0] * .5 + .5, self.fov, self.depth_range, self.depth_scale, self.log_scale)
            input_dict['reproj'] = reproj_sweep.astype(np.float32)

        input_dict.pop('points', None)
        input_dict.pop('gt_names', None)

        return input_dict
    
    def test_collate_fn(self, data):
        output = {}
        keys = data[0].keys()
        for k in keys:
            v = [d[k] for d in data]
            if k not in ['reproj', 'raw']:
                v = torch.from_numpy(np.stack(v, 0))
            else:
                v = [d[k] for d in data]
            output[k] = v
        return output

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
        return ret