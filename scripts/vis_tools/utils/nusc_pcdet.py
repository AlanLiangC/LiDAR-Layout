from pathlib import Path
import pickle
import numpy as np
from lidm.data.nuscenes_layout_dataset import nuScenesLayoutVal

class NUSC_PCDet:
    def __init__(self, data_root):
        self.raw_root = Path('/home/alan/AlanLiang/Dataset/pcdet_Nuscenes/v1.0-trainval')
        self.data_root = Path(data_root)
        info_path = Path(data_root) / 'nuscenes_infos_val.pkl'
        with open(info_path, 'rb') as f:
            self.data_infos = pickle.load(f)
        # scene dataset
        self.scene_dataset = nuScenesLayoutVal(
            root='/home/alan/AlanLiang/Projects/AlanLiang/CentralScene/data/nuscenes',
            split='val_scans',
            use_scene_rels=True,
            with_changes=False,
            with_CLIP=True,
            seed=False,
            bin_angle=False,
            dataset='nuscenes',
            recompute_feats=False,
            recompute_clip=False,
            eval=True,
            eval_type='none')

    @property
    def __len__(self):
        return len(self.data_infos)

    def get_points(self, index, max_sweeps=1):
        info = self.data_infos[index]
        lidar_path = self.raw_root / info['lidar_path']
        points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]

        sweep_points_list = [points]
        sweep_times_list = [np.zeros((points.shape[0], 1))]

        for k in np.random.choice(len(info['sweeps']), max_sweeps - 1, replace=False):
            points_sweep, times_sweep = self.get_sweep(info['sweeps'][k])
            sweep_points_list.append(points_sweep)
            sweep_times_list.append(times_sweep)

        points = np.concatenate(sweep_points_list, axis=0)
        times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

        points = np.concatenate((points, times), axis=1)
        return points

    def get_boxes(self, index):
        info = self.data_infos[index]
        return info['gt_boxes']

    def get_sweep(self, sweep_info):
        def remove_ego_points(points, center_radius=1.0):
            mask = ~((np.abs(points[:, 0]) < center_radius) & (np.abs(points[:, 1]) < center_radius))
            return points[mask]

        lidar_path = self.data_root / sweep_info['lidar_path']
        points_sweep = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]
        points_sweep = remove_ego_points(points_sweep).T
        if sweep_info['transform_matrix'] is not None:
            num_points = points_sweep.shape[1]
            points_sweep[:3, :] = sweep_info['transform_matrix'].dot(
                np.vstack((points_sweep[:3, :], np.ones(num_points))))[:3, :]

        cur_times = sweep_info['time_lag'] * np.ones((1, points_sweep.shape[1]))
        return points_sweep.T, cur_times.T
    
    def get_scene_nodes(self, index):
        pass