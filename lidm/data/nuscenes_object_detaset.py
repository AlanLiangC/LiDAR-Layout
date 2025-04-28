import os
import pickle
import random
import numpy as np
from collections import defaultdict
import torch
from torch.utils.data import Dataset

CLASS_NAME = ['car', 'truck', 'pedestrian', 'bicycle', 'motorcycle', 'bus', 'construction_vehicle', 'trailer']

def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot

class NuscenesObject(Dataset):
    def __init__(self, data_root, pkl_path, split, **kwargs):
        super().__init__()
        self.data_root = data_root
        self.pkl_path = pkl_path
        self.split = split
        self.num_samples = 1024
        self.prepare_data()

    def prepare_data(self):
        fg_objects_file = open(self.pkl_path, 'rb')
        fg_objects_dict = pickle.load(fg_objects_file)
        self.data = []
        self.class_samples = []
        for class_idx, class_name in enumerate(CLASS_NAME):
            fg_objects_list = fg_objects_dict[class_name]
            self.data.extend(fg_objects_list)
            self.class_samples.extend([class_idx]*len(fg_objects_list))

        combined = list(zip(self.data, self.class_samples))
        random.shuffle(combined)
        self.data, self.class_samples = zip(*combined)
        
        if self.split == 'val':
            self.data = self.data[:10000]

    def load_points(self, fg_path):
        fg_path = os.path.join(self.data_root, fg_path)
        fg_points = np.fromfile(fg_path, dtype=np.float32).reshape(-1,5)[:,:3]
        return fg_points

    def __len__(self):
        return len(self.data)

    def norm_fg_points(self, fg_points, box3d):
        # fg_points_center = box3d[:3][None, :]
        # fg_points = (fg_points - fg_points_center)[np.newaxis, :, :] # NOTE: Already to center
        rotation = -np.array([box3d[-1]])
        fg_points = rotate_points_along_z(fg_points[np.newaxis, :, :], rotation)[0]
        fg_points[:,0] /= box3d[3]
        fg_points[:,1] /= box3d[4]
        fg_points[:,2] /= box3d[5]
        return fg_points

    def sample_points(self, points):

        N = len(points)
        
        if N <= self.num_samples:
            indices = np.random.choice(N, self.num_samples, replace=True)
            return points[indices]
        
        pts_depth = np.linalg.norm(points[:, 0:3], axis=1)
        pts_near_flag = pts_depth < 0.1
        far_idxs_choice = np.where(pts_near_flag == 0)[0]
        near_idxs = np.where(pts_near_flag == 1)[0]
        choice = []
        if self.num_samples > len(far_idxs_choice):
            near_idxs_choice = np.random.choice(near_idxs, self.num_samples - len(far_idxs_choice), replace=False)
            choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
                if len(far_idxs_choice) > 0 else near_idxs_choice
        else: 
            choice = np.arange(0, len(points), dtype=np.int32)
            choice = np.random.choice(choice, self.num_samples, replace=False)
        np.random.shuffle(choice)
        return points[choice]

    def __getitem__(self, index):
        data_dict = {}
        fg_info = self.data[index]
        if fg_info['num_points_in_gt'] < 50:
            random_index = random.randint(0, self.__len__()-1)
            return self.__getitem__(random_index)
        
        fg_points_path = fg_info['path']
        fg_points = self.load_points(fg_points_path)
        box3d = fg_info['box3d_lidar'][:7]
        fg_points = self.norm_fg_points(fg_points, box3d)
        fg_points = self.sample_points(fg_points)
        data_dict.update({
            'fg_points': fg_points,
            'fg_class': np.array([self.class_samples[index]])
        })
        return data_dict
    
    def collate_fn(self, batch_list, _unused=False):
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret = {}
        batch_size_ratio = 1

        for key, val in data_dict.items():

            ret[key] = torch.from_numpy(np.stack(val, axis=0)).float()

        ret['batch_size'] = batch_size * batch_size_ratio
        return ret

if __name__ == "__main__":
    from tqdm import tqdm
    dataset = NuscenesObject(data_root='/home/alan/AlanLiang/Dataset/pcdet_Nuscenes/v1.0-trainval',
                             pkl_path='/home/alan/AlanLiang/Dataset/pcdet_Nuscenes/v1.0-trainval/nuscenes_dbinfos_10sweeps_withvelo.pkl',
                             split='train')
    for i in tqdm(range(dataset.__len__())):
        fg_points = dataset.__getitem__(i)
