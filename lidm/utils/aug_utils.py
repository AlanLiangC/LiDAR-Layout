import torch
import numpy as np

def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False

def get_lidar_transform(config, split):
    transform_list = []
    if config['rotate']:
        transform_list.append(RandomRotateAligned())
    if config['flip']:
        transform_list.append(RandomFlip())

    
    return Compose(transform_list) if len(transform_list) > 0 and split == 'train' else None

def get_lidar_box_transform(config, split):
    transform_list = []
    if config['flip_w_box']:
        transform_list.append(RandomRotateAligned_w_Box())
    if config['rotate_w_box']:
        transform_list.append(RandomFlip_w_Box())

    return Compose_w_Box(transform_list) if len(transform_list) > 0 and split == 'train' else None

def get_camera_transform(config, split):
    # import open_clip
    # transform = open_clip.image_transform((224, 224), split == 'train', resize_longest_max=True)
    # TODO
    transform = None
    return transform

def mask_points_by_range(points, limit_range):
    mask = (points[:, 0] >= limit_range[0]) & (points[:, 0] <= limit_range[3]) \
           & (points[:, 1] >= limit_range[1]) & (points[:, 1] <= limit_range[4]) \
           & (points[:, 2] >= limit_range[2]) & (points[:, 2] <= limit_range[5])

    return mask

def get_anno_transform(config, split):
    if config['keypoint_drop'] and split == 'train':
        drop_range = config['keypoint_drop_range'] if 'keypoint_drop_range' in config else (5, 60)
        transform = RandomKeypointDrop(drop_range)
    else:
        transform = None
    return transform

def random_flip_along_x(gt_boxes, points, return_flip=False, enable=None):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C)
    Returns:
    """
    if enable is None:
        enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        gt_boxes[:, 1] = -gt_boxes[:, 1]
        gt_boxes[:, 6] = -gt_boxes[:, 6]
        points[:, 1] = -points[:, 1]
        
        if gt_boxes.shape[1] > 7:
            gt_boxes[:, 8] = -gt_boxes[:, 8]
    if return_flip:
        return gt_boxes, points, enable
    return gt_boxes, points


def random_flip_along_y(gt_boxes, points, return_flip=False, enable=None):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C)
    Returns:
    """
    if enable is None:
        enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        gt_boxes[:, 0] = -gt_boxes[:, 0]
        gt_boxes[:, 6] = -(gt_boxes[:, 6] + np.pi)
        points[:, 0] = -points[:, 0]

        if gt_boxes.shape[1] > 7:
            gt_boxes[:, 7] = -gt_boxes[:, 7]
    if return_flip:
        return gt_boxes, points, enable
    return gt_boxes, points

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

def global_rotation(gt_boxes, points, rot_range, return_rot=False, noise_rotation=None):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C),
        rot_range: [min, max]
    Returns:
    """
    if noise_rotation is None: 
        noise_rotation = np.random.uniform(rot_range[0], rot_range[1])
    points = rotate_points_along_z(points[np.newaxis, :, :], np.array([noise_rotation]))[0]
    gt_boxes[:, 0:3] = rotate_points_along_z(gt_boxes[np.newaxis, :, 0:3], np.array([noise_rotation]))[0]
    gt_boxes[:, 6] += noise_rotation
    if gt_boxes.shape[1] > 7:
        gt_boxes[:, 7:9] = rotate_points_along_z(
            np.hstack((gt_boxes[:, 7:9], np.zeros((gt_boxes.shape[0], 1))))[np.newaxis, :, :],
            np.array([noise_rotation])
        )[0][:, 0:2]

    if return_rot:
        return gt_boxes, points, noise_rotation
    return gt_boxes, points

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, pcd, pcd1=None):
        for t in self.transforms:
            pcd, pcd1 = t(pcd, pcd1)
        return pcd, pcd1

class Compose_w_Box(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data_dict):
        for t in self.transforms:
            data_dict = t(data_dict)
        return data_dict


class RandomFlip(object):
    def __init__(self, p=1.):
        self.p = p

    def __call__(self, coord, coord1=None):
        if np.random.rand() < self.p:
            if np.random.rand() < 0.5:
                coord[:, 0] = -coord[:, 0]
                if coord1 is not None:
                    coord1[:, 0] = -coord1[:, 0]
            if np.random.rand() < 0.5:
                coord[:, 1] = -coord[:, 1]
                if coord1 is not None:
                    coord1[:, 1] = -coord1[:, 1]
        return coord, coord1

class RandomFlip_w_Box(object):
    def __init__(self, p=1.):
        self.p = p

    def __call__(self, data_dict):
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        for cur_axis in ['x', 'y']:
            assert cur_axis in ['x', 'y']
            if cur_axis == 'x':
                func = random_flip_along_x
            else:
                func = random_flip_along_y
            gt_boxes, points, enable = func(
                gt_boxes, points, return_flip=True
            )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

class RandomRotateAligned(object):
    def __init__(self, rot=np.pi / 4, p=1.):
        self.rot = rot
        self.p = p

    def __call__(self, coord, coord1=None):
        if np.random.rand() < self.p:
            angle_z = np.random.uniform(-self.rot, self.rot)
            cos_z, sin_z = np.cos(angle_z), np.sin(angle_z)
            R = np.array([[cos_z, -sin_z, 0], [sin_z, cos_z, 0], [0, 0, 1]])
            coord = np.dot(coord, R)
            if coord1 is not None:
                coord1 = np.dot(coord1, R)
        return coord, coord1

class RandomRotateAligned_w_Box(object):
    def __init__(self):
        self.range = [-0.3925, 0.3925]

    def __call__(self, data_dict):
        rot_range = self.range
        if not isinstance(rot_range, list):
            rot_range = [-rot_range, rot_range]
        gt_boxes, points, noise_rot = global_rotation(
            data_dict['gt_boxes'], data_dict['points'], rot_range=rot_range, return_rot=True
        )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

class RandomKeypointDrop(object):
    def __init__(self, num_range=(5, 60), p=.5):
        self.num_range = num_range
        self.p = p

    def __call__(self, center, category=None):
        if np.random.rand() < self.p:
            num = len(center)
            if num > self.num_range[0]:
                num_kept = np.random.randint(self.num_range[0], min(self.num_range[1], num))
                idx_kept = np.random.choice(num, num_kept, replace=False)
                center, category = center[idx_kept], category[idx_kept]
        return center, category


# class ResizeMaxSize(object):
#     def __init__(self, max_size, interpolation=InterpolationMode.BICUBIC, fn='max', fill=0):
#         super().__init__()
#         if not isinstance(max_size, int):
#             raise TypeError(f"Size should be int. Got {type(max_size)}")
#         self.max_size = max_size
#         self.interpolation = interpolation
#         self.fn = min if fn == 'min' else min
#         self.fill = fill
#
#     def forward(self, img):
#         width, height = img.size
#         scale = self.max_size / float(max(height, width))
#         if scale != 1.0:
#             new_size = tuple(round(dim * scale) for dim in (height, width))
#             img = F.resize(img, new_size, self.interpolation)
#             pad_h = self.max_size - new_size[0]
#             pad_w = self.max_size - new_size[1]
#             img = F.pad(img, padding=[pad_w//2, pad_h//2, pad_w - pad_w//2, pad_h - pad_h//2], fill=self.fill)
#         return img
