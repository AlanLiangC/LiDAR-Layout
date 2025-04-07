import torch
import numpy as np
import fvdb
# JIT
from torch.utils.cpp_extension import load

dvr = load("dvr", sources=["../lidm/models/ae/lib/dvr/dvr.cpp", "../lidm/models/ae/lib/dvr/dvr.cu"], verbose=True, extra_cuda_cflags=['-allow-unsupported-compiler'])

def scale_range(range_img, depth_scale, log_scale=True):
    range_img = torch.where(range_img < 0, 0, range_img)

    if log_scale:
        # log scale
        range_img = torch.log2(range_img + 0.0001 + 1)

    range_img = range_img / depth_scale
    range_img = range_img * 2. - 1.

    range_img = torch.clip(range_img, -1, 1)
    return range_img

def range2pcd_gpu(range_img, fov, depth_range, depth_scale, log_scale=True, **kwargs):
    # laser parameters
    size = range_img.squeeze().shape
    fov_up = range_img.new_ones([1]) * fov[0] / 180.0 * torch.pi  # field of view up in rad
    fov_down = range_img.new_ones([1]) * fov[1] / 180.0 * torch.pi  # field of view down in rad
    fov_range = torch.abs(fov_down) + torch.abs(fov_up)  # get field of view total in rad

    # inverse transform from depth
    depth = (range_img * depth_scale).flatten()
    if log_scale:
        depth = torch.exp2(depth) - 1

    scan_x, scan_y = torch.meshgrid(torch.arange(size[1]), torch.arange(size[0]), indexing="xy")
    scan_x = (scan_x / size[1]).to(range_img.device)
    scan_y = (scan_y / size[0]).to(range_img.device)

    yaw = (torch.pi * (scan_x * 2 - 1)).flatten()
    pitch = ((1.0 - scan_y) * fov_range - abs(fov_down)).flatten()

    pcd = range_img.new_zeros((len(yaw), 3))
    pcd[:, 0] = torch.cos(yaw) * torch.cos(pitch) * depth
    pcd[:, 1] = -torch.sin(yaw) * torch.cos(pitch) * depth
    pcd[:, 2] = torch.sin(pitch) * depth

    # mask out invalid points
    mask = torch.logical_and(depth > depth_range[0], depth < depth_range[1])
    pcd = pcd[mask, :]

    return pcd, mask

def range2pcd(range_img, fov, depth_range, depth_scale, log_scale=True, **kwargs):
    # laser parameters
    size = range_img.shape
    fov_up = fov[0] / 180.0 * np.pi  # field of view up in rad
    fov_down = fov[1] / 180.0 * np.pi  # field of view down in rad
    fov_range = abs(fov_down) + abs(fov_up)  # get field of view total in rad

    # inverse transform from depth
    depth = (range_img * depth_scale).flatten()
    if log_scale:
        depth = np.exp2(depth) - 1

    scan_x, scan_y = np.meshgrid(np.arange(size[1]), np.arange(size[0]))
    scan_x = scan_x.astype(np.float64) / size[1]
    scan_y = scan_y.astype(np.float64) / size[0]

    yaw = (np.pi * (scan_x * 2 - 1)).flatten()
    pitch = ((1.0 - scan_y) * fov_range - abs(fov_down)).flatten()

    pcd = np.zeros((len(yaw), 3))
    pcd[:, 0] = np.cos(yaw) * np.cos(pitch) * depth
    pcd[:, 1] = -np.sin(yaw) * np.cos(pitch) * depth
    pcd[:, 2] = np.sin(pitch) * depth

    # mask out invalid points
    mask = np.logical_and(depth > depth_range[0], depth < depth_range[1])
    pcd = pcd[mask, :]

    return pcd, mask

def range2feature_gpu(range_feature, mask, is_sh):
    feature = range_feature.flatten(1,2).permute(1,0)
    if is_sh:
        N, _ = feature.shape
        feature = feature.view(N, 4, 16)
    return feature[mask]

def point2voxel(batch, offset, scaler, input_grid):
    '''
    input:
        points: [B,N,3]
    return
        voxel
    '''
    target_voxel_size = float(scaler[0,0,0])
    raw_points = batch['points_for_cube']

    # input_points = ((raw_points - offset) / scaler).float()
    # input_tindex = input_points.new_zeros([input_points.shape[0], input_points.shape[1]])
    # input_occupancy = dvr.init(input_points, input_tindex, input_grid) # N x T1 x H x L x W

    # xyzs = []
    # for batch_idx in range(input_occupancy.shape[0]):
    #     batch_occupancy = input_occupancy[batch_idx].squeeze().permute(2,1,0)
    #     ijk = torch.nonzero(batch_occupancy == 1, as_tuple=False)
    #     # xyz = ijk.float() * scaler[0] + offset[0] + scaler[0]/2
    #     xyzs.append(ijk)

    xyzs = []
    for batch_idx in range(int(raw_points[:,0].max() + 1)):
        batch_mask = raw_points[:,0] == batch_idx
        xyz = raw_points[batch_mask][:,1:] - offset[0]
        xyzs.append(xyz)
    target_grid = fvdb.sparse_grid_from_points(
            fvdb.JaggedTensor(xyzs), voxel_sizes=target_voxel_size, origins=[target_voxel_size / 2.] * 3)
    # target_grid = fvdb.sparse_grid_from_ijk(
    #         fvdb.JaggedTensor(xyzs), voxel_sizes=target_voxel_size, origins=[target_voxel_size / 2.] * 3)
    # temp_grid = fvdb.sparse_grid_from_points(
    #         fvdb.JaggedTensor([ijk.float() for ijk in xyzs]), voxel_sizes=target_voxel_size, origins=[target_voxel_size / 2.] * 3)
    batch['INPUT_PC'] = target_grid
    return batch
