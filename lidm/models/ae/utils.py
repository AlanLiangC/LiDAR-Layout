import torch

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
    size = range_img.shape[1:]
    fov_up = range_img.new_ones([1]) * fov[0] / 180.0 * torch.pi  # field of view up in rad
    fov_down = range_img.new_ones([1]) * fov[1] / 180.0 * torch.pi  # field of view down in rad
    fov_range = torch.abs(fov_down) + torch.abs(fov_up)  # get field of view total in rad

    # inverse transform from depth
    depth = (range_img * depth_scale).flatten()
    if log_scale:
        depth = torch.exp2(depth) - 1

    scan_x, scan_y = torch.meshgrid(torch.arange(size[1]), torch.arange(size[0]))
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

    # not change TODO: Validation 
    # depth = range_img.flatten()
    # scan_x, scan_y = torch.meshgrid(torch.arange(size[1]), torch.arange(size[0]))
    # scan_x = (scan_x / size[1]).to(range_img.device)
    # scan_y = (scan_y / size[0]).to(range_img.device)

    # yaw = (torch.pi * (scan_x * 2 - 1)).flatten()
    # pitch = ((1.0 - scan_y) * fov_range - abs(fov_down)).flatten()

    # pcd = range_img.new_zeros((len(yaw), 3))
    # pcd[:, 0] = torch.cos(yaw) * torch.cos(pitch) * depth
    # pcd[:, 1] = -torch.sin(yaw) * torch.cos(pitch) * depth
    # pcd[:, 2] = torch.sin(pitch) * depth
    # mask = depth > 0
    # pcd = pcd[mask, :]

    return pcd, mask

def range2feature_gpu(range_feature, mask, is_sh):
    feature = range_feature.flatten(1,2).permute(1,0)
    if is_sh:
        N, _ = feature.shape
        feature = feature.view(N, 4, 16)
    return feature[mask]

