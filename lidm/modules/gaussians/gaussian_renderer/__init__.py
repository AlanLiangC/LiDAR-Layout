#
# Copyright (C) 2025, Fudan Zhang Vision Group
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE and LICENSE_gaussian_splatting.md files.
#
import torch
import math
from .diff_gaussian_rasterization_2d import GaussianRasterizationSettings, GaussianRasterizer
from lidm.modules.gaussians.utils.sh_utils import eval_sh

def render(viewpoint_camera, pc, bg_color: torch.Tensor, scaling_modifier=1.0, scale_factor=1.0,
           override_color=None, env_map=None,
           time_shift=None, other=[], mask=None, is_training=False, device='cuda'):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros((pc.get_xyz.shape[0], 4), dtype=pc.get_xyz.dtype, requires_grad=True, device=device) + 0
    try:
        screenspace_points.requires_grad_(True)
        screenspace_points.retain_grad()
    except:
        pass


    tanfovx = math.tan(-0.5)
    tanfovy = math.tan(-0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=3,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
        vfov=viewpoint_camera.vfov,
        hfov=viewpoint_camera.hfov,
        scale_factor=scale_factor
    )

    assert raster_settings.bg.shape[0] == 4

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means2D = screenspace_points
    opacity = pc.get_opacity
    scales = None
    rotations = None
    cov3D_precomp = None

    means3D = pc.get_xyz

    scales = pc.get_scaling
    rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    shs = pc.get_features
    feature_list = other

    if len(feature_list) > 0:
        features = torch.cat(feature_list, dim=1)
        S_other = features.shape[1]
    else:
        features = torch.zeros_like(means3D[:, :0])
        S_other = 0

    # Prefilter
    mask = (opacity[:, 0] > 1 / 255) if mask is None else mask & (opacity[:, 0] > 1 / 255)

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    contrib, rendered_image, rendered_feature, rendered_depth, rendered_opacity, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        features=features,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
        mask=mask)

    _, rendered_intensity_sh, rendered_raydrop = rendered_image.split([2, 1, 1], dim=0)
    rendered_other, rendered_normal = rendered_feature.split([S_other, 3], dim=0)
    rendered_normal = rendered_normal / (rendered_normal.norm(dim=0, keepdim=True) + 1e-8)

    if env_map is not None:
        lidar_raydrop_prior_from_envmap = env_map(viewpoint_camera.towards)
        rendered_raydrop = lidar_raydrop_prior_from_envmap + (1 - lidar_raydrop_prior_from_envmap) * rendered_raydrop

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "contrib": contrib,
        "depth": rendered_depth[[0]],
        "depth_mean": rendered_depth[[0]],
        "depth_median": rendered_depth[[1]],
        "distortion": rendered_depth[[2]],
        "depth_square": rendered_depth[[3]],
        "alpha": rendered_opacity,
        "feature": rendered_other,
        "normal": rendered_normal,
        "intensity_sh": rendered_intensity_sh,
        "raydrop": rendered_raydrop.clamp(0, 1)
    }


def render_range_map(args, cam_front, cam_back, gaussians, renderFunc, renderArgs, env_map, hw):
    assert cam_front.towards == "forward" and cam_back.towards == "backward"
    assert cam_front.colmap_id + args.frames == cam_back.colmap_id

    EPS = 1e-5
    h, w = hw
    breaks = (0, w // 2, 3 * w // 2, w * 2)

    depth_pano = torch.zeros([3, h, w * 2]).cuda()
    intensity_sh_pano = torch.zeros([1, h, w * 2]).cuda()
    raydrop_pano = torch.zeros([1, h, w * 2]).cuda()
    gt_depth_pano = torch.zeros([1, h, w * 2]).cuda()
    gt_intensity_pano = torch.zeros([1, h, w * 2]).cuda()

    for idx, viewpoint in enumerate([cam_front, cam_back]):
        depth_gt = viewpoint.pts_depth
        intensity_gt = viewpoint.pts_intensity
        render_pkg = renderFunc(viewpoint, gaussians, *renderArgs, env_map=env_map)

        depth = render_pkg['depth']
        alpha = render_pkg['alpha']
        raydrop_render = render_pkg['raydrop']

        depth_var = render_pkg['depth_square'] - depth ** 2
        depth_median = render_pkg["depth_median"]
        var_quantile = depth_var.median() * 10

        depth_mix = torch.zeros_like(depth)
        depth_mix[depth_var > var_quantile] = depth_median[depth_var > var_quantile]
        depth_mix[depth_var <= var_quantile] = depth[depth_var <= var_quantile]

        depth = torch.cat([depth_mix, depth, depth_median])

        if args.sky_depth:
            sky_depth = 900
            depth = depth / alpha.clamp_min(EPS)
            if args.depth_blend_mode == 0:  # harmonic mean
                depth = 1 / (alpha / depth.clamp_min(EPS) + (1 - alpha) / sky_depth).clamp_min(EPS)
            elif args.depth_blend_mode == 1:
                depth = alpha * depth + (1 - alpha) * sky_depth

        intensity_sh = render_pkg['intensity_sh']

        if idx % 2 == 0:  # 前180度
            depth_pano[:, :, breaks[1]:breaks[2]] = depth
            gt_depth_pano[:, :, breaks[1]:breaks[2]] = depth_gt

            intensity_sh_pano[:, :, breaks[1]:breaks[2]] = intensity_sh
            gt_intensity_pano[:, :, breaks[1]:breaks[2]] = intensity_gt

            raydrop_pano[:, :, breaks[1]:breaks[2]] = raydrop_render

            continue
        else:
            depth_pano[:, :, breaks[2]:breaks[3]] = depth[:, :, 0:(breaks[3] - breaks[2])]
            depth_pano[:, :, breaks[0]:breaks[1]] = depth[:, :, (w - breaks[1] + breaks[0]):w]

            gt_depth_pano[:, :, breaks[2]:breaks[3]] = depth_gt[:, :, 0:(breaks[3] - breaks[2])]
            gt_depth_pano[:, :, breaks[0]:breaks[1]] = depth_gt[:, :, (w - breaks[1] + breaks[0]):w]

            intensity_sh_pano[:, :, breaks[2]:breaks[3]] = intensity_sh[:, :, 0:(breaks[3] - breaks[2])]
            intensity_sh_pano[:, :, breaks[0]:breaks[1]] = intensity_sh[:, :, (w - breaks[1] + breaks[0]):w]

            gt_intensity_pano[:, :, breaks[2]:breaks[3]] = intensity_gt[:, :, 0:(breaks[3] - breaks[2])]
            gt_intensity_pano[:, :, breaks[0]:breaks[1]] = intensity_gt[:, :, (w - breaks[1] + breaks[0]):w]

            raydrop_pano[:, :, breaks[2]:breaks[3]] = raydrop_render[:, :, 0:(breaks[3] - breaks[2])]
            raydrop_pano[:, :, breaks[0]:breaks[1]] = raydrop_render[:, :, (w - breaks[1] + breaks[0]):w]

    return depth_pano, intensity_sh_pano, raydrop_pano, gt_depth_pano, gt_intensity_pano
