#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import torch
import math
from diff_lidargs_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer

def generate_neural_gaussians(pc , visible_mask=None):
    ## view frustum filtering for acceleration    
    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device)
    
    xyz = pc.get_anchor[visible_mask]
    neural_opacity = pc.get_opacity
    mask = (neural_opacity>0.0)
    mask = mask.view(-1)

    # select opacity 
    xyz = xyz[mask]
    opacity = neural_opacity[mask]
    color = pc.get_color[mask]
    raydrop = pc.get_raydrop[mask]
    color = torch.cat([color,raydrop],dim=1) # color and raydrop will be concatenated together , as the rasterization' input
    scaling = pc.get_scaling[mask]
    scaling = torch.clamp_max(scaling, max=0.1)
    rot = pc.get_rotation[mask]

    return xyz, color, opacity, scaling, rot, neural_opacity, mask


def render(viewpoint_camera, pc, debug, bg_color : torch.Tensor, scaling_modifier = 1.0, visible_mask=None, retain_grad=True):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """        
    xyz, color, opacity, scaling, rot, neural_opacity, mask = generate_neural_gaussians(pc, visible_mask)

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros((xyz.shape[0], 4), dtype=xyz.dtype, requires_grad=True, device="cuda") + 0
    if retain_grad:
        try:
            screenspace_points.retain_grad()
        except:
            pass

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        bg=bg_color,
        scale_modifier=1.0,
        depth_threshold=0.001,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.lidar_center,
        prefiltered=False,
        beam_inclinations = viewpoint_camera.beam_inclinations,  # TODO 输入一个beam
        debug=debug,
        lidar_far = int(56),
        lidar_near = int(0)
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, allmap, pixels = rasterizer(
        means3D = xyz,
        means2D = screenspace_points,
        shs = None,
        colors_precomp = color,
        opacities = opacity,
        scales = scaling,
        rotations = rot,
        cov3D_precomp = None)
    
    depth = allmap[0:1]

    return {"render": rendered_image,
            "depth":depth,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "scaling": scaling,
            }



def prefilter_voxel(viewpoint_camera, pc, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_anchor, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    depth_max = 80.0
    depth_min = 0.0

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.lidar_center,
        prefiltered=False,
        beam_inclinations = viewpoint_camera.beam_inclinations,
        debug=pipe.debug,
        lidar_far = int(depth_max),
        lidar_near = int(depth_min)
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_anchor
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    radii_pure = rasterizer.visible_filter(means3D = means3D,
        scales = scales[:,:3],
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    return radii_pure > 0

def mini_render(viewpoint_camera, pc, debug, bg_color : torch.Tensor, scaling_modifier = 1.0, visible_mask=None, retain_grad=False):

    xyz = pc.get_xyz
    color = pc.color
    opacity = pc.get_opacity
    scaling = pc.get_scaling
    rot = pc.get_rotation

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros((xyz.shape[0], 4), dtype=xyz.dtype, requires_grad=True, device="cuda") + 0
    if retain_grad:
        try:
            screenspace_points.retain_grad()
        except:
            pass

    depth_max = 80
    depth_min = 0

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        bg=bg_color,
        scale_modifier=1.0,
        depth_threshold=0.001,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.lidar_center,
        prefiltered=False,
        beam_inclinations = viewpoint_camera.beam_inclinations,  # TODO 输入一个beam
        debug=debug,
        lidar_far = int(depth_max),
        lidar_near = int(depth_min)
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, allmap, pixels = rasterizer(
        means3D = xyz,
        means2D = screenspace_points,
        shs = None,
        colors_precomp = color,
        opacities = opacity,
        scales = scaling,
        rotations = rot,
        cov3D_precomp = None)
    
    depth = allmap[0:1]
    return {"render": rendered_image,
            "depth":depth,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "scaling": scaling,
            }
