import numpy as np
import torch
from torch import nn
from pointcept.models.builder import MODELS
from .utils.general_utils import build_scaling_rotation, strip_symmetric, inverse_sigmoid
from .utils.graphics_utils import get_beam_inclinations
from .render import render
from .utils.cameras import Camera

@MODELS.register_module("GSDecoder")
class GaussianModel(nn.Module):

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize
    
    def __init__(self,
                 feat_dim: int=64, 
                 n_offsets: int=6, 
                 voxel_size: float=0.0,
                 update_depth: int=3, 
                 update_init_factor: int=16,
                 update_hierachy_factor: int=4,
                 use_feat_bank : bool = False,
                 appearance_dim : int = 0,
                 ratio : int = 1,
                 add_opacity_dist : bool = True,
                 add_cov_dist : bool = True,
                 add_color_dist : bool = True,
                 color_channel : int = 2,
                 *args, 
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.setup_functions()
        self.feat_dim = feat_dim
        self.n_offsets = n_offsets
        self.voxel_size = voxel_size
        self.update_depth = update_depth
        self.update_init_factor = update_init_factor
        self.update_hierachy_factor = update_hierachy_factor
        self.use_feat_bank = use_feat_bank

        self.color_channel = color_channel
        self.appearance_dim = appearance_dim
        self.embedding_appearance = None
        self.embedding_appearance_rd = None
        self.ratio = ratio
        self.add_opacity_dist = add_opacity_dist
        self.add_cov_dist = add_cov_dist
        self.add_color_dist = add_color_dist
        self.add_opacity_dist = add_opacity_dist

        feat_dim_temp = 32
        self.mlp_offset = nn.Sequential(
            nn.Linear(feat_dim, feat_dim_temp),
            nn.ReLU(True),
            nn.Linear(feat_dim_temp, n_offsets * 3),
            nn.Sigmoid()
        )
        self.mlp_opacity = nn.Sequential(
            nn.Linear(feat_dim, feat_dim_temp),
            nn.ReLU(True),
            nn.Linear(feat_dim_temp, n_offsets),
            nn.Tanh()
        )

        self.mlp_cov = nn.Sequential(
            nn.Linear(feat_dim, feat_dim_temp),
            nn.ReLU(True),
            nn.Linear(feat_dim_temp, 6*self.n_offsets),
        )

        self.mlp_color = nn.Sequential(
            nn.Linear(feat_dim, feat_dim_temp), # 
            nn.ReLU(True),
            nn.Linear(feat_dim_temp, (self.color_channel-1)*self.n_offsets), 
            nn.Sigmoid()                           
        )
        self.mlp_raydrop = nn.Sequential(
            nn.Linear(feat_dim, feat_dim_temp),
            nn.ReLU(True),
            nn.Linear(feat_dim_temp, 1*self.n_offsets),
            nn.Sigmoid()
        )
        self.img_size = [32,1024]

    @property
    def get_mlp(self):
        return self.mlp_cov

    @property
    def get_anchor(self):
        return self._anchor
    
    @property
    def get_color(self):
        return self._color

    @property
    def get_raydrop(self):
        return self._raydrop

    @property
    def get_opacity(self):
        return self._opacity

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    def forward(self, point):
        feat = point.feat
        point.grid_offset = self.mlp_offset(feat)
        point.opacity = self.mlp_opacity(feat)
        point.scale_rot = self.mlp_cov(feat)
        point.color = self.mlp_color(feat)
        point.raydrop = self.mlp_raydrop(feat)
        return point

    def create_from_pcd(self, point, batch_idx):
        mask = point.batch == batch_idx
        points_num = mask.sum()
        coord = point.coord[mask].unsqueeze(1).repeat(1,self.n_offsets,1)
        offset = point.grid_offset[mask].reshape(points_num, self.n_offsets, -1)
        self._anchor = (coord + offset).reshape(-1, 3)

        scale_rot = point.scale_rot[mask].reshape(points_num, self.n_offsets, -1)
        self._scaling = scale_rot[..., :2].reshape(-1, 2)
        self._rotation = scale_rot[..., 2:].reshape(-1, 4)
        self._opacity = point.opacity[mask].reshape(-1, 1)
        self._color = point.color[mask].reshape(-1, self.color_channel-1)
        self._raydrop = point.raydrop[mask].reshape(-1, 1)

    def build_camera(self, device='cuda'):
        l2c = np.array([1, 0, 0, 0,
                        0, 1, 0, 0,
                        0, 0, 1, 0,
                        0, 0, 0, 1]).reshape(4, 4)
        
        R = np.transpose(l2c[:3, :3])
        T = l2c[:3, 3]
        self.bg_color = torch.zeros([3]).to(device)
        beam_inclinations = get_beam_inclinations(10.0,40.0,self.img_size[0]).copy()

        # render dir  # 
        i, j = np.meshgrid(np.arange(self.img_size[1], dtype=np.float32),
                        np.arange(self.img_size[0], dtype=np.float32),
                        indexing='xy')
        beta = -(i - self.img_size[1] / 2.0) / self.img_size[1] * 2.0 * np.pi
        alpha = np.expand_dims(beam_inclinations[::-1], 1).repeat(self.img_size[1], 1)
        dirs = np.stack([
            np.cos(alpha) * np.cos(beta),
            np.cos(alpha) * np.sin(beta),
            np.sin(alpha),
        ], -1)
        self.camera_info = Camera(colmap_id=0, 
                                  image_size=self.img_size,
                                  R = R,
                                  T = T,
                                  FoVx=None,
                                  FoVy=None,
                                  gt_alpha_mask=None,
                                  image_name='*',
                                  uid=0,
                                  beam_inclinations=beam_inclinations,
                                  lidar_center=np.zeros([1, 3]),
                                  ray_dir=dirs,
                                  data_device = device
        )
    def scale_range(self, range_img, log_scale=True, depth_scale=5.84):
        range_img = torch.where(range_img < 0, 0, range_img)

        if log_scale:
            # log scale
            range_img = torch.log2(range_img + 0.0001 + 1)

        range_img = range_img / depth_scale
        range_img = range_img * 2. - 1.

        range_img = torch.clip(range_img, -1, 1)
        return range_img

    def decode(self, point):
        point = self(point)
        depth_list = []
        ray_drop_list = []
        if not hasattr(self, 'camera_info'):
            self.build_camera(device=point.feat.device)

        batch_size = int(point.batch.max() + 1)
        for batch_idx in range(batch_size):
            self.create_from_pcd(point, batch_idx)
            render_pkj = render(self.camera_info, self, False, self.bg_color)
            depth = render_pkj['depth']
            ray_drop = render_pkj['render'][1:2,...]
            depth_list.append(self.scale_range(depth))
            ray_drop_list.append(ray_drop)

        point.pred_range = torch.cat(depth_list, dim=0)
        point.pred_ray_drop = torch.cat(ray_drop_list, dim=0)

        return point