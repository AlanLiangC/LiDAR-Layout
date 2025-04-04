import torch
import numbers as np
from simple_knn._C import distCUDA2
from .utils.general_utils import build_scaling_rotation, strip_symmetric, inverse_sigmoid

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.scaling_t_activation = torch.exp
        self.scaling_t_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

        self.intensity_activation = torch.sigmoid

    def __init__(self):
        self.max_sh_degree = 2
        self.setup_functions()

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_scaling_t(self):
        return self.scaling_t_activation(self._scaling_t)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    def get_xyz_SHM(self, t):
        a = 1 / self.T * np.pi * 2
        return self._xyz + self._velocity * torch.sin((t - self._t) * a) / a

    @property
    def get_inst_velocity(self):
        return self._velocity * torch.exp(-self.get_scaling_t / self.T / 2 * self.velocity_decay)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_t(self):
        return self._t

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_intensity(self):
        return self.intensity_activation(self._intensity)

    @property
    def get_max_sh_channels(self):
        return (self.max_sh_degree + 1) ** 2
    
    def get_marginal_t(self, timestamp):
        return torch.exp(-0.5 * (self.get_t - timestamp) ** 2 / self.get_scaling_t ** 2)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, xyz, rot, scale, opacity, sh):
        self._xyz = xyz
        self._features_dc = sh[:, :, 0:1].transpose(1, 2).contiguous()
        self._features_rest = sh[:, :, 1:].transpose(1, 2).contiguous()

        scale = self.scaling_inverse_activation(scale.clamp_min(0.01))

        self._scaling = scale

        self._rotation = rot
        self._opacity = opacity
        self.max_radii2D = xyz.new_zeros((self.get_xyz.shape[0]))
        self._t = xyz.new_zeros((self.get_xyz.shape[0]))
