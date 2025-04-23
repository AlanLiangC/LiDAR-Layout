from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def generate_polar_coords(H: int, W: int, device: torch.device = "cpu") -> torch.Tensor:
    """
    theta: azimuthal angle in [-pi, pi]
    phi: polar angle in [0, pi]
    """
    phi = (0.5 - torch.arange(H, device=device) / H) * torch.pi
    theta = (1 - torch.arange(W, device=device) / W) * 2 * torch.pi - torch.pi
    [phi, theta] = torch.meshgrid([phi, theta], indexing="ij")
    angles = torch.stack([phi, theta])
    return angles[None]

class LiDARUtility(nn.Module):
    def __init__(
        self,
        resolution: tuple[int],
        depth_format: Literal["log_depth", "inverse_depth", "depth"],
        min_depth: float,
        max_depth: float,
        ray_angles: torch.Tensor = None,
    ):
        super().__init__()
        assert depth_format in ("log_depth", "inverse_depth", "depth")
        self.resolution = resolution
        self.depth_format = depth_format
        self.min_depth = min_depth
        self.max_depth = max_depth
        if ray_angles is None:
            ray_angles = generate_polar_coords(*resolution)
        else:
            assert ray_angles.ndim == 4 and ray_angles.shape[1] == 2
        ray_angles = F.interpolate(
            ray_angles,
            size=(self.resolution[0], self.resolution[1]),
            mode="nearest-exact",
        )
        self.register_buffer("ray_angles", ray_angles.float())

    @staticmethod
    def denormalize(x: torch.Tensor) -> torch.Tensor:
        """Scale from [-1, +1] to [0, 1]"""
        return (x + 1) / 2

    @staticmethod
    def normalize(x: torch.Tensor) -> torch.Tensor:
        """Scale from [0, 1] to [-1, +1]"""
        return x * 2 - 1

    @torch.no_grad()
    def to_xyz(self, metric: torch.Tensor) -> torch.Tensor:
        assert metric.dim() == 4
        mask = (metric > self.min_depth) & (metric < self.max_depth)
        phi = self.ray_angles[:, [0]]
        theta = self.ray_angles[:, [1]]
        grid_x = metric * phi.cos() * theta.cos()
        grid_y = metric * phi.cos() * theta.sin()
        grid_z = metric * phi.sin()
        xyz = torch.cat((grid_x, grid_y, grid_z), dim=1)
        xyz = xyz * mask.float()
        return xyz

    @torch.no_grad()
    def convert_depth(
        self,
        metric: torch.Tensor,
        mask: torch.Tensor | None = None,
        depth_format: str = None,
    ) -> torch.Tensor:
        """
        Convert metric depth in [0, `max_depth`] to normalized depth in [0, 1].
        """
        if depth_format is None:
            depth_format = self.depth_format
        if mask is None:
            mask = self.get_mask(metric)
        if depth_format == "log_depth":
            normalized = torch.log2(metric + 1) / np.log2(self.max_depth + 1)
        elif depth_format == "inverse_depth":
            normalized = self.min_depth / metric.add(1e-8)
        elif depth_format == "depth":
            normalized = metric.div(self.max_depth)
        else:
            raise ValueError
        normalized = normalized.clamp(0, 1) * mask
        return normalized

    @torch.no_grad()
    def revert_depth(
        self,
        normalized: torch.Tensor,
        image_format: str = None,
    ) -> torch.Tensor:
        """
        Revert normalized depth in [0, 1] back to metric depth in [0, `max_depth`].
        """
        if image_format is None:
            image_format = self.depth_format
        if image_format == "log_depth":
            metric = torch.exp2(normalized * np.log2(self.max_depth + 1)) - 1
        elif image_format == "inverse_depth":
            metric = self.min_depth / normalized.add(1e-8)
        elif image_format == "depth":
            metric = normalized.mul(self.max_depth)
        else:
            raise ValueError
        return metric * self.get_mask(metric)

    def get_mask(self, metric):
        mask = (metric > self.min_depth) & (metric < self.max_depth)
        return mask.float()
