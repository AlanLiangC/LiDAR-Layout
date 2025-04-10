import torch
from torch import nn
from pointcept.models.builder import MODELS

@MODELS.register_module("GSLoss")
class GaussianLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ray_drop_loss = nn.MSELoss()

    def l1_loss(self, gred_range, gt_range):
        return torch.abs((gred_range - gt_range)).mean()

    def forward(self, point):
        # raydrop
        gt_ray_drop = point.ray_drop
        pred_ray_drop = point.pred_ray_drop
        raydrop_loss = self.ray_drop_loss(gt_ray_drop, pred_ray_drop).mean()

        # depth
        gt_depth = point.range_img * gt_ray_drop
        pred_depth = point.pred_range * gt_ray_drop
        depth_loss = self.l1_loss(pred_depth, gt_depth)

        loss = raydrop_loss + depth_loss

        return loss