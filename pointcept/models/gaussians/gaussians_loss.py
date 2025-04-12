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

        # ------------------------ depth grad loss -------------------------#
        pred_grad_x = torch.abs(pred_depth[:, :, :-1] - pred_depth[:, :, 1:])
        gt_grad_x = torch.abs(gt_depth[:, :, :-1] - gt_depth[:, :, 1:])
        grad_clip_x = 0.01
        grad_mask_x = torch.where(gt_grad_x < grad_clip_x, 1, 0)
        mask_dx = gt_ray_drop[:, :, :-1] * grad_mask_x
        grad_loss = self.l1_loss(pred_grad_x * mask_dx, gt_grad_x * mask_dx)

        # ------------------------ raydrop grad loss -------------------------#
        pred_raydrop_grad_x = torch.abs(pred_ray_drop[:, :, :-1] - pred_ray_drop[:, :, 1:])
        gt_raydrop_grad_x = torch.abs(gt_ray_drop[:, :, :-1] - gt_ray_drop[:, :, 1:])
        raydrop_grad_loss = self.l1_loss(pred_raydrop_grad_x*gt_raydrop_grad_x, gt_raydrop_grad_x)


        loss = raydrop_loss + depth_loss + grad_loss + raydrop_grad_loss

        return loss