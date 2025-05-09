import torch
from torch import nn
import torch.nn.functional as F
from . import weights_init, l1, l2, hinge_d_loss, vanilla_d_loss, measure_perplexity, square_dist_loss
from .geometric import GeoConverter
from .discriminator import NLayerDiscriminator, LiDARNLayerDiscriminator, LiDARNLayerDiscriminatorV2, PointNet
from .perceptual import PerceptualLoss
from .chamfer.chamfer3D.dist_chamfer_3D import chamfer_3DDist
from .utils import chamfer_distance_cuda

VERSION2DISC = {'v0': NLayerDiscriminator, 'v1': LiDARNLayerDiscriminator, 'v2': LiDARNLayerDiscriminatorV2}


class VQGeoLPIPSWithDiscriminator(nn.Module):
    def __init__(self, disc_start, codebook_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_out_channels=1, disc_factor=1.0, disc_weight=1.0,
                 mask_factor=0.0, chamfer_factor=0.0, smooth_factor=0.1, norm_factor=0.1, use_actnorm=False, disc_conditional=False,
                 disc_ndf=64, disc_loss="hinge", n_classes=None, pixel_loss="l1", disc_version='v1',
                 geo_factor=1.0, curve_length=4, perceptual_factor=1.0, perceptual_type='rangenet_final',
                 dataset_config=dict()):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        assert pixel_loss in ["l1", "l2"]
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        self.mask_factor = mask_factor
        self.geo_factor = geo_factor

        # scale of reconstruction loss
        self.rec_scale = 1
        if mask_factor > 0:
            self.rec_scale += 1.
        if geo_factor > 0:
            self.rec_scale += 1.
        if perceptual_factor > 0:
            self.rec_scale += 1.

        if pixel_loss == "l1":
            self.pixel_loss = l1
        else:
            self.pixel_loss = l2

        self.perceptual_factor = perceptual_factor
        self.chamfer_factor = chamfer_factor
        self.smooth_factor = smooth_factor
        self.norm_factor = norm_factor
        if perceptual_factor > 0.:
            print(f"{self.__class__.__name__}: Running with LPIPS.")
            self.perceptual_loss = PerceptualLoss(perceptual_type, dataset_config.depth_scale,
                                                  dataset_config.log_scale).eval()

        disc_cls = VERSION2DISC[disc_version]
        self.discriminator = disc_cls(input_nc=disc_in_channels,
                                      output_nc=disc_out_channels,
                                      n_layers=disc_num_layers,
                                      use_actnorm=use_actnorm,
                                      ndf=disc_ndf).apply(weights_init)
        self.discriminator_iter_start = disc_start
        if disc_loss == "hinge":
            self.disc_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.disc_loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")
        print(f"VQGeoLPIPSWithDiscriminator running with {disc_loss} loss.")
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional
        self.n_classes = n_classes

        self.geometry_converter = GeoConverter(curve_length, False, dataset_config)  # force converting xyz output
        self.geo_loss = square_dist_loss

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, codebook_loss, inputs, reconstructions, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train", predicted_indices=None, masks=None):
        input_coord = self.geometry_converter(inputs)
        rec_coord = self.geometry_converter(reconstructions[:, 0:1].contiguous())
        gt_depth = self.geometry_converter.batch_rescale_depth(inputs)
        pred_depth = self.geometry_converter.batch_rescale_depth(reconstructions[:, 0:1].contiguous())

        ############# Reconstruction #############
        # pixel reconstruction loss
        if self.mask_factor > 0 and masks is not None:
            pixel_rec_loss = self.pixel_loss(inputs.contiguous(), reconstructions[:, 0:1].contiguous())
            mask_rec_loss = self.pixel_loss(masks.contiguous(), reconstructions[:, 1:2].contiguous()) * self.mask_factor
        else:
            pixel_rec_loss = self.pixel_loss(inputs.contiguous(), reconstructions.contiguous())
            mask_rec_loss = torch.tensor(0.0)

        # geometry reconstruction loss (bev)
        if self.geo_factor > 0:
            geo_rec_loss = self.geo_loss(input_coord[:, :2], rec_coord[:, :2]) * self.geo_factor
        else:
            geo_rec_loss = torch.tensor(0.0)

        # perceptual loss
        if self.perceptual_factor > 0:
            perceptual_loss = self.perceptual_loss((inputs.contiguous(), input_coord),
                                                   (reconstructions[:, 0:1].contiguous(), rec_coord)) * self.perceptual_factor
        else:
            perceptual_loss = torch.tensor(0.0)

        # smooth loss
        if self.smooth_factor > 0:
            gt_grad_x = gt_depth[:, 0, :, :-1] - gt_depth[:, 0, :, 1:]
            gt_grad_y = gt_depth[:, 0, :-1, :] - gt_depth[:, 0, 1:, :]
            mask_x = (torch.where(gt_depth[:, 0, :, :-1] > 0, 1, 0) *
                      torch.where(gt_depth[:, 0, :, 1:] > 0, 1, 0))
            mask_y = (torch.where(gt_depth[:, 0, :-1, :] > 0, 1, 0) *
                      torch.where(gt_depth[:, 0, 1:, :] > 0, 1, 0))

            grad_clip = 0.01
            grad_mask_x = torch.where(torch.abs(gt_grad_x) < grad_clip, 1, 0) * mask_x
            grad_mask_x = grad_mask_x.bool()
            grad_mask_y = torch.where(torch.abs(gt_grad_y) < grad_clip, 1, 0) * mask_y
            grad_mask_y = grad_mask_y.bool()

            pred_grad_x = pred_depth[:, 0, :, :-1] - pred_depth[:, 0, :, 1:]
            pred_grad_y = pred_depth[:, 0, :-1, :] - pred_depth[:, 0, 1:, :]
            loss_smooth = (F.l1_loss(pred_grad_x[grad_mask_x], gt_grad_x[grad_mask_x])
                           + F.l1_loss(pred_grad_y[grad_mask_y], gt_grad_y[grad_mask_y])) * self.smooth_factor

        else:
            loss_smooth = torch.tensor(0.0)

        # norm loss
        if self.norm_factor > 0:
            surf_normal = self.geometry_converter.batch_range2normal(input_coord)
            render_normal = self.geometry_converter.batch_range2normal(rec_coord)
            loss_normal_consistency = (1 - (render_normal * surf_normal).sum(dim=1)[:, 1:-1, 1:-1]).mean() * self.norm_factor

        else:
            loss_normal_consistency = torch.tensor(0.0)

        # overall reconstruction loss
        rec_loss = (pixel_rec_loss + mask_rec_loss + geo_rec_loss + perceptual_loss) / self.rec_scale
        nll_loss = rec_loss
        nll_loss = torch.mean(nll_loss) + loss_smooth + loss_normal_consistency

        ############# GAN #############
        disc_factor = 0. if global_step > self.discriminator_iter_start else self.disc_factor
        # update generator (input: img, mask, coord, [cond])
        if optimizer_idx == 0:
            disc_recons = reconstructions.contiguous()
            if self.geo_factor > 0:
                disc_recons = torch.cat((disc_recons, rec_coord[:, :2].contiguous()), dim=1)
            if cond is not None and self.disc_conditional:
                disc_recons = torch.cat((disc_recons, cond), dim=1)
            logits_fake = self.discriminator(disc_recons)

            # adversarial loss
            g_loss = -torch.mean(logits_fake)

            try:
                d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
            except RuntimeError:
                assert not self.training
                d_weight = torch.tensor(0.0)

            loss = nll_loss + d_weight * disc_factor * g_loss + self.codebook_weight * codebook_loss.mean()

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/quant_loss".format(split): codebook_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/pix_rec_loss".format(split): pixel_rec_loss.detach().mean(),
                   "{}/geo_rec_loss".format(split): geo_rec_loss.detach().mean(),
                   "{}/mask_rec_loss".format(split): mask_rec_loss.detach().mean(),
                   "{}/perceptual_loss".format(split): perceptual_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean()}

            if predicted_indices is not None:
                assert self.n_classes is not None
                with torch.no_grad():
                    perplexity, cluster_usage = measure_perplexity(predicted_indices, self.n_classes)
                log[f"{split}/perplexity"] = perplexity
                log[f"{split}/cluster_usage"] = cluster_usage
            return loss, log

        # update discriminator (input: img, mask, coord, [cond])
        if optimizer_idx == 1:
            disc_inputs, disc_recons = inputs.contiguous().detach(), reconstructions.contiguous().detach()
            if self.mask_factor > 0 and masks is not None:
                disc_inputs = torch.cat((disc_inputs, masks.contiguous().detach()), dim=1)
            if self.geo_factor > 0:
                disc_inputs = torch.cat((disc_inputs, input_coord[:, :2].contiguous()), dim=1)
                disc_recons = torch.cat((disc_recons, rec_coord[:, :2].contiguous()), dim=1)
            if cond is not None:
                disc_inputs = torch.cat((disc_inputs, cond), dim=1)
                disc_recons = torch.cat((disc_recons, cond), dim=1)
            logits_real = self.discriminator(disc_inputs)
            logits_fake = self.discriminator(disc_recons)

            # gan loss
            d_loss = self.disc_loss(logits_real, logits_fake) * disc_factor

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()}

            return d_loss, log

    def forward_s2(self, inputs, reconstructions, split='train'):
        input_coord = self.geometry_converter(inputs)
        rec_coord = self.geometry_converter(reconstructions[:, 0:1].contiguous())
        gt_depth = self.geometry_converter.batch_rescale_depth(inputs)
        pred_depth = self.geometry_converter.batch_rescale_depth(reconstructions[:, 0:1].contiguous())

        loss_lidar = F.l1_loss(inputs, reconstructions)

        # chamfer loss
        if self.chamfer_factor > 0:
            cham_fn = chamfer_3DDist()
            gt_points = input_coord.flatten(2,3).permute(0,2,1)
            pred_points = rec_coord.flatten(2,3).permute(0,2,1)
            dist1, dist2, _, _ = cham_fn(pred_points, gt_points)
            loss_chamfer = (dist1.mean() + dist2.mean()) * self.chamfer_factor

        else:
            loss_chamfer = torch.tensor(0.0)

        # smooth loss
        if self.smooth_factor > 0:
            gt_grad_x = gt_depth[:, 0, :, :-1] - gt_depth[:, 0, :, 1:]
            gt_grad_y = gt_depth[:, 0, :-1, :] - gt_depth[:, 0, 1:, :]
            mask_x = (torch.where(gt_depth[:, 0, :, :-1] > 0, 1, 0) *
                      torch.where(gt_depth[:, 0, :, 1:] > 0, 1, 0))
            mask_y = (torch.where(gt_depth[:, 0, :-1, :] > 0, 1, 0) *
                      torch.where(gt_depth[:, 0, 1:, :] > 0, 1, 0))

            grad_clip = 0.01
            grad_mask_x = torch.where(torch.abs(gt_grad_x) < grad_clip, 1, 0) * mask_x
            grad_mask_x = grad_mask_x.bool()
            grad_mask_y = torch.where(torch.abs(gt_grad_y) < grad_clip, 1, 0) * mask_y
            grad_mask_y = grad_mask_y.bool()

            pred_grad_x = pred_depth[:, 0, :, :-1] - pred_depth[:, 0, :, 1:]
            pred_grad_y = pred_depth[:, 0, :-1, :] - pred_depth[:, 0, 1:, :]
            loss_smooth = (F.l1_loss(pred_grad_x[grad_mask_x], gt_grad_x[grad_mask_x])
                           + F.l1_loss(pred_grad_y[grad_mask_y], gt_grad_y[grad_mask_y])) * self.smooth_factor

        else:
            loss_smooth = torch.tensor(0.0)

        # norm loss
        if self.norm_factor > 0:
            surf_normal = self.geometry_converter.batch_range2normal(input_coord)
            render_normal = self.geometry_converter.batch_range2normal(rec_coord)
            loss_normal_consistency = (1 - (render_normal * surf_normal).sum(dim=1)[:, 1:-1, 1:-1]).mean() * self.norm_factor

        else:
            loss_normal_consistency = torch.tensor(0.0)


        rec_loss = loss_lidar.mean() + loss_chamfer.mean() + loss_smooth.mean() + loss_normal_consistency.mean()

        log = {"{}/loss_lidar".format(split): rec_loss.clone().detach().mean()}

        return rec_loss, log
    
class VQGeoLPIPSWithDiscriminator1D(nn.Module):
    def __init__(self, discriminator_config, dataset_config=dict(), disc_conditional=False):
        super().__init__()
        self.discriminator_weight = 1.0
        self.discriminator = PointNet(
            pts_dim=discriminator_config.pts_dim,
            x=discriminator_config.latent_times,
            cls_num=discriminator_config.cls_num
        )
        self.classifical_loss = nn.CrossEntropyLoss()

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, inputs, fg_class, reconstructions, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train", predicted_indices=None, masks=None):

        if optimizer_idx == 0:
            disc_recons = reconstructions.permute(0,2,1).contiguous()
            rec_loss = chamfer_distance_cuda(inputs, reconstructions)
            logits_fake, global_logits_fake = self.discriminator(disc_recons)
            # adversarial loss
            g_loss = -torch.mean(global_logits_fake)
            perception_loss = self.classifical_loss(logits_fake, fg_class.squeeze().long())

            try:
                d_weight = self.calculate_adaptive_weight(rec_loss, g_loss, last_layer=last_layer)
            except RuntimeError:
                assert not self.training
                d_weight = torch.tensor(0.0)

            loss = rec_loss + d_weight * g_loss + perception_loss*0.1

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                    "{}/rec_loss".format(split): rec_loss.detach().mean(),
                    "{}/disc_loss".format(split): g_loss.detach().mean(),
                    }

            if predicted_indices is not None:
                assert self.n_classes is not None
                with torch.no_grad():
                    perplexity, cluster_usage = measure_perplexity(predicted_indices, self.n_classes)
                log[f"{split}/perplexity"] = perplexity
                log[f"{split}/cluster_usage"] = cluster_usage
            return loss, log

        if optimizer_idx == 1:
            disc_inputs, disc_recons = inputs.contiguous().permute(0,2,1).detach(), reconstructions.permute(0,2,1).contiguous().detach()
            logits_real, global_logits_real = self.discriminator(disc_inputs)
            _, global_logits_fake = self.discriminator(disc_recons)
            perception_loss = self.classifical_loss(logits_real, fg_class.squeeze().long())

            # gan loss
            d_loss = hinge_d_loss(global_logits_real, global_logits_fake) + perception_loss

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): global_logits_real.detach().mean(),
                   "{}/logits_fake".format(split): global_logits_fake.detach().mean()}

            return d_loss, log