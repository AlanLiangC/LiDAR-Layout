import torch
import numpy as np
from .autoencoder import VQModel
from .utils import range2pcd_gpu, range2feature_gpu, scale_range
from lidm.modules.gaussians.gaussian_model import GaussianModel
from lidm.modules.gaussians.gaussian_renderer import render
from lidm.modules.gaussians.utils.cameras import Camera
from lidm.modules.diffusion import model_lidm

class VQModel_Gaus(VQModel):
    def __init__(self, ddconfig, n_embed, embed_dim, lossconfig=None, ckpt_path=None, ignore_keys=[], 
                 image_key="image", colorize_nlabels=None, monitor=None, batch_resize_range=None, 
                 scheduler_config=None, lr_g_factor=1, remap=None, sane_index_shape=False, use_ema=False, 
                 lib_name='ldm', use_mask=False, dataset_config=None, **kwargs):
        
        super().__init__(ddconfig, n_embed, embed_dim, lossconfig, ckpt_path, ignore_keys, image_key, 
                         colorize_nlabels, monitor, batch_resize_range, scheduler_config, lr_g_factor, 
                         remap, sane_index_shape, use_ema, lib_name, use_mask, **kwargs)
        
        self.gaus_decoder = model_lidm.Gaus_Decoder(**ddconfig.gdconfig)
        dataset_config = ddconfig.gdconfig.dataset_config
        self.img_size = dataset_config.size
        self.fov = [dataset_config.fov[1], dataset_config.fov[0]]
        self.depth_range = dataset_config.depth_range
        self.depth_scale = dataset_config.depth_scale
        self.log_scale = dataset_config.log_scale
        self.basic_gaus = GaussianModel()
        self.build_camera_bg = False

        self.xyz_scale_factor = 1.0

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def build_camera(self): # nuscenes TODO: Other dataset
        l2c = np.array([1, 0, 0, 0,
                        0, 0, -1, 0,
                        0, 1, 0, 0,
                        0, 0, 0, 1]).reshape(4, 4)
        
        R = np.transpose(l2c[:3, :3])
        T = l2c[:3, 3]

        R_back = R @ np.array([-1, 0, 0,
                                0, 1, 0,
                                0, 0, -1]).reshape(3, 3)
        T_back = T * np.array([-1, 1, -1])

        self.forward_view_point = Camera(
            colmap_id=0,
            uid=0,
            R=R,
            T=T,
            vfov=self.fov,
            hfov=[ -90, 90 ],
            data_device=self.device,
            resolution=[int(self.img_size[1] // 2), self.img_size[0]],
            towards='forwards'
        )

        self.backward_view_point = Camera(
            colmap_id=1,
            uid=1,
            R=R_back,
            T=T_back,
            vfov=self.fov,
            hfov=[ -90, 90 ],
            data_device=self.device,
            resolution=[int(self.img_size[1] // 2), self.img_size[0]],
            towards='backwards'
        )

    def custom_to_pcd(self, x):
        x = (torch.clip(x, -1., 1.) + 1.) / 2.
        # xyz_cpu = range2pcd(x.squeeze().detach().cpu().numpy(), fov=[self.fov[1], self.fov[0]], depth_range=self.depth_range, depth_scale=self.depth_scale, log_scale=self.log_scale)
        # np.savetxt('/home/alan/AlanLiang/Projects/AlanLiang/LiDAR-Layout/scripts/ALTest/points_cpu.txt', xyz_cpu[0])
        xyz, mask = range2pcd_gpu(x, fov=[self.fov[1], self.fov[0]], depth_range=self.depth_range, depth_scale=self.depth_scale, log_scale=self.log_scale)
        # np.savetxt('/home/alan/AlanLiang/Projects/AlanLiang/LiDAR-Layout/scripts/ALTest/points_gpu.txt', xyz.squeeze().detach().cpu().numpy())

        return xyz, mask

    def custom_to_feature(self, x, mask=None, is_sh=False):
        return range2feature_gpu(x, mask, is_sh)

    def render_range(self, depth, rot_out, scale_out, opacity_out, sh_out):
        if not self.build_camera_bg:
            self.bg_color = torch.tensor([0, 0, 0, 1], dtype=torch.float32, device=self.device)
            self.build_camera()
            self.build_camera_bg = True

        depth_all = []
        for batch_depth, batch_rot, batch_scale_out, batch_opacity_out, batch_sh_out in zip(depth, rot_out, scale_out, opacity_out, sh_out):
            xyz, mask = self.custom_to_pcd(batch_depth)
            rot = self.custom_to_feature(batch_rot,mask)
            scale = self.custom_to_feature(batch_scale_out,mask)
            opacity = self.custom_to_feature(batch_opacity_out,mask)
            sh = self.custom_to_feature(batch_sh_out,mask, is_sh=True)
            self.basic_gaus.create_from_pcd(xyz, rot, scale, opacity, sh)
            render_pkg = render(self.forward_view_point, self.basic_gaus, bg_color=self.bg_color, scale_factor=self.xyz_scale_factor, device=self.device)
            forward_depth = render_pkg["depth"]
            render_pkg = render(self.backward_view_point, self.basic_gaus, bg_color=self.bg_color, scale_factor=self.xyz_scale_factor, device=self.device)
            backward_depth = render_pkg["depth"]
            batch_depth = torch.cat([forward_depth, backward_depth], dim=-1)
            depth_all.append(batch_depth)
        return torch.stack(depth_all, dim=0)

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        # decode depth
        dec_depth = self.decoder(quant)
        # decode gaus
        rot_out, scale_out, opacity_out, sh_out = self.gaus_decoder(quant)
        # gaus rander
        render_range = self.render_range(dec_depth, rot_out, scale_out, opacity_out, sh_out)
        render_range = scale_range(render_range, self.depth_scale, self.log_scale)
        return dec_depth, render_range
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        # https://github.com/pytorch/pytorch/issues/37142
        # try not to fool the heuristics
        x = self.get_input(batch, self.image_key)
        m = self.get_mask(batch) if self.use_mask else None
        x_rec, qloss, ind = self(x, return_pred_indices=True)
        xrec_s1, xrec_s2 = x_rec

        if optimizer_idx == 0:
            # autoencoder
            aeloss_s1, log_dict_ae_s1 = self.loss(qloss, x, xrec_s1, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train",
                                            predicted_indices=None, masks=m)
            
            aeloss_s2, log_dict_ae_s2 = self.loss(qloss, x, xrec_s2, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train",
                                            predicted_indices=None, masks=m)
            self.log_dict(log_dict_ae_s1, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae_s2, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            # if self.global_step < 1000:
            #     return aeloss_s1 + 0.1 * aeloss_s2
            return aeloss_s1 + aeloss_s2
            # return aeloss_s1

        if optimizer_idx == 1:
            # discriminator
            discloss_s1, log_dict_disc_s1 = self.loss(qloss, x, xrec_s1, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train",
                                                masks=m)
            discloss_s2, log_dict_disc_s2 = self.loss(qloss, x, xrec_s2, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train",
                                                masks=m)
            self.log_dict(log_dict_disc_s1, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc_s2, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            # if self.global_step < 1000:
            #     return discloss_s1 + 0.1 * discloss_s2
            return discloss_s1 + discloss_s2
            # return discloss_s1

    def validation_step(self, batch, batch_idx):
        log_dict = self._validation_step(batch, batch_idx)
        if self.use_ema:
            with self.ema_scope():
                log_dict_ema = self._validation_step(batch, batch_idx, suffix="_ema")
        return log_dict

    def _validation_step(self, batch, batch_idx, suffix=""):
        x = self.get_input(batch, self.image_key)
        m = self.get_mask(batch) if self.use_mask else None
        xrec, qloss, ind = self(x, return_pred_indices=True)
        xrec_s1, xrec_s2 = xrec
        aeloss_s1, log_dict_ae_s1 = self.loss(qloss, x, xrec_s1, 0,
                                        self.global_step,
                                        last_layer=self.get_last_layer(),
                                        split="val" + suffix,
                                        predicted_indices=None,
                                        masks=m
                                        )
        aeloss_s2, log_dict_ae_s2 = self.loss(qloss, x, xrec_s2, 0,
                                        self.global_step,
                                        last_layer=self.get_last_layer(),
                                        split="val" + suffix,
                                        predicted_indices=None,
                                        masks=m
                                        )

        discloss_s1, log_dict_disc_s1 = self.loss(qloss, x, xrec_s1, 1,
                                            self.global_step,
                                            last_layer=self.get_last_layer(),
                                            split="val" + suffix,
                                            predicted_indices=None,
                                            masks=m
                                            )
        discloss_s2, log_dict_disc_s2 = self.loss(qloss, x, xrec_s1, 1,
                                            self.global_step,
                                            last_layer=self.get_last_layer(),
                                            split="val" + suffix,
                                            predicted_indices=None,
                                            masks=m
                                            )
        
        rec_loss_s1 = log_dict_ae_s1[f"val{suffix}/rec_loss"]
        self.log(f"val{suffix}/rec_loss", rec_loss_s1,
                 prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"val{suffix}/aeloss", aeloss_s1,
                 prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        del log_dict_ae_s1[f"val{suffix}/rec_loss"]
        self.log_dict(log_dict_ae_s1)
        self.log_dict(log_dict_disc_s1)

        rec_loss_s2 = log_dict_ae_s2[f"val{suffix}/rec_loss"]
        self.log(f"val{suffix}/rec_loss", rec_loss_s2,
                 prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"val{suffix}/aeloss", aeloss_s2,
                 prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        del log_dict_ae_s2[f"val{suffix}/rec_loss"]
        self.log_dict(log_dict_ae_s2)
        self.log_dict(log_dict_disc_s2)

        return self.log_dict
    
    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, plot_ema=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        # self.dec_depth = x.requires_grad_(True)
        if only_inputs:
            log["inputs"] = x
            return log
        xrec, _ = self(x)
        xrec_s1, xrec_s2 = xrec
        if self.use_mask:
            mask = xrec_s1[:, 1:2] < 0.
            xrec_s1 = xrec_s1[:, 0:1]
            xrec_s1[mask] = -1.
            xrec_s2 = xrec_s2[:, 0:1]
            xrec_s2[mask] = -1.
        log["inputs"] = x
        log["reconstructions_s1"] = xrec_s1
        log["reconstructions_s2"] = xrec_s2
        if plot_ema:
            with self.ema_scope():
                xrec_ema, _ = self(x)
                log["reconstructions_ema"] = xrec_ema
        return log