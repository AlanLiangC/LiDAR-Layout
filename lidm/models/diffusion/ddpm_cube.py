import gc
import torch
from pytorch_lightning.utilities.distributed import rank_zero_only
import fvdb.nn as fvnn
from fvdb.nn import VDBTensor
from .ddpm import LatentDiffusion
from ...modules.unets.embedder_util import get_embedder
from ...utils.misc_utils import log_txt_as_img, exists, default, ismap, isimage, mean_flat, count_params, instantiate_from_config, print_fn, preprocess_angle2sincos
from ...modules.basic import make_beta_schedule, extract_into_sparse_tensor, noise_like
from ...models.diffusion.ddim import DDIMCubeSampler

class CubeLatentDiffusion(LatentDiffusion):
    def __init__(self, first_stage_config, cond_stage_config, cube_condition_config, num_timesteps_cond=None, 
                 cond_stage_key="image", cond_stage_trainable=False, concat_mode=True, 
                 cond_stage_forward=None, conditioning_key=None, scale_factor=1, scale_by_std=False, 
                 use_mask=False, *args, **kwargs):
        unet_config = kwargs['unet_config']
        unet_num_blocks = first_stage_config.params.unetconfig.params.num_blocks
        num_input_channels = first_stage_config.params.unetconfig.params.f_maps * 2 ** (unet_num_blocks - 1) # Fix by using VAE hparams
        num_input_channels = int(num_input_channels / first_stage_config.params.unetconfig.params.cut_ratio)

        out_channels = num_input_channels
        num_classes = None
        use_spatial_transformer = False
        context_dim=None

        if cube_condition_config.get('use_pos_embed_high', False):
            embed_fn, input_ch = get_embedder(5)
            num_input_channels += input_ch

        unet_config.params.num_input_channels = num_input_channels
        unet_config.params.out_channels = out_channels
        unet_config.params.num_classes = num_classes
        unet_config.params.use_spatial_transformer = use_spatial_transformer
        unet_config.params.context_dim = context_dim
        kwargs['unet_config'] = unet_config

        super().__init__(first_stage_config, cond_stage_config, num_timesteps_cond, cond_stage_key, 
                         cond_stage_trainable, concat_mode, cond_stage_forward, conditioning_key, 
                         scale_factor, scale_by_std, use_mask, *args, **kwargs)

        self.cube_condition_config = cube_condition_config
        self.pos_embedder = embed_fn

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        # only for very first batch
        if self.scale_by_std and self.current_epoch == 0 and self.global_step == 0 and batch_idx == 0 and not self.restarted_from_ckpt:
            assert self.scale_factor == 1., 'rather not use custom rescaling and std-rescaling simultaneously'
            # set rescale weight to 1./std of encodings
            self.print_fn("### USING STD-RESCALING ###")
            z = self.get_input(batch)[0]
            z = z.feature.jdata
            del self.scale_factor
            self.register_buffer('scale_factor', 1. / z.flatten().std())
            self.print_fn(f"setting self.scale_factor to {self.scale_factor}")
            self.print_fn("### USING STD-RESCALING ###")

    @torch.no_grad()
    def encode_first_stage(self, batch):
        return self.first_stage_model._encode(batch, use_mode=False)
    
    def get_pos_embed_high(self, h):
        xyz = h[:, :3] # N, 3
        xyz = self.pos_embedder(xyz) # N, C
        return xyz

    @torch.no_grad()
    def get_input(self, batch, k=None, return_first_stage_outputs=False, force_c_encode=False,
                  cond_key=None, return_original_cond=False, bs=None):

        # encoding
        encoder_posterior = self.encode_first_stage(batch)
        z = self.get_first_stage_encoding(encoder_posterior).detach()
        if self.model.conditioning_key is not None:
            if cond_key is None:
                cond_key = self.cond_stage_key
            if cond_key != self.first_stage_key:
                if cond_key in ['caption', 'bbox', 'center', 'camera']:
                    xc = batch[cond_key]
                elif cond_key in ['class_label']:
                    xc = batch
                else:
                    xc = super().get_input(batch, cond_key).to(self.device)
            else:
                xc = x
            # if bs is not None:
            #     xc = xc[:bs]
            if not self.cond_stage_trainable or force_c_encode:
                if isinstance(xc, (dict, list)):
                    c = self.get_learned_conditioning(xc)
                else:
                    c = self.get_learned_conditioning(xc.to(self.device))
            else:
                c = xc
            if bs is not None:
                if not isinstance(c, dict):
                    c = c[:bs]

            if self.use_positional_encodings:
                pos_x, pos_y = self.compute_latent_shifts(batch)
                ckey = __conditioning_keys__[self.model.conditioning_key]
                c = {ckey: c, 'pos_x': pos_x, 'pos_y': pos_y}

        else:
            c = None
            xc = None
            if self.use_positional_encodings:
                pos_x, pos_y = self.compute_latent_shifts(batch)
                c = {'pos_x': pos_x, 'pos_y': pos_y}
        out = [z, c]
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([None, xrec])
        if return_original_cond:
            out.append(xc)
        return out

    def shared_step(self, batch, **kwargs):
        x, c = self.get_input(batch)
        loss = self(x, c)
        return loss

    def p_losses(self, x_start, cond, t, noise=None, **kwargs):
        latent_data = x_start.feature.jdata
        noise = default(noise, lambda: torch.randn_like(latent_data))
        timesteps_sparse = t.long()
        timesteps_sparse = timesteps_sparse[x_start.feature.jidx.long()] # N, 1
        x_noisy = self.q_sample(x_start=latent_data, t=timesteps_sparse, noise=noise)
        x_noisy = VDBTensor(grid=x_start.grid, feature=x_start.grid.jagged_like(x_noisy))
        if self.cube_condition_config.get('use_pos_embed_high', False):
            pos_embed = self.get_pos_embed_high(x_noisy.grid.ijk.jdata)
            x_noisy = VDBTensor.cat([x_noisy, pos_embed], dim=1)
        model_output = self.apply_model(x_noisy, t, cond)
        pred = model_output.feature.jdata
        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError()

        # simple loss
        loss_simple = self.get_loss(pred, target, mean=False).mean()
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        # vlb loss
        loss_vlb = self.get_loss(pred, target, mean=False).mean()
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})

        # total loss
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_sparse_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_sparse_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)


    def forward(self, x, c, *args, **kwargs):
        bsz = x.grid.grid_count
        t = torch.randint(0, self.num_timesteps, (bsz,), device=self.device).long()

        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:
                c = self.get_learned_conditioning(c)
            if self.shorten_cond_schedule:  # TODO: drop this option
                tc = self.cond_ids[t].to(self.device)
                c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))
        return self.p_losses(x, c, t, *args, **kwargs)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        if batch_idx % 100 == 0:
            gc.collect()
            torch.cuda.empty_cache()
        _, loss_dict_no_ema = self.shared_step(batch)
        with self.ema_scope():
            _, loss_dict_ema = self.shared_step(batch)
            loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)

    def training_step(self, batch, batch_idx):
        if batch_idx % 100 == 0:
            gc.collect()
            torch.cuda.empty_cache()
        loss, loss_dict = self.shared_step(batch)

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss
    

    @torch.no_grad()
    def decode_first_stage(self, z, predict_cids=False, force_not_quantize=False, return_res=False):

        latent_data = 1. / self.scale_factor * z.feature.jdata
        z = fvnn.VDBTensor(z.grid, z.grid.jagged_like(latent_data))
        res = self.first_stage_model.unet.FeaturesSet()
        return self.first_stage_model._decode(res, z, is_testing=False, return_res=return_res)
    
    @torch.no_grad()
    def sample_log(self, grids, cond, ddim, ddim_steps, **kwargs):
        if ddim:
            ddim_sampler = DDIMCubeSampler(self)
            samples, intermediates = ddim_sampler.sample(ddim_steps, grids, cond, verbose=self.verbose, **kwargs)

        return samples, intermediates

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=4, sample=True, ddim_steps=200, ddim_eta=1., return_keys=None,
                   quantize_denoised=False, inpaint=False, plot_denoise_rows=False, plot_progressive_rows=False,
                   plot_diffusion_rows=False, dset=None, **kwargs):

        use_ddim = ddim_steps is not None

        log = dict()
        z, c, x, xrec, xc = self.get_input(batch, self.first_stage_key,
                                           return_first_stage_outputs=True,
                                           force_c_encode=True,
                                           return_original_cond=True,
                                           bs=N)
        grids = z.grid

        with self.ema_scope("Plotting"):
            samples, z_denoise_row = self.sample_log(grids=grids, cond=c, ddim=use_ddim,
                                                        ddim_steps=ddim_steps, eta=ddim_eta)
        res = self.decode_first_stage(samples, return_res=True)
        for i in range(self.first_stage_model.geoconfig.tree_depth):
            temp_grid = res.structure_grid[i]
            log.update({
                f'decode_{i}': temp_grid.grid_to_world(temp_grid.ijk.float()).jdata.cpu().numpy()
            })
        temp_grid = batch['INPUT_PC']
        log.update({
            f'input_grid': temp_grid.grid_to_world(temp_grid.ijk.float()).jdata.cpu().numpy()
        })
        return log

