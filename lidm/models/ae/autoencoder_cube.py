import gc
import numpy as np
import torch
import fvdb
import fvdb.nn as fvnn
import pytorch_lightning as pl
import torch.nn.functional as F
from contextlib import contextmanager
from ...modules.ema import LitEma
from ...utils.misc_utils import instantiate_from_config
from ...modules.xcube.cube_base_encoder import Encoder
# from ...modules.xcube.cube_encoder_w_pt import Encoder

from .utils import point2voxel, reparametrize

class CubeAEModel(pl.LightningModule):
    def __init__(self,
                 monitor=None,
                 geoconfig=None,
                 edconfig=None,
                 unetconfig=None,
                 lossconfig=None,
                 ckpt_path=None,
                 ignore_keys=[],
                 scheduler_config=None,
                 lr_g_factor=1.0,
                 use_ema=False,
                 **kwargs
                 ):
        super().__init__()
        self.geoconfig = geoconfig
        self.tree_depth = geoconfig.tree_depth
        # callback
        if monitor is not None:
            self.monitor = monitor
        self.init_geoconfig()
        self.encoder = Encoder(**edconfig)           
        self.unet = instantiate_from_config(unetconfig)
        self.loss = instantiate_from_config(lossconfig) if lossconfig is not None else None
        if isinstance(geoconfig.voxel_size, int) or isinstance(geoconfig.voxel_size, float):
            self.geoconfig.voxel_size = [self.geoconfig.voxel_size] * 3
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.scheduler_config = scheduler_config
        self.lr_g_factor = lr_g_factor

    def init_geoconfig(self):
        pc_range = self.geoconfig.point_cloud_range
        voxel_size = self.geoconfig.voxel_size
        self.n_height = int((pc_range[5] - pc_range[2]) / voxel_size)
        self.n_length = int((pc_range[4] - pc_range[1]) / voxel_size)
        self.n_width = int((pc_range[3] - pc_range[0]) / voxel_size)

        self.input_grid = [1, self.n_height, self.n_length, self.n_width]
        self.output_grid = [1, self.n_height, self.n_length, self.n_width]

        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.offset = torch.nn.parameter.Parameter(
            torch.Tensor(self.pc_range[:3])[None, None, :], requires_grad=False
        )
        self.scaler = torch.nn.parameter.Parameter(
            torch.Tensor([self.voxel_size] * 3)[None, None, :], requires_grad=False
        )

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, batch, out: dict):
        input_grid = batch['input_grid']
        hash_tree = batch['hash_tree']
        unet_feat = self.encoder(input_grid, batch)
        unet_feat = fvnn.VDBTensor(input_grid, input_grid.jagged_like(unet_feat))
        unet_res, rec_grid, dist_features = self.unet(unet_feat, hash_tree)

        out.update({'tree': unet_res.structure_grid})
        out.update({
            'structure_features': unet_res.structure_features,
            'dist_features': dist_features,
        })
        out.update({'gt_grid': input_grid})
        out.update({'gt_tree': hash_tree})
        out.update({'rec_grid': rec_grid})

        return out

    def build_hash_tree(self, input_xyz):
        if self.geoconfig.use_fvdb_loader:
            if isinstance(input_xyz, dict):
                return input_xyz
            return self.build_hash_tree_from_grid(input_xyz)
        
        return self.build_hash_tree_from_points(input_xyz)

    def build_hash_tree_from_points(self, input_xyz):
        if isinstance(input_xyz, torch.Tensor):
            input_xyz = fvdb.JaggedTensor(input_xyz)
        elif isinstance(input_xyz, fvdb.JaggedTensor):
            pass
        else:
            raise NotImplementedError
        
        hash_tree = {}
        for depth in range(self.geoconfig.tree_depth):
            if depth != 0 and not self.geoconfig.use_hash_tree:
                break
            voxel_size = [sv * 2 ** depth for sv in self.geoconfig.voxel_size]
            origins = [sv / 2. for sv in voxel_size]            
            hash_tree[depth] = fvdb.sparse_grid_from_nearest_voxels_to_points(input_xyz, 
                                                                              voxel_sizes=voxel_size, 
                                                                              origins=origins)
        return hash_tree

    def build_hash_tree_from_grid(self, input_grid):
        hash_tree = {}
        input_xyz = input_grid.grid_to_world(input_grid.ijk.float())
        
        for depth in range(self.geoconfig.tree_depth):
            if depth != 0 and not self.geoconfig.use_hash_tree:
                break
            voxel_size = [sv * 2 ** depth for sv in self.geoconfig.voxel_size]
            origins = [sv / 2. for sv in voxel_size]
            
            if depth == 0:
                hash_tree[depth] = input_grid
            else:
                hash_tree[depth] = fvdb.sparse_grid_from_points(input_xyz, 
                                                                voxel_sizes=voxel_size, 
                                                                origins=origins)
        return hash_tree

    def get_input(self, batch):
        batch = point2voxel(batch, self.offset, self.scaler, self.input_grid)
        hash_tree = self.build_hash_tree(batch['INPUT_PC'])
        input_grid = hash_tree[0]
        batch.update({'input_grid': input_grid})
        if not self.geoconfig.use_hash_tree:
            hash_tree = None
        batch.update({'hash_tree': hash_tree})
        return batch

    def get_mask(self, batch):
        mask = batch['mask']
        # if len(mask.shape) == 3:
        #     mask = mask[..., None]
        return mask

    def training_step(self, batch, batch_idx):

        if batch_idx % 1 == 0:
            gc.collect()
            torch.cuda.empty_cache()

        out = {'idx': batch_idx}
        batch = self.get_input(batch)
        out = self(batch, out)
        loss_dict, metric_dict, latent_dict = self.loss(batch, out, 
                                                        compute_metric=False, 
                                                        global_step=self.global_step,
                                                        current_epoch=self.current_epoch)
        
        loss_sum = loss_dict.get_sum()
        self.log("train_loss/sum", loss_sum,
                 prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        if self.geoconfig.enable_anneal:
            self.log('anneal_kl_weight', self.loss.get_kl_weight(self.global_step),
                 prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
            
        self.log_dict(loss_dict)
        self.log_dict(metric_dict)
        self.log_dict(latent_dict)

        return loss_sum

    def validation_step(self, batch, batch_idx):
        log_dict = self._validation_step(batch, batch_idx)
        if self.use_ema:
            with self.ema_scope():
                log_dict_ema = self._validation_step(batch, batch_idx, suffix="_ema")
        return log_dict

    def _validation_step(self, batch, batch_idx, suffix=""):
        if batch_idx % 1 == 0:
            gc.collect()
            torch.cuda.empty_cache()

        out = {'idx': batch_idx}
        batch = self.get_input(batch)
        out = self(batch, out)
        loss_dict, metric_dict, latent_dict = self.loss(batch, out, 
                                                        compute_metric=True, 
                                                        global_step=self.global_step,
                                                        current_epoch=self.current_epoch)
        
        loss_sum = loss_dict.get_sum()
        self.log("val_loss", loss_sum,
                 prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_step", self.global_step,
                 prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)

        self.log_dict(loss_dict)
        self.log_dict(metric_dict)
        self.log_dict(latent_dict)

        return self.log_dict

    def configure_optimizers(self):
        lr_g = self.lr_g_factor * self.learning_rate

        opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
                                  list(self.unet.parameters()),
                                  lr=lr_g, betas=(0.5, 0.9))

        if self.scheduler_config is not None:
            # build scheduler
            import functools
            from torch.optim.lr_scheduler import LambdaLR
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt_ae, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }
            ]
            return [opt_ae], scheduler
        return [opt_ae], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        batch = self.get_input(batch)
        log = dict()
        input_grid = batch['input_grid']
        hash_tree = batch['hash_tree']
        unet_feat = self.encoder(input_grid, batch)
        unet_feat = fvnn.VDBTensor(input_grid, input_grid.jagged_like(unet_feat))
        res, output_x, _ = self.unet(unet_feat, hash_tree)
        # encode & decode
        for i in range(self.geoconfig.tree_depth):
            temp_grid = hash_tree[i]
            log.update({
                f'encode_{i}': temp_grid.grid_to_world(temp_grid.ijk.float()).jdata.cpu().numpy()
            })
            temp_grid = res.structure_grid[i]
            log.update({
                f'decode_{i}': temp_grid.grid_to_world(temp_grid.ijk.float()).jdata.cpu().numpy()
            })
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x
    
    @torch.no_grad()
    def _encode(self, batch, use_mode=False):
        batch = self.get_input(batch)
        input_grid = batch['input_grid']
        hash_tree = batch['hash_tree']

        unet_feat = self.encoder(input_grid, batch)
        unet_feat = fvnn.VDBTensor(input_grid, input_grid.jagged_like(unet_feat))
        _, x, mu, log_sigma = self.unet.encode(unet_feat, hash_tree=hash_tree)
        if use_mode:
            sparse_feature = mu
        else:
            sparse_feature = reparametrize(mu, log_sigma)
        
        return fvnn.VDBTensor(x.grid, x.grid.jagged_like(sparse_feature))
    
    def _decode(self, res, latents, is_testing=True):
        res, output_x = self.unet.decode(res, latents, is_testing=is_testing)
        return output_x

class CubeModelInterface(CubeAEModel):
    def __init__(self, monitor=None, geoconfig=None, edconfig=None, unetconfig=None, 
                 lossconfig=None, ckpt_path=None, ignore_keys=[], scheduler_config=None, 
                 lr_g_factor=1, use_ema=False, **kwargs):
        super().__init__(monitor, geoconfig, edconfig, unetconfig, lossconfig, ckpt_path, 
                         ignore_keys, scheduler_config, lr_g_factor, use_ema, **kwargs)
        
    @torch.no_grad()
    def encode(self, batch):
        return self._encode(batch, use_mode=False)