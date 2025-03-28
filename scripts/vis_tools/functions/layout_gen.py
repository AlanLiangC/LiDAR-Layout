
import torch
from omegaconf import OmegaConf
from sample_layout import load_model
from ..utils.helpers import postprocess_sincos2arctan

class LidarScene:
    def __init__(self, ckpt_file_path, dataset):
        self.ckpt_file_path = ckpt_file_path
        logdir = '/'.join(ckpt_file_path.split('/')[:-1])
        config_file_path = f'{logdir}/config.yaml'
        self.configs = OmegaConf.load(config_file_path)
        self.dataset = dataset

    def build_model(self):
        self.configs.model.params.cond_stage_config.params.vocab = self.dataset.vocab
        self.model, _ = load_model(self.configs, self.ckpt_file_path)

    def inference_sample(self, data_dict):
        rec = self.model.log_images(data_dict, split='val', **self.configs)
        angles_pred = postprocess_sincos2arctan(rec[:,-2:])
        boxes_pred_den = self.dataset.re_scale_box(torch.concat([rec[:,:6], angles_pred], dim=-1))
        return boxes_pred_den