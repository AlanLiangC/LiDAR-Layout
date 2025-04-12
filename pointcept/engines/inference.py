import torch
from .test import TesterBase
from pointcept.utils.registry import Registry
from pointcept.datasets import build_dataset, point_collate_fn, collate_fn

INFERENCERS = Registry("inferencers")

@INFERENCERS.register_module()
class Inferencer(TesterBase):
    def __init__(self, cfg, model=None, test_loader=None, verbose=False):
        super().__init__(cfg, model, test_loader, verbose)
        self.model.eval()

    def inference_sample(self, index):
        data_sample = self.test_loader.dataset.__getitem__(index)
        data_dict = collate_fn([data_sample])
        for key, value in data_dict.items():
            if isinstance(value, torch.Tensor):
                data_dict[key] = value.to('cuda')
        pred_range = self.model(data_dict)
        return pred_range