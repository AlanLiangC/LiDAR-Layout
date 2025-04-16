import torch
import torch.nn as nn
import fvdb
from fvdb import GridBatch

from .utils.embedder_util import get_embedder
from pointcept.models.builder import build_model
from pointcept.models.utils.structure import Point

class Encoder(nn.Module):
    def __init__(self, c_dim, use_input_intensity, backbone):
        super().__init__()
        self.use_input_intensity = use_input_intensity
        self.backbone = build_model(backbone)

    def forward(self, grid: GridBatch, batch) -> torch.Tensor:
        ret = {}
        coords = grid.grid_to_world(grid.ijk.float()).jdata
        grid_coord = grid.ijk.jdata.long()
        grid_coord -= grid_coord.min(dim=0)[0][None,:]
        ret.update({
            'coord': coords,
            'grid_coord': grid_coord,
            'feat': coords,
            'offset': batch['offset'],
        })
        point = Point(ret)
        point = self.backbone(point)
        unet_feat = point.feat

        return unet_feat