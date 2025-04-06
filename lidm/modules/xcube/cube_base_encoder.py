import torch
import torch.nn as nn
import fvdb
from fvdb import GridBatch

from .utils.embedder_util import get_embedder

class Encoder(nn.Module):
    def __init__(self, c_dim, use_input_intensity):
        super().__init__()
        self.use_input_intensity = use_input_intensity

        embed_fn, input_ch = get_embedder(5)
        self.pos_embedder = embed_fn
        
        input_dim = 0
        input_dim += input_ch

        if self.use_input_intensity:
            input_dim += 1

        self.mix_fc = nn.Linear(input_dim, c_dim)

    def forward(self, grid: GridBatch, batch) -> torch.Tensor:
     
        coords = grid.grid_to_world(grid.ijk.float()).jdata
        unet_feat = self.pos_embedder(coords)
        
        if self.use_input_intensity:
            input_intensity = fvdb.JaggedTensor(batch['intensity'])
            unet_feat = torch.cat([unet_feat, input_intensity.jdata], dim=1)

        unet_feat = self.mix_fc(unet_feat)
        return unet_feat