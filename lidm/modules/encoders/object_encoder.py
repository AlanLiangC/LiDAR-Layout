import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import numpy as np
from ...ops.chamferdist import knn_gpu

def build_lattice(H, W):
    N = H * W # number of grids
    # generate grid points within the range of (0, 1)
    margin = 1e-4
    h_p = np.linspace(0+margin, 1-margin, H, dtype=np.float32)
    w_p = np.linspace(0+margin, 1-margin, W, dtype=np.float32)
    grid_points = np.array(list(itertools.product(h_p, w_p))) # (N, 2)
    # generate grid indices
    h_i = np.linspace(0, H-1, H, dtype=np.int64)
    w_i = np.linspace(0, W-1, W, dtype=np.int64)
    grid_indices = np.array(list(itertools.product(h_i, w_i))) # (N, 2)
    return grid_points, grid_indices

def index_points(pc, idx):
    # pc: [B, N, C]
    # 1) idx: [B, S] -> pc_selected: [B, S, C]
    # 2) idx: [B, S, K] -> pc_selected: [B, S, K, C]
    device = pc.device
    B = pc.shape[0]
    view_shape = list(idx.shape) 
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B).to(device).view(view_shape).repeat(repeat_shape)
    pc_selected = pc[batch_indices, idx, :]
    return pc_selected

def knn_on_gpu(source_pts, query_pts, k):
    # source_pts: [B, N, C]
    # query_pts: [B, M, C]
    # knn_idx: [B, M, k] (sorted, from close to far)
    assert source_pts.device.type == 'cuda'
    assert query_pts.device.type == 'cuda'
    assert source_pts.size(0) == query_pts.size(0)
    assert source_pts.size(2) == query_pts.size(2)
    knn_idx = knn_gpu(p1=query_pts, p2=source_pts, K=k, return_nn=False, return_sorted=True)[1]
    return knn_idx

def knn_search(source_pts, query_pts, k):
    # source_pts: [B, N, C]
    # query_pts: [B, M, C]
    # knn_idx: [B, M, k] (sorted, from close to far)
    assert source_pts.device.type == query_pts.device.type
    device_type = source_pts.device.type
    assert device_type in ['cpu', 'cuda']
    knn_idx = knn_on_gpu(source_pts, query_pts, k)
    return knn_idx

class FC(nn.Module):
    def __init__(self, ic, oc, is_bn, nl, slope=None):
        super(FC, self).__init__()
        assert isinstance(is_bn, bool)
        assert nl in ['none', 'relu', 'leakyrelu', 'tanh', 'sigmoid']
        if nl == 'leakyrelu':
            assert slope is not None
        if slope is not None:
            assert nl == 'leakyrelu'
            assert slope>=0 and slope<=1
        self.is_bn = is_bn
        self.nl = nl
        self.linear = nn.Linear(ic, oc, bias=False)
        if self.is_bn:
            self.bn = nn.BatchNorm1d(oc)
        if nl == 'relu':
            self.activate = nn.ReLU(inplace=True)
        if nl == 'leakyrelu':
            self.activate = nn.LeakyReLU(negative_slope=slope, inplace=True)
        if nl == 'tanh':
            self.activate = nn.Tanh()
        if nl == 'sigmoid':
            self.activate = nn.Sigmoid()
    def forward(self, x):
        # x: [B, ic]
        # y: [B, oc]
        y = self.linear(x) # [B, oc]
        if self.is_bn:
            y = self.bn(y)
        if self.nl != 'none':
            y = self.activate(y)
        return y

class SMLP(nn.Module):
    def __init__(self, ic, oc, is_bn, nl, slope=None):
        super(SMLP, self).__init__()
        assert isinstance(is_bn, bool)
        assert nl in ['none', 'relu', 'leakyrelu', 'tanh', 'sigmoid']
        if nl == 'leakyrelu':
            assert slope is not None
        if slope is not None:
            assert nl == 'leakyrelu'
            assert slope>=0 and slope<=1
        self.is_bn = is_bn
        self.nl = nl
        self.conv = nn.Conv2d(in_channels=ic, out_channels=oc, kernel_size=1, bias=False)
        if self.is_bn:
            self.bn = nn.BatchNorm2d(oc)
        if nl == 'relu':
            self.activate = nn.ReLU(inplace=True)
        if nl == 'leakyrelu':
            self.activate = nn.LeakyReLU(negative_slope=slope, inplace=True)
        if nl == 'tanh':
            self.activate = nn.Tanh()
        if nl == 'sigmoid':
            self.activate = nn.Sigmoid()
    def forward(self, x):
        # x: [B, N, ic]
        # y: [B, N, oc]
        x = x.permute(0, 2, 1).contiguous().unsqueeze(-1) # [B, ic, N, 1]
        y = self.conv(x) # [B, oc, N, 1]
        if self.is_bn:
            y = self.bn(y)
        if self.nl != 'none':
            y = self.activate(y)   
        y = y.squeeze(-1).permute(0, 2, 1).contiguous() # [B, N, oc]
        return y

class ResSMLP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResSMLP, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.smlp_1 = SMLP(in_channels, in_channels, True, 'none')
        self.smlp_2 = SMLP(in_channels, out_channels, True, 'none')
        if in_channels != out_channels:
            self.shortcut = SMLP(in_channels, out_channels, True, 'none')
        self.nl = nn.ReLU(inplace=True)
    def forward(self, in_ftr):
        # in_ftr: [B, N, in_channels]
        out_ftr = self.smlp_2(self.nl(self.smlp_1(in_ftr)))
        if self.in_channels != self.out_channels:
            out_ftr = self.nl(self.shortcut(in_ftr) + out_ftr)
        else:
            out_ftr = self.nl(in_ftr + out_ftr)
        return out_ftr # [B, N, out_channels]

class NbrAgg(nn.Module):
    def __init__(self, num_neighbors, out_channels):
        super(NbrAgg, self).__init__()
        self.num_neighbors = num_neighbors
        self.out_channels = out_channels
        self.smlp_1 = nn.Sequential(SMLP(7, 16, True, 'relu'), SMLP(16, out_channels, True, 'relu'))
        self.smlp_2 = SMLP(3, out_channels, True, 'relu')
        self.smlp_3 = SMLP(out_channels*2, out_channels, True, 'relu')
    def forward(self, pts):
        # pts: [B, N, 3]
        assert pts.ndim == 3 
        assert pts.size(2) == 3
        B, N, K, C = pts.size(0), pts.size(1), self.num_neighbors, self.out_channels
        knn_idx = knn_search(pts, pts, K+1)
        knn_pts = index_points(pts, knn_idx)
        abs_pts = knn_pts[:, :, :1, :]
        rel_nbs = knn_pts[:, :, 1:, :] - knn_pts[:, :, :1, :]
        dists = torch.sqrt((rel_nbs ** 2).sum(dim=-1, keepdim=True) + 1e-8)
        concat = torch.cat((abs_pts.repeat(1, 1, K, 1), rel_nbs, dists), dim=-1)
        nbs_pooled = self.smlp_1(concat.view(B*N, K, -1)).view(B, N, K, -1).max(dim=2)[0]
        pts_lifted = self.smlp_2(pts)
        pts_ebd = self.smlp_3(torch.cat((pts_lifted, nbs_pooled), dim=-1))
        return pts_ebd

class CdwExtractor(nn.Module):
    def __init__(self):
        super(CdwExtractor, self).__init__()
        self.loc_agg = NbrAgg(16, 32)
        self.res_smlp_1 = ResSMLP(32, 64)
        self.res_smlp_2 = ResSMLP(128, 128)
        self.fuse = SMLP(352, 512, True, 'relu')
        self.att_pool = AttPool(512)
        self.fc = nn.Sequential(FC(1024, 512, True, 'relu'), FC(512, 1024, True, 'relu'), FC(1024, 1024, False, 'none'))
    def forward(self, pts):
        B, N, _ = pts.size()
        ftr_1 = self.loc_agg(pts)
        ftr_2 = self.res_smlp_1(ftr_1)
        ftr_3 = self.res_smlp_2(torch.cat((ftr_2, ftr_2.max(dim=1, keepdim=True)[0].repeat(1, N, 1)), dim=-1))
        ftr_4 = self.fuse(torch.cat((ftr_1, ftr_2, ftr_3, ftr_3.max(dim=1, keepdim=True)[0].repeat(1, N, 1)), dim=-1))
        cdw = self.fc(torch.cat((ftr_4.max(dim=1)[0], self.att_pool(ftr_4)), dim=-1))
        return cdw


class AttPool(nn.Module):
    def __init__(self, in_chs):
        super(AttPool, self).__init__()
        self.in_chs = in_chs
        self.linear_transform = SMLP(in_chs, in_chs, False, 'none')
    def forward(self, x):
        bs = x.size(0)
        num_pts = x.size(1)
        assert x.ndim==3 and x.size(2)==self.in_chs
        scores = F.softmax(self.linear_transform(x), dim=1)
        y = (x * scores).sum(dim=1)
        return y

class G2SD(nn.Module):
    def __init__(self, num_grids, **kwargs):
        super(G2SD, self).__init__()
        self.num_grids = num_grids
        self.grid_size = int(np.sqrt(num_grids))
        assert self.grid_size**2 == self.num_grids
        self.lattice = torch.tensor(build_lattice(self.grid_size, self.grid_size)[0])
        self.backbone = CdwExtractor()
        fold_1_1 = SMLP(1026, 256, True, 'relu')
        fold_1_2 = SMLP(256, 128, True, 'relu')
        fold_1_3 = SMLP(128, 64, True, 'relu')
        fold_1_4 = SMLP(64, 3, False, 'none')
        self.fold_1 = nn.Sequential(fold_1_1, fold_1_2, fold_1_3, fold_1_4)
        fold_2_1 = SMLP(1027, 256, True, 'relu')
        fold_2_2 = SMLP(256, 128, True, 'relu')
        fold_2_3 = SMLP(128, 64, True, 'relu')
        # fold_2_4 = 
        self.conv_out = SMLP(64, 3, False, 'none')
        self.fold_2 = nn.Sequential(fold_2_1, fold_2_2, fold_2_3, self.conv_out)

    def encode(self, pts):
            cdw = self.backbone(pts)
            return cdw
    
    def decode(self, cdw):
            B, device = cdw.size(0), cdw.device
            grids = (self.lattice).unsqueeze(0).repeat(B, 1, 1).to(device)
            cdw_dup = cdw.unsqueeze(1).repeat(1, self.num_grids, 1)
            concat_1 = torch.cat((cdw_dup, grids), dim=-1)
            rec_1 = self.fold_1(concat_1)
            concat_2 = torch.cat((cdw_dup, rec_1), dim=-1)
            rec_2 = self.fold_2(concat_2)
            return rec_2

    def forward(self, pts):
        B, N, device = pts.size(0), pts.size(1), pts.device
        grids = (self.lattice).unsqueeze(0).repeat(B, 1, 1).to(device)
        cdw = self.backbone(pts)
        cdw_dup = cdw.unsqueeze(1).repeat(1, self.num_grids, 1)
        concat_1 = torch.cat((cdw_dup, grids), dim=-1)
        rec_1 = self.fold_1(concat_1)
        concat_2 = torch.cat((cdw_dup, rec_1), dim=-1)
        rec_2 = self.fold_2(concat_2)
        return rec_2

if __name__ == "__main__":
    model = G2SD(64).to('cuda')
    points = torch.randn(4,1024,3).to('cuda')
    rel = model(points)

    print(rel.shape)