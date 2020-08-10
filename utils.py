'''
Copied from https://github.com/wentaoyuan/pcn
'''

import torch
import torch.nn as nn
from utils.pointnet_util import *
def chamfer_distance(p1, p2):

    '''
    Calculate Chamfer Distance between two point sets
    :param p1: size[B, N, D]
    :param p2: size[B, M, D]
    :param debug: whether need to output debug info
    :return: sum of all batches of Chamfer Distance of two point sets
    '''

    # assert p1.size(0) == p2.size(0) and p1.size(2) == p2.size(2)


    p1 = p1.unsqueeze(1)
    p2 = p2.unsqueeze(1)

    p1 = p1.repeat(1, p2.size(2), 1, 1)

    p1 = p1.transpose(1, 2)

    p2 = p2.repeat(1, p1.size(1), 1, 1)

    dist = p1 - p2
    #dist = torch.add(p1, torch.neg(p2))

    dist = torch.norm(dist, 2, dim=3)

    dist = torch.min(dist, dim=2)[0]

    dist = torch.sum(dist)

    return dist


class pointNet(nn.Module):
    def __init__(self, global_feature_size, **kwargs):
        super(pointNet, self).__init__(**kwargs)
        self.global_feature_size = global_feature_size
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, (1, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, (1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, (1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Linear(256, global_feature_size),
            nn.BatchNorm1d(global_feature_size),
            nn.ReLU()
        )

    def forward(self, x, **kwargs):
        if len(x.shape) == 3:
            x = torch.unsqueeze(x, 1)

        # Bx1xNx3
        out = self.layer1(x)

        # Bx1xNx256
        out, _ = torch.max(out, 2, keepdim=True)

        # Bx1x1x256
        out = torch.squeeze(out)
        out = self.layer2(out)
        return out


def feature_expansion(points, npoint, nsample, radius=0.1):
    """
    Args:
        points: point cloud of shape (B, N, 3)
        npoint: integer, number of centroids
        nsample: integer, number of points to sample for each centroids
        radius: ball query radius

    Returns:
        grouped_xyz: tensor of shape (B * npoint * nsample * 10)
                     (xi, yi, zi, xj, yj, zj, xi-xj, yi-yj, zi-zj, dis)
                     where xj, yj, zj is the query center of xyzi
    """
    idx = farthest_point_sample(points, npoint)  # BxS
    new_xyz = index_points(points, idx)  # (B * npoint * 3)  centroids
    idx_grouped = query_ball_point(radius, nsample, points, new_xyz)  # (B * npoint * nsample)
    grouped_xyz = index_points(points, idx_grouped)  # (B * npoint * nsample * 3)

    npoint_xyz = torch.unsqueeze(new_xyz, 2)  # (B * npoint * 1 * 3)
    npoint_xyz = npoint_xyz.repeat(1, 1, nsample, 1)  # (B * npoint * nsample * 3)

    abs_xyz = grouped_xyz - npoint_xyz

    dis = torch.sum(abs_xyz ** 2, -1, keepdim=True)

    grouped_xyz = torch.cat([grouped_xyz, npoint_xyz, abs_xyz, dis], dim=-1)

    return grouped_xyz

class SpectralNorm:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        size = weight.size()
        weight_mat = weight.contiguous().view(size[0], -1)
        with torch.no_grad():
            v = weight_mat.t() @ u
            v = v / v.norm()
            u = weight_mat @ v
            u = u / u.norm()
        sigma = u @ weight_mat @ v
        weight_sn = weight / sigma
        # weight_sn = weight_sn.view(*size)

        return weight_sn, u

    @staticmethod
    def apply(module, name):
        fn = SpectralNorm(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', weight)
        input_size = weight.size(0)
        u = weight.new_empty(input_size).normal_()
        module.register_buffer(name, weight)
        module.register_buffer(name + '_u', u)

        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight_sn, u = self.compute_weight(module)
        setattr(module, self.name, weight_sn)
        setattr(module, self.name + '_u', u)


def spectral_norm(module, name='weight'):
    SpectralNorm.apply(module, name)

    return module
def spectral_init(module, gain=1):
    nn.init.kaiming_uniform_(module.weight, gain)
    if module.bias is not None:
        module.bias.data.zero_()

    return spectral_norm(module)

class SelfAttention(nn.Module):
    def __init__(self, in_channel, gain=1):
        super().__init__()

        self.query = spectral_init(nn.Conv1d(in_channel, in_channel // 8, 1),
                                   gain=gain)
        self.key = spectral_init(nn.Conv1d(in_channel, in_channel // 8, 1),
                                 gain=gain)
        self.value = spectral_init(nn.Conv1d(in_channel, in_channel, 1),
                                   gain=gain)

        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, input):
        shape = input.shape
        flatten = input.view(shape[0], shape[1], -1)
        query = self.query(flatten).permute(0, 2, 1)
        key = self.key(flatten)
        value = self.value(flatten)
        query_key = torch.bmm(query, key)
        attn = F.softmax(query_key, 1)
        attn = torch.bmm(value, attn)
        attn = attn.view(*shape)
        out = self.gamma * attn + input

        return out


class modelCompletion(nn.Module):
    def __init__(self, global_feature_size, npoint, nsample, radius=0.1, **kwargs):
        super(modelCompletion, self).__init__(**kwargs)
        self.global_feature_size = global_feature_size
        self.npoint = npoint
        self.nsample = nsample
        self.radius = radius

        self.pointnet = pointNet(global_feature_size)

        self.mlp1 = nn.Sequential(
            nn.Conv2d(10, 64, [1, 1]),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, [1, 1]),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, [1, 1]),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d([1, nsample])
        )

        self.attention = SelfAttention(256)
        self.mlp2 = nn.Sequential(
            nn.Conv1d(npoint, 1024, 1),
            nn.ReLU(),
            nn.Conv1d(1024, npoint, 1)
        )

        # decoder
        self.mlp3 = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 2048 * 3)
        )

    def forward(self, pl):
        """
        Args:
            pl: point cloud of shape (B, N, 3)
        """

        global_feature = self.pointnet(pl)  # (B, feature_size)
        global_feature = torch.unsqueeze(global_feature, 1)  # (B, 1, feature_size)
        global_feature = global_feature.repeat(1, self.npoint, 1)

        local_feature = feature_expansion(pl, self.npoint, self.nsample,
                                          radius=self.radius)  # B x npoint x nsample x 10
        local_feature = local_feature.permute(0, 3, 1, 2)
        local_feature = self.mlp1(local_feature)  # (B, npoint, 1, 256)
        local_feature = torch.squeeze(local_feature)  # (B, npoint, 256)

        mix = torch.cat([local_feature, global_feature], dim=-1)  # (B, npoint, 512)

        mix = self.attention(mix)
        mix = torch.cat([mix, global_feature], dim=-1)  # (B, npoint, 768)
        code = self.mlp2(mix)
        code, _ = torch.max(code, 1)  # (B, 768)

        out = self.mlp3(code)
        out = out.view(pl.size(0), -1, 3)
        return out


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )

        self.mlp2 = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 1024, 1)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Bx3xN

        feature = self.mlp1(x)  # Bx256xN

        global_feature, _ = torch.max(feature, -1, True)  # Bx256x1

        feature = torch.cat([feature, global_feature.repeat(1, 1, feature.size(2))], 1)  # Bx512xN

        feature = self.mlp2(feature)  # Bx1024xN

        global_feature, _ = torch.max(feature, -1)  # Bx1024
        del _
        return global_feature


class DecoderCD(nn.Module):
    def __init__(self, grid_scale=0.05, grid_size=2, num_coarse=512):
        super(DecoderCD, self).__init__()
        self.grid_size = grid_size
        self.grid_scale = grid_scale
        self.num_coarse = num_coarse
        self.num_fine = grid_size ** 2 * num_coarse
        # output units = grid_size**2 * num_coarse
        # num_coarse is the number of patches in fine grained completion cloud point

        x = torch.linspace(-grid_scale, grid_scale, grid_size)
        y = torch.linspace(-grid_scale, grid_scale, grid_size)
        grid = torch.meshgrid(x, y)
        grid = torch.reshape(torch.stack(grid, 2), [-1, 2]).t().contiguous()  # 2, grid_size**2
        grid.unsqueeze_(0)  # (1, 2, grid_size**2)
        self.grid = grid.repeat(1, 1, num_coarse)

        self.mlp1 = nn.Sequential(
            nn.Linear(1024, 1024),  # global feature size 1024
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * self.num_coarse)
        )

        self.mlp2 = nn.Sequential(
            nn.Conv1d(1024 + 3 + 2, 512, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 3, 1)
        )

    def forward(self, feature):
        """
        Args:
            feature: Bx1024
        """
        coarse = self.mlp1(feature)
        coarse = torch.reshape(coarse, (-1, 3, self.num_coarse))  # B x 3 x num_coarse
        grid_feature = self.grid.repeat(feature.size(0), 1, 1)
        # center
        center = torch.unsqueeze(coarse, 3).repeat(1, 1, 1, self.grid_size ** 2)  # (B, 3, num_coarse, grid_size**2)
        center = torch.reshape(center, (-1, 3, self.num_fine))

        global_feature = torch.unsqueeze(feature, 2).repeat(1, 1, self.num_fine)

        feat = torch.cat([grid_feature, center, global_feature], 1)

        fine = self.mlp2(feat) + center

        return coarse.permute(0, 2, 1), fine.permute(0, 2, 1)


class PCN(nn.Module):
    def __init__(self):
        super(PCN, self).__init__()
        self.encoder = Encoder()
        self.decoder = DecoderCD()

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def to(self, device, **kwargs):
        self.decoder.grid = self.decoder.grid.to(device)
        super(PCN, self).to(device, **kwargs)

