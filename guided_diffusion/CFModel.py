import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models.resnet as resnet_utils

from torch.hub import load_state_dict_from_url

from .nn import timestep_embedding

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


# =======================================================
# Resnet

def conv3x3(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=3, padding=1)


class ResNetFeatures(resnet_utils.ResNet):
    def forward(self, x0):

        x0 = self.conv1(x0)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x0 = self.maxpool(x0)

        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return x0, x1, x2, x3, x4


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNetFeatures(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', resnet_utils.BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs), resnet_utils.BasicBlock.expansion


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', resnet_utils.BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs), resnet_utils.BasicBlock.expansion


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', resnet_utils.Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs), resnet_utils.BasicBlock.expansion


# =======================================================
# Decoder


class Transition(nn.Module):
    """docstring for Transition"""
    def __init__(self, embedding_size):
        super().__init__()
        self.embedding_size = embedding_size
        self.time_embed = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.SELU(),
            nn.Linear(embedding_size, 2 * embedding_size)
        )

    # Extracted from https://github.com/naoto0804/pytorch-AdaIN/blob/master/function.py
    @staticmethod
    def calc_mean_std(feat, eps=1e-5):

        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    def forward(self, x, timesteps):

        shape = x.shape
        mean, std = self.calc_mean_std(x)
        t_embs = self.time_embed(timestep_embedding(timesteps, self.embedding_size))
        t_embs = t_embs.view(x.size(0), -1, 1, 1)
        tmean, t_logstd = t_embs[:, :self.embedding_size], t_embs[:, self.embedding_size:]
        x = ((x - mean) / std) * t_logstd.exp() + tmean
        return x


class DeconvolutionBlock(nn.Module):
    # set of upsample + 1 conv + conv
    def __init__(self, in_channels, out_channels=None, residual=True):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.conv1 = conv3x3(in_channels, out_channels)
        self.conv2 = conv3x3(out_channels, out_channels) if residual else None
        self.conv3 = conv3x3(out_channels, out_channels)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x_deep, x_shallow=None):
        x_deep = F.interpolate(x_deep, scale_factor=2)
        x_deep = self.conv1(x_deep)
        if self.conv2 is not None:
            x_deep = x_deep + self.conv2(x_shallow)
        x_shallow = self.bn1(x_deep)
        x_shallow = F.relu(x_shallow, inplace=True)
        x_shallow = self.conv3(x_shallow)
        x_shallow = self.bn2(x_shallow)
        return F.relu(x_shallow, inplace=True)


class Interpolate(nn.Module):
    def forward(self, x):
        return F.interpolate(x, scale_factor=2)
        

class Decoder(nn.Module):

    def __init__(self, expansion=1):
        super().__init__()
        self.expansion = expansion
        self.t4 = Transition(512 * expansion)
        self.layer4 = self._make_layer(512, 256, r=False)
        self.t3 = Transition(256 * expansion)
        self.layer3 = self._make_layer(256, 128)
        # self.t2 = Transition(128 * expansion)
        self.layer2 = self._make_layer(128, 64)
        # self.t1 = Transition(64 * expansion)
        self.layer1 = self._make_layer(64, 64)
        # self.t0 = Transition(64)
        self.layer0 = self._make_layer(64, 64, expand=False)

        self.pred = nn.Sequential(
            # Interpolate(),
            conv3x3(64, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=1)
        )

    def _make_layer(self, in_channels, out_channels, r=False, expand=True):
        in_channels = self.expansion * in_channels
        out_channels = (self.expansion if expand else 1) * out_channels
        return DeconvolutionBlock(in_channels, out_channels, r)

    def forward(self, x0, x1, x2, x3, x4, timesteps):

        x4 = self.t4(x4, timesteps)
        x4 = self.layer4(x4)  # 7x7
        x3 = self.t3(x3, timesteps)
        x3 = self.layer3(x4, x3)  # 14x14
        x2 = self.layer2(x3, x2)  # 28x28
        x1 = self.layer1(x2, x1)  # 56x56
        x0 = self.layer0(x1, x0)  # 112x112
        x0 = self.pred(x0)  # 224x224
        return x0


# =======================================================
# Network


class EncoderDecoder(nn.Module):

    def __init__(self, resnet):
        super().__init__()
        if '18' in resnet:
            resnet = resnet18
        elif '34' in resnet:
            resnet = resnet34
        elif '50' in resnet:
            resnet = resnet50
        self.backbone, expansion = resnet(pretrained=True)
        del self.backbone.fc
        del self.backbone.avgpool

        self.decoder = Decoder(expansion)
        mean = torch.tensor([0.485 * 2 - 1, 0.456 * 2 - 1, 0.406 * 2 - 1])
        std = torch.tensor([0.229 * 2, 0.224 * 2, 0.225 * 2])
        self.register_buffer('mu', mean.view(1, -1, 1, 1))
        self.register_buffer('std', std.view(1, -1, 1, 1))

    def forward(self, x, timesteps):
        x = (x - self.mu) / self.std

        x0, x1, x2, x3, x4 = self.backbone(x)
        return self.decoder(x0, x1, x2, x3, x4, timesteps)


def get_translator(resnet):
    return EncoderDecoder(resnet)


def get_model(resnet):
    return getattr(resnet_utils, resnet)(pretrained=True)
