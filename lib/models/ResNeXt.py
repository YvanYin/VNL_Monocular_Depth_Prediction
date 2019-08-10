from collections import OrderedDict
import torch.nn as nn
from lib.core.config import cfg

# ---------------------------------------------------------------------------- #
# Bits for specific architectures (ResNeXt50, ResNeXt101, ...)
# ---------------------------------------------------------------------------- #

def ResNeXt50_32x4d_body_stride16():
    return ResNeXt_body((3, 4, 6, 3), 32, 4, 16)


def ResNeXt101_32x4d_body_stride16():
    return ResNeXt_body((3, 4, 23, 3), 32, 4, 16)


class ResNeXt_body(nn.Module):
    def __init__(self, block_counts, cardinality, base_width, output_stride):
        super().__init__()
        self.block_counts = block_counts
        self.convX = len(block_counts) + 1
        self.num_layers = (sum(block_counts) + 3 * (self.convX == 4)) * 3 + 2

        self.res1 = basic_bn_stem()
        dim_in = 64
        res5_dilate = int(32 / output_stride)
        res5_stride = 2 if res5_dilate == 1 else 1
        res4_dilate = 1 if res5_dilate <= 2 else 2
        res4_stride = 2 if res4_dilate == 1 else 1

        self.res2, dim_in = add_stage(dim_in, 256, block_counts[0], cardinality, base_width,
                                      dilation=1, stride_init=1)
        self.res3, dim_in = add_stage(dim_in, 512, block_counts[1], cardinality, base_width,
                                      dilation=1, stride_init=2)
        self.res4, dim_in = add_stage(dim_in, 1024, block_counts[2], cardinality, base_width,
                                      dilation=res4_dilate, stride_init=res4_stride)
        self.res5, dim_in = add_stage(dim_in, 2048, block_counts[3], cardinality, base_width,
                                      dilation=res5_dilate, stride_init=res5_stride)
        self.spatial_scale = 1 / output_stride
        self.dim_out = dim_in
        self._init_modle()

    def forward(self, x):
        for i in range(self.convX):
            x = getattr(self, 'res%d' % (i + 1))(x)
        return x


    def train(self, mode=True):
        # Override
        self.training = mode

        for i in range(1, self.convX + 1):
            getattr(self, 'res%d' % i).train(mode)
    def _init_modle(self):
        def freeze_params(m):
            for p in m.parameters():
                p.requires_grad = False
        if cfg.MODEL.FREEZE_BACKBONE_BN:
            self.apply(lambda m: freeze_params(m) if isinstance(m, nn.BatchNorm2d) else None)

def basic_bn_stem():
    conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
    return nn.Sequential(OrderedDict([
        ('conv1', conv1),
        ('bn1', nn.BatchNorm2d(64)),
        ('relu', nn.ReLU(inplace=True)),
        ('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))]))

def add_stage(inplanes, outplanes, nblocks, cardinality, base_width, dilation=1, stride_init=2):
    """Make a stage consist of `nblocks` residual blocks.
    Returns:
        - stage module: an nn.Sequentail module of residual blocks
        - final output dimension
    """
    res_blocks = []
    stride = stride_init
    for _ in range(nblocks):
        res_blocks.append(ResNeXtBottleneck(
            inplanes, outplanes, stride, dilation, cardinality, base_width
        ))
        inplanes = outplanes
        stride = 1
    return nn.Sequential(*res_blocks), outplanes


class ResNeXtBottleneck(nn.Module):
    """
    RexNeXt bottleneck type C (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)
    """

    def __init__(self, in_channels, out_channels, stride, dilate, cardinality=32, base_width=4):
        """ Constructor
        Args:
            in_channels: input channel dimensionality
            out_channels: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            cardinality: num of convolution groups.
            base_width: base number of channels in each group.
            widen_factor: factor to reduce the input dimensionality before convolution.
        """
        super(ResNeXtBottleneck, self).__init__()
        width_ratio = out_channels / 256.
        D = cardinality * base_width * int(width_ratio)
        self.conv1 = nn.Conv2d(in_channels, D, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(D)
        self.conv2 = nn.Conv2d(D, D, kernel_size=3, stride=stride, padding=dilate, dilation=dilate, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(D)
        self.conv3 = nn.Conv2d(D, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential()
            self.shortcut.add_module('conv',
                                     nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0,
                                               bias=False))
            self.shortcut.add_module('bn', nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.shortcut is not None:
            residual = self.shortcut(x)

        out += residual
        out = self.relu(out)
        return out



