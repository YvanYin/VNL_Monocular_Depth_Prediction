import torch.nn as nn
import math
from lib.core.config import cfg
# ---------------------------------------------------------------------------- #
# Bits for specific architectures (ResNeXt50, ResNeXt101, ...)
# ---------------------------------------------------------------------------- #
def MobileNetV2_body():
    return MobileNetV2()

def MobileNetV2_body_stride16():
    return MobileNetV2(output_stride=16)

def MobileNetV2_body_stride8():
    return MobileNetV2(output_stride=8)


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, dilation=1):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, groups=hidden_dim, bias=False, padding=dilation, dilation=dilation),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, groups=hidden_dim, bias=False, padding=dilation, dilation=dilation),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            out = self.conv(x)
            out += x
            return out
        else:
            return self.conv(x)

def add_block(res_setting, input_channel, width_mult=1, dilation=1):
    # building inverted residual blocks
    block = []
    for t, c, n, s in res_setting:
        output_channel = int(c * width_mult)
        for i in range(n):
            if i == 0:
                block.append(InvertedResidual(input_channel, output_channel, s, expand_ratio=t, dilation=dilation))
            else:
                block.append(InvertedResidual(input_channel, output_channel, 1, expand_ratio=t, dilation=dilation))
            input_channel = output_channel
    return nn.Sequential(*block), output_channel


class MobileNetV2(nn.Module):
    def __init__(self, width_mult=1., output_stride=32):
        super(MobileNetV2, self).__init__()
        input_channel = 32
        last_channel = 320
        self.convX = 5
        stride1 = 1 if 32 / output_stride == 4 else 2
        stride2 = 1 if 32 / output_stride > 1 else 2
        dilation1 = 1 if stride1 == 2 else 2
        dilation2 = 1 if stride2 == 2 else (2 if stride1 == 2 else 4)

        interverted_residual_setting_block2 = [
             #t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
        ]
        interverted_residual_setting_block3 = [
            # t, c, n, s
            [6, 32, 3, 2],
        ]
        interverted_residual_setting_block4 = [
            # t, c, n, s
            [6, 64, 4, stride1],
            [6, 96, 3, 1],
        ]
        interverted_residual_setting_block5 = [
            # t, c, n, s
            [6, 160, 3, stride2],
            [6, 320, 1, 1],
        ]


        # building first layer
        #assert cfg.CROP_SIZE[0] % 32 == 0 and cfg.CROP_SIZE[1] % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = last_channel
        self.res1 = nn.Sequential(conv_bn(3, input_channel, 2))

        self.res2, output_channel = add_block(interverted_residual_setting_block2, input_channel, width_mult)

        self.res3, output_channel = add_block(interverted_residual_setting_block3, output_channel, width_mult)

        self.res4, output_channel = add_block(interverted_residual_setting_block4, output_channel, width_mult, dilation1)

        self.res5, output_channel = add_block(interverted_residual_setting_block5, output_channel, width_mult, dilation2)

        self._initialize_weights()

    def forward(self, x):
        for i in range(self.convX):
            x = getattr(self, 'res%d' % (i + 1))(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()