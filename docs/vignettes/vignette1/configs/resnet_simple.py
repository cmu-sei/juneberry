import sys
import logging
import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock as BB
from torchvision.models.resnet import Bottleneck as BN
from typing import Type, Union

logger = logging.getLogger(__name__)


class PoolPadSkip(nn.Module):

    # Implements "Option A" from He et al. (2015):
    #
    #    > When the dimensions increase (dotted line shortcuts in Fig. 3), we consider two options:
    #    > (A) The shortcut still performs identity mapping, with extra zero entries padded for
    #    > increasing dimensions. This option introduces no extra parameter; (B) The projection
    #    > shortcut in Eqn.(2) is used to match dimensions (done by 1×1 convolutions). For both
    #    > options, when the shortcuts go across feature maps of two sizes, they are performed
    #    > with a stride of 2.

    def __init__(self, stride):
        super().__init__()
        self.avgpool = nn.AvgPool2d(1, stride)

    def forward(self, x):
        out = self.avgpool(x)
        device = out.get_device() if torch.cuda.is_available() else None
        pad = torch.zeros(out.shape[0], out.shape[1], out.shape[2], out.shape[3], device=device)
        return torch.cat((out, pad), 1)


class ResNetCustom(nn.Module):

    def __init__(self, block, n, num_classes=10, groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNetCustom, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 16
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None or a 3-element tuple, got {}".format(
                replace_stride_with_dilation))

        self.groups = groups
        self.base_width = width_per_group

        # The first layer is 3 x 3 convolutions.
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=False)

        # Then we use a stack of 6n layers with 3 x 3 convolutions on the feature maps of sizes
        # {32, 16, 8} respectively, with 2n layers for each feature map size. The numbers of filters are
        # {16, 32, 64} respectively. The subsampling is performed by convolutions with a stride of 2.
        self.layer1 = self._make_layer(block, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 64, n, stride=2, dilate=replace_stride_with_dilation[1])

        # The network ends with a global average pooling, a 10-way fully-connected layer, and softmax.
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Note: torchvision.ResNet uses 'fan_out'; He et al. (2015) uses the fan_in equivalent
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block: Type[Union[BB, BN]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = PoolPadSkip(stride)

        layers = [block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation,
                        norm_layer)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


class Resnet32x32:
    def __call__(self, img_width, img_height, channels, num_classes, layers):
        if img_width != 32 or img_height != 32 or channels != 3:
            logger.error("The model only works with 32x32 RGB images.")
            sys.exit(-1)
        elif (layers - 2) % 6 != 0:
            logger.error("Layers argument missing or incorrect. (Layers - 2) % 6 must be zero for ResNet6n2.")
            sys.exit(-1)
        else:
            model = ResNetCustom(block=BB, n=int((layers - 2) / 6), num_classes=num_classes)
            return model
