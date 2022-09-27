#! /usr/bin/env python3

# ======================================================================================================================
#  Copyright 2021 Carnegie Mellon University.
#
#  NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS"
#  BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER
#  INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED
#  FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM
#  FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.
#
#  Released under a BSD (SEI)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.
#
#  [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.
#  Please see Copyright notice for non-US Government use and distribution.
#
#  This Software includes and/or makes use of the following Third-Party Software subject to its own license:
#
#  1. PyTorch (https://github.com/pytorch/pytorch/blob/master/LICENSE) Copyright 2016 facebook, inc..
#  2. NumPY (https://github.com/numpy/numpy/blob/master/LICENSE.txt) Copyright 2020 Numpy developers.
#  3. Matplotlib (https://matplotlib.org/3.1.1/users/license.html) Copyright 2013 Matplotlib Development Team.
#  4. pillow (https://github.com/python-pillow/Pillow/blob/master/LICENSE) Copyright 2020 Alex Clark and contributors.
#  5. SKlearn (https://github.com/scikit-learn/sklearn-docbuilder/blob/master/LICENSE) Copyright 2013 scikit-learn
#      developers.
#  6. torchsummary (https://github.com/TylerYep/torch-summary/blob/master/LICENSE) Copyright 2020 Tyler Yep.
#  7. pytest (https://docs.pytest.org/en/stable/license.html) Copyright 2020 Holger Krekel and others.
#  8. pylint (https://github.com/PyCQA/pylint/blob/main/LICENSE) Copyright 1991 Free Software Foundation, Inc..
#  9. Python (https://docs.python.org/3/license.html#psf-license) Copyright 2001 python software foundation.
#  10. doit (https://github.com/pydoit/doit/blob/master/LICENSE) Copyright 2014 Eduardo Naufel Schettino.
#  11. tensorboard (https://github.com/tensorflow/tensorboard/blob/master/LICENSE) Copyright 2017 The TensorFlow
#                  Authors.
#  12. pandas (https://github.com/pandas-dev/pandas/blob/master/LICENSE) Copyright 2011 AQR Capital Management, LLC,
#             Lambda Foundry, Inc. and PyData Development Team.
#  13. pycocotools (https://github.com/cocodataset/cocoapi/blob/master/license.txt) Copyright 2014 Piotr Dollar and
#                  Tsung-Yi Lin.
#  14. brambox (https://gitlab.com/EAVISE/brambox/-/blob/master/LICENSE) Copyright 2017 EAVISE.
#  15. pyyaml  (https://github.com/yaml/pyyaml/blob/master/LICENSE) Copyright 2017 Ingy döt Net ; Kirill Simonov.
#  16. natsort (https://github.com/SethMMorton/natsort/blob/master/LICENSE) Copyright 2020 Seth M. Morton.
#  17. prodict  (https://github.com/ramazanpolat/prodict/blob/master/LICENSE.txt) Copyright 2018 Ramazan Polat
#               (ramazanpolat@gmail.com).
#  18. jsonschema (https://github.com/Julian/jsonschema/blob/main/COPYING) Copyright 2013 Julian Berman.
#
#  DM21-0689
#
# ======================================================================================================================

import importlib
import logging
import sys

import torch.nn as nn
from torchvision.models.resnet import BasicBlock as BB
from torchvision.models.resnet import Bottleneck as BN
from torchvision.models.resnet import ResNet

import torch.nn.functional as F

logger = logging.getLogger(__name__)


class Wrapper:
    """
    Basic wrapper for torchvision models classes
    """

    def __call__(self, className, classArgs, num_classes):
        mod = importlib.import_module('torchvision.models')
        my_class = getattr(mod, className)
        return my_class(**classArgs)


class ResNetCustom(nn.Module):

    def __init__(self, block, n, num_classes=10, zero_init_residual=False, groups=1,
                 width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
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
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=False)
        self.layer1 = ResNet._make_layer(self, block, 16, 2 * n)
        self.layer2 = ResNet._make_layer(self, block, 32, 2 * n, stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = ResNet._make_layer(self, block, 64, 2 * n, stride=2, dilate=replace_stride_with_dilation[1])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BN):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BB):
                    nn.init.constant_(m.bn2.weight, 0)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)


class Resnet32x32:
    def __call__(self, img_width, img_height, channels, num_classes, layers):
        if img_width != 32 or img_height != 32 or channels != 3:
            logging.error("The model only works with 32x32 RGB images.")
            sys.exit(-1)
        elif (layers - 2) % 6 != 0:
            logging.error("Layers argument missing or incorrect. (Layers - 2) % 6 must be zero for ResNet6n2.")
            sys.exit(-1)
        else:
            model = ResNetCustom(block=BB, n=int((layers - 2) / 6), num_classes=num_classes)
            return model


class PreActBB(BB):
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super().__init__(inplanes, planes, stride, downsample, groups,
                         base_width, dilation, norm_layer)
        self.bn1 = norm_layer(inplanes)

    def forward(self, x):
        shortcut = x
        out = F.relu(self.bn1(x), inplace=False)
        if self.downsample is not None:
            shortcut = self.downsample(x)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out), inplace=False))
        out += shortcut
        return out


class PreactResNetCustom(ResNetCustom):
    def __init__(self, block, n, num_classes=10, zero_init_residual=False, groups=1,
                 width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
        super(PreactResNetCustom, self).__init__(block, n, num_classes, zero_init_residual, groups,
                                                 width_per_group, replace_stride_with_dilation, norm_layer)
        self.bn1 = nn.BatchNorm2d(64 * block.expansion)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = self.avgpool(out)
        out = out.flatten(1)
        out = self.fc(out)
        return out


class PreActResnet32x32:
    def __call__(self, img_width, img_height, channels, num_classes, layers):
        if img_width != 32 or img_height != 32 or channels != 3:
            logging.error("The model only works with 32x32 RGB images.")
            sys.exit(-1)
        elif (layers - 2) % 6 != 0:
            logging.error("Layers argument missing or incorrect. (Layers - 2) % 6 must be zero for ResNet6n2.")
            sys.exit(-1)
        else:
            model = PreactResNetCustom(block=PreActBB, n=int((layers - 2) / 6), num_classes=num_classes)
            return model
