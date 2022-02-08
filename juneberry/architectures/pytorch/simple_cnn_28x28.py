#! /usr/bin/env python3

# ======================================================================================================================
# Juneberry - General Release
#
# Copyright 2021 Carnegie Mellon University.
#
# NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS"
# BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER
# INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED
# FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM
# FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.
#
# Released under a BSD (SEI)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.
#
# [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see
# Copyright notice for non-US Government use and distribution.
#
# This Software includes and/or makes use of Third-Party Software subject to its own license.
#
# DM21-0884
#
# ======================================================================================================================

import logging
import sys
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class mnist_CNN(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.CNNlayers = nn.Sequential(
                nn.Conv2d(in_channels=channels, out_channels=8, kernel_size=3, padding=1, stride=1),
                nn.MaxPool2d(kernel_size=2), 
                nn.Conv2d(8, 16, 2, 1, 1),
                nn.ReLU(),
                nn.MaxPool2d(4) 
        )

        self.linearLayers = nn.Sequential(
                nn.Linear(in_features=16*3*3, out_features=64),   
                nn.ReLU(),
                nn.Linear(64, 10) 
        )

    def forward(self, x):
        x = self.CNNlayers(x)
        x = torch.flatten(x, 1)
        x = self.linearLayers(x)
        return x


class CNN28x28:
    def __call__(self, img_width, img_height, channels, num_classes):
        if img_width != 28 or img_height != 28:
            logger.error("The model only works with 28x28 images.")
            sys.exit(-1)
        model = mnist_CNN(channels)
        return model
