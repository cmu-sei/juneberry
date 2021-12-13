#! /usr/bin/env python3

import torchvision

# From a download point of view there is only one file so we don't need to also do "train=False"
train = torchvision.datasets.CIFAR10(root="dataroot", train=True, download=True)

