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

import logging
import sys

import torch.nn as nn

logger = logging.getLogger(__name__)


class BinaryNet(nn.Module):
    def __init__(self, num_features=2):
        super(BinaryNet, self).__init__()
        self.layer_1 = nn.Linear(num_features, 16)
        self.layer_out = nn.Linear(16, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.relu(self.layer_1(input))
        x = self.sigmoid(self.layer_out(x))

        return x


class BinaryModel:
    def __call__(self, num_classes):
        if num_classes != 2:
            logger.error("This model only works with binary tabular datasets.")
            sys.exit(-1)

        return BinaryNet()


class MultiClassNet(nn.Module):
    def __init__(self, num_features=2):
        super(MultiClassNet, self).__init__()
        self.layer_1 = nn.Linear(num_features, 4)
        self.layer_out = nn.Linear(4, 3)

        self.relu = nn.ReLU()
        self.activation = nn.Softmax(dim=1)

    def forward(self, input):
        x = self.relu(self.layer_1(input))
        x = self.activation(self.layer_out(x))

        return x


class MultiClassModel:
    def __call__(self, num_classes):
        if num_classes != 3:
            logger.error("This model is intended to work with 3-class tabular datasets.")
            sys.exit(-1)

        return MultiClassNet()


class XORClassifier(nn.Module):
    """ 
    Simple MLP for XOR tests. Roughly follows implementation of 
       https://courses.cs.washington.edu/courses/cse446/18wi/sections/section8/XOR-Pytorch.html

    Includes a nonlinear boolean parameter to flip the Sigmoid activation function between the
    two linear layers on or off.
    """

    def __init__(self, in_features, num_classes, hidden_features, nonlinear=True):
        super(XORClassifier, self).__init__()

        self.linear1 = nn.Linear(in_features, hidden_features)

        if nonlinear:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Identity()

        self.linear2 = nn.Linear(hidden_features, num_classes)

        # Override default initialization with a N(0,1) distribution
        self.linear1.weight.data.normal_(0, 1)
        self.linear2.weight.data.normal_(0, 1)

        # Log the initialized values for testing reproducibility
        logger.info("Initialized two linear layers with the following weights:")
        logger.info(self.linear1.weight)
        logger.info(self.linear2.weight)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x) 
        x = self.linear2(x)
        return x


class XORModel:
    """
    Juneberry wrapper class for the XOR tests
    """

    def __call__(self, in_features=2, num_classes=2, hidden_features=2, nonlinear=True):

        # Juneberry requires classes to be natural counts, while the pytorch
        # binary loss functions, BCELoss and MSELoss, require a single output node.
        # 
        # Consequently, decrease num_class when it is set to two. 
        if num_classes == 2:
            num_classes = 1

        return XORClassifier(in_features, num_classes, hidden_features, nonlinear)
