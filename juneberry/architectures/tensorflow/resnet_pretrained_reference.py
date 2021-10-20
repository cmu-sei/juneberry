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

from tensorflow.keras.applications import resnet50
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

logger = logging.getLogger(__name__)


class Resnet50Pretrained:
    # This is the base Resent50 model from tensorflow.  Pretrained on imagenet.
    def __call__(self, num_classes, img_width, img_height, channels, labels):
        # https://keras.io/applications/
        return resnet50.ResNet50()


class Resnet50Finetuned:
    # Comments about the model
    # fine-tuned: base model is not trainable (only last dense layer is trainable)

    def __call__(self, num_classes, img_width, img_height, channels, labels):
        # https://keras.io/applications/
        base_model = resnet50.ResNet50(include_top=False,
                                       input_tensor=Input(shape=(img_height, img_width, channels)),
                                       pooling='avg')
        base_model.trainable = False
        x = base_model.output
        # prediction = Dense(len(set(labels)), activation="softmax")(x)
        prediction = Dense(num_classes, activation="softmax")(x)

        model = Model(inputs=base_model.input, outputs=prediction)

        return model
