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

from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from gloro.layers import MinMax

from gloro import GloroNet

def _add_activation(z, activation_type='relu'):
    if activation_type == 'relu':
        return Activation('relu')(z)

    elif activation_type == 'elu':
        return Activation('elu')(z)

    elif activation_type == 'softplus':
        return Activation('softplus')(z)

    elif activation_type == 'minmax':
        return MinMax()(z)

    else:
        raise ValueError(f'unknown activation type: {activation_type}')


def cnn_2C2F(
        input_shape,
        num_classes,
        activation='relu',
        initialization='orthogonal',
):

    # AOM: Hopefully this matches "(16,4,2).(32,4,2).100"
    x = Input(input_shape)
    z = Conv2D(
        16, 4, strides=2, padding='same', kernel_initializer=initialization)(x)
    z = _add_activation(z, activation)

    z = Conv2D(
        32, 4, strides=2, padding='same', kernel_initializer=initialization)(z)
    z = _add_activation(z, activation)

    z = Flatten()(z)
    z = Dense(100, kernel_initializer=initialization)(z)
    z = _add_activation(z, activation)
    y = Dense(num_classes, kernel_initializer=initialization)(z)

    return x, y


class GloroCNN2C2F():
    def __call__(self, num_classes, img_width, img_height, channels, epsilon,
                 activation='relu', initialization='orthogonal'):
        x, y = cnn_2C2F(
            input_shape=(img_height, img_width, channels),
            num_classes=num_classes,
            activation=activation,
            initialization=initialization)

        #return Model(x, y)
        return GloroNet(x, y, epsilon=epsilon)
