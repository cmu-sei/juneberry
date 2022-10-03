#! /usr/bin/env python3

# ======================================================================================================================
# Juneberry - Release 0.5
#
# Copyright 2022 Carnegie Mellon University.
#
# NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS"
# BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER
# INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED
# FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM
# FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.
#
# Released under a BSD (SEI)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.
#
# [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution. Please see
# Copyright notice for non-US Government use and distribution.
#
# This Software includes and/or makes use of Third-Party Software each subject to its own license.
#
# DM22-0856
#
# ======================================================================================================================

from juneberry.config.model import PytorchOptions
import juneberry.pytorch.utils as pyt_utils


class DummyLoss:
    def __init__(self, model):
        model['return'] = 'World'

    def __call__(self, predicted, target):
        return predicted + target


def test_make_loss():
    # NOTE: This is run with the current directory (not the root test directory) in the python path
    config = PytorchOptions.from_dict({'loss_fn': 'pytorch.test_utils.DummyLoss'})
    model = {'input': 'Hello'}

    loss = pyt_utils.make_loss(config, model, False)
    assert model['return'] == 'World'
    assert loss(2, 3) == 5


class DummyLR:
    def __init__(self, optimizer, epochs, foo):
        self.optimizer = optimizer
        self.epochs = epochs
        self.foo = foo


def test_make_lr_schedule():
    lr_options = PytorchOptions.from_dict({
        "lr_schedule_args": {
            "epochs": 25,
            "foo": "bar"
        },
        "lr_schedule_fn": "pytorch.test_utils.DummyLR"
    })

    # These epochs should override
    lr_scheduler = pyt_utils.make_lr_scheduler(lr_options, "hello", 10)
    assert lr_scheduler.optimizer == "hello"
    assert lr_scheduler.epochs == 10
    assert lr_scheduler.foo == "bar"
