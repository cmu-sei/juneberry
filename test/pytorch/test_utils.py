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

import random

from juneberry.config.dataset import SamplingAlgo, SamplingConfig
from juneberry.config.model import PytorchOptions, SplittingConfig
from juneberry.lab import Lab
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


def test_sample_and_split():
    # NOTE: We do not test actual randomization as we leave that to other tests
    randomizer1 = random.Random()
    randomizer1.seed(1234)

    randomizer2 = random.Random()
    randomizer2.seed(1234)

    sampling = SamplingConfig(SamplingAlgo.RANDOM_FRACTION, {"fraction": 0.5}, randomizer1)
    splitting = SplittingConfig("random_fraction", {"fraction": 0.1}, randomizer2)

    max_size = 200
    a, b = pyt_utils.sample_and_split(max_size, sampling_config=sampling, splitting_config=splitting)

    # Now we should have all the values.
    assert len(a) == 90
    assert len(b) == 10
    # They shouldn't have any of the same
    assert len(set(a) & set(b)) == 0

    # They should all be under max
    for i in a:
        assert i < max_size
    for i in b:
        assert i < max_size

    # The remainder should be 100 or half
    full = set(range(200))
    full = full - set(a)
    full = full - set(b)

    assert len(full) == 100


def test_dataset_view():
    # Construct a trivial subset view and see that it remaps
    dataset = ['a', 'b', 'c', 'd', 'e']
    new_view = ['c', 'a', 'b', 'e']
    new_order = [2, 0, 1, 4]
    view = pyt_utils.DatasetView(dataset, new_order)

    assert len(view) == 4

    # NOTE: The view is NOT an iterable
    for i in range(len(view)):
        assert view[i] == new_view[i]

class NoOpTransform():
    def __init__(self, type_name):
        self.type_name = type_name
        self.invoked = 0
        self.last_item = None

    def __call__(self, item):
        assert type(item).__name__ == self.type_name
        self.last_item = item
        self.invoked += 1
        return item


def test_torchvision_dataset_construction():
    lab = Lab()

    transform = NoOpTransform('Image')
    target_transform = NoOpTransform('Tensor')

    # Use the fake dataset to see if we construct something and we can get an argument into it.
    # The default size is 1000 so we need to do something different.
    dataset = pyt_utils.construct_torchvision_dataset(lab, 'torchvision.datasets.FakeData', '', {'size': 100},
                                                      data_transforms=transform, target_transforms=target_transform)
    assert dataset
    assert len(dataset) == 100

    # Now, grab one and check that the transforms are invoked.
    one = dataset[0]
    assert transform.invoked == 1
    assert target_transform.invoked == 1

