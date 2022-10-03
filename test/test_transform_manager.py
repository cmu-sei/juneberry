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

import juneberry.transforms.transform_manager


def test_init_with_args():
    """
    Tests that we can construct a transform with specific kwargs
    :return:
    """
    config = [
        {
            'fqcn': 'moddir.simple_mod.ClassWithInitAndUnaryCall',
            'kwargs': {'name': 'frodo'}
        }
    ]

    ttm = juneberry.transforms.transform_manager.TransformManager(config)
    assert len(ttm) == 1
    assert ttm.get_fqn(0) == 'moddir.simple_mod.ClassWithInitAndUnaryCall'
    assert ttm.transform("baggins") == 'frodo baggins'


def test_inti_with_opt_args() -> None:
    """
    Tests that we can construct something with optional args.
    :return: None
    """
    config = [
        {
            'fqcn': 'moddir.simple_mod.ClassWithInitAndUnaryCall'
        }
    ]

    # NOTE: 'name' is expected but bar isn't, but 'bar' should be filtered out.
    opt_args = {"name": "frodo", "bar": 1234}
    ttm = juneberry.transforms.transform_manager.TransformManager(config, opt_args)
    assert len(ttm) == 1
    assert ttm.get_fqn(0) == 'moddir.simple_mod.ClassWithInitAndUnaryCall'
    assert ttm.transform("baggins") == 'frodo baggins'


def test_call_with_opt_args():
    """
    In this test we see if the TTM can take a set of optional arguments and apply
    the correct ones to the appropriate transforms.
    :return: None
    """
    config = [
        {
            'fqcn': 'moddir.simple_mod.ClassWithUnaryCallWithOptArg1'
        },
        {
            'fqcn': 'moddir.simple_mod.ClassWithUnaryCallWithOptArg2'
        }
    ]

    ttm = juneberry.transforms.transform_manager.TransformManager(config)
    opt_args = {'opt1': 'gamgee', 'opt2': 'hobbit', 'other': 'unused'}
    assert len(ttm) == 2
    assert ttm.get_fqn(0) == 'moddir.simple_mod.ClassWithUnaryCallWithOptArg1'
    assert ttm.get_fqn(1) == 'moddir.simple_mod.ClassWithUnaryCallWithOptArg2'
    assert ttm.all_opt_args == {'opt1', 'opt2'}

    # Each transform appends the optX arg to the object. Since we run them all in order,
    # we should get the '<obj> <opt1> <opt2>' as output.
    assert ttm.transform("sam", **opt_args) == 'sam gamgee hobbit'


def test_labeled_transforms():
    """
    In this test we look to see that the LabeledTransformManager supports transforms
    that CAN return two values (obj, label) and that we update the label.
    :return:
    """
    config = [
        {
            'fqcn': 'moddir.simple_mod.LabeledTransformExample'
        },
        {
            'fqcn': 'moddir.simple_mod.ClassWithUnaryCallWithOptArg2'
        }
    ]

    ttm = juneberry.transforms.transform_manager.LabeledTransformManager(config)
    opt_args = {'opt1': 'took', 'opt2': 'hobbit', 'label': 3, 'other': 'unused'}
    assert len(ttm) == 2
    assert ttm.get_fqn(0) == 'moddir.simple_mod.LabeledTransformExample'
    assert ttm.get_fqn(1) == 'moddir.simple_mod.ClassWithUnaryCallWithOptArg2'
    assert ttm.all_opt_args == {'opt1', 'opt2', 'label'}

    # In this case our labeled transform appends the opt arg and increments the number.
    # We can also mix in non-label changing ones that also appends data.
    assert ttm.transform("peregrine", **opt_args) == ('peregrine took hobbit', 4)


class StagedTransformManagerHarness(juneberry.transforms.transform_manager.StagedTransformManager):
    def __init__(self, consistent_seed: int, consistent, per_epoch_seed: int, per_epoch):
        super().__init__(consistent_seed, consistent, per_epoch_seed, per_epoch)
        self.save_called = False
        self.restore_called = False
        self.set_seeds_called = []

    def save_random_state(self):
        self.save_called = True

    def restore_random_state(self):
        self.restore_called = True

    def set_seeds(self, seed):
        self.set_seeds_called.append(seed)

    def reset(self):
        self.save_called = False
        self.restore_called = False
        self.set_seeds_called = []


class ConcatTransform():
    def __init__(self, val):
        self.val = val

    def __call__(self, obj, **kwargs):
        return f"{obj} {self.val}"


def test_staged_transform():
    """
    A base case test for a staged transform manager.
    :return:
    """
    consistent = ConcatTransform('baggins')
    per_epoch = ConcatTransform('hobbit')
    stm = StagedTransformManagerHarness(
        consistent_seed=100,
        consistent=consistent,
        per_epoch_seed=5,
        per_epoch=per_epoch)

    res = stm("frodo", index = 1, epoch= 2)
    assert stm.save_called
    assert stm.restore_called
    # The seeds should be consistent + index, per_epoch + index + epoch
    assert stm.set_seeds_called == [100 + 1, 5 + 1 + 2]
    assert res == "frodo baggins hobbit"

