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

import math
from unittest.mock import MagicMock

from juneberry.filesystem import ModelManager
from juneberry.pytorch.acceptance_checker import AcceptanceChecker


def acc_fun(x, y):
    return x - y


def loss_fn(x, y):
    if math.isclose(x, y, abs_tol=0.001):
        return 0
    return -x - -y


def test_epoch_checker():
    mm = ModelManager("test_model")
    ac = AcceptanceChecker(mm, comparator=acc_fun, max_epochs=5)
    ac._save_model = MagicMock()
    input_sample = None

    value = 0.1
    while not ac.done:
        ac.add_checkpoint(None, input_sample, value)
        value += 0.1

    # The epoch based on should get to the end and NEVER call with a revision
    assert ac.current_epoch == 5
    ac._save_model.assert_called_with(None, input_sample)


def test_threshold_checker():
    mm = ModelManager("test_model")
    ac = AcceptanceChecker(mm, comparator=acc_fun, max_epochs=10, threshold=0.6)
    ac._save_model = MagicMock()
    input_sample = None

    value = 0.1
    while not ac.done:
        ac.add_checkpoint(None, input_sample, value)
        value += 0.1

    # This should reach the sixth epoch because of THRESHOLD
    assert ac.current_epoch == 6
    ac._save_model.assert_called_with(None, input_sample)


def test_threshold_epoch_checker():
    mm = ModelManager("test_model")
    ac = AcceptanceChecker(mm, comparator=acc_fun, max_epochs=10, threshold=100)
    ac._save_model = MagicMock()
    input_sample = None

    value = 0.1
    while not ac.done:
        ac.add_checkpoint(None, input_sample, value)
        value += 0.1

    # This should reach the TENTH epoch because of EPOCH
    assert ac.current_epoch == 10
    ac._save_model.assert_called_with(None, input_sample)


def test_plateau_checker():
    mm = ModelManager("test_model")
    ac = AcceptanceChecker(mm, comparator=acc_fun, max_epochs=10, plateau_count=3)
    ac._save_model = MagicMock()
    input_sample = None

    #          1    2    3    4    5    6    7
    values = [0.1, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3]
    value_idx = 0
    while not ac.done:
        ac.add_checkpoint(None, input_sample, values[value_idx])
        value_idx += 1

    # This should be the 5th because we have a plateau of 3. Then we should see a rename and some deletes.
    assert ac.current_epoch == 5
    ac._save_model.assert_called_with(None, input_sample)


def test_plateau_epoch():
    mm = ModelManager("test_model")
    ac = AcceptanceChecker(mm, comparator=acc_fun, max_epochs=5, plateau_count=5)
    ac._save_model = MagicMock()
    input_sample = None

    #          1    2    3    4    5    6    7
    values = [0.1, 0.2, 0.3, 0.4, 0.4, 0.4, 0.4]
    value_idx = 0
    while not ac.done:
        ac.add_checkpoint(None, input_sample, values[value_idx])
        value_idx += 1

    # This should be the 5th because we have a plateau of 3. Then we should see a rename and some deletes.
    assert ac.current_epoch == 5
    ac._save_model.assert_called_with(None, input_sample)


def test_plateau_threshold():
    mm = ModelManager("test_model")
    ac = AcceptanceChecker(mm, comparator=acc_fun, max_epochs=5, threshold=0.2, plateau_count=5)
    ac._save_model = MagicMock()
    input_sample = None

    #          1    2    3    4    5    6    7
    values = [0.1, 0.2, 0.3, 0.4, 0.4, 0.4, 0.4]
    value_idx = 0
    while not ac.done:
        ac.add_checkpoint(None, input_sample, values[value_idx])
        value_idx += 1

    # This should be the 5th because we have a plateau of 3. Then we should see a rename and some deletes.
    assert ac.current_epoch == 2
    ac._save_model.assert_called_with(None, input_sample)


def test_comparator():
    # Used for loss and other comparator things
    mm = ModelManager("test_model")
    ac = AcceptanceChecker(mm, comparator=loss_fn, max_epochs=10, threshold=0.7)
    ac._save_model = MagicMock()
    input_sample = None

    value = 1
    while not ac.done:
        ac.add_checkpoint(None, input_sample, value)
        value -= 0.1

    # This should reach the sixth epoch because of THRESHOLD
    assert ac.current_epoch == 4
    ac._save_model.assert_called_with(None, input_sample)
