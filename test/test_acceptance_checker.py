#! /usr/bin/env python3

# ==========================================================================================================================================================
#  Copyright 2021 Carnegie Mellon University.
#
#  NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS"
#  BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER
#  INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED
#  FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM
#  FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT. Released under a BSD (SEI)-style license, please see license.txt
#  or contact permission@sei.cmu.edu for full terms.
#
#  [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see
#  Copyright notice for non-US Government use and distribution.
#
#  This Software includes and/or makes use of the following Third-Party Software subject to its own license:
#  1. Pytorch (https://github.com/pytorch/pytorch/blob/master/LICENSE) Copyright 2016 facebook, inc..
#  2. NumPY (https://github.com/numpy/numpy/blob/master/LICENSE.txt) Copyright 2020 Numpy developers.
#  3. Matplotlib (https://matplotlib.org/3.1.1/users/license.html) Copyright 2013 Matplotlib Development Team.
#  4. pillow (https://github.com/python-pillow/Pillow/blob/master/LICENSE) Copyright 2020 Alex Clark and contributors.
#  5. SKlearn (https://github.com/scikit-learn/sklearn-docbuilder/blob/master/LICENSE) Copyright 2013 scikit-learn
#      developers.
#  6. torchsummary (https://github.com/TylerYep/torch-summary/blob/master/LICENSE) Copyright 2020 Tyler Yep.
#  7. adversarial robust toolbox (https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/LICENSE)
#      Copyright 2018 the adversarial robustness toolbox authors.
#  8. pytest (https://docs.pytest.org/en/stable/license.html) Copyright 2020 Holger Krekel and others.
#  9. pylint (https://github.com/PyCQA/pylint/blob/master/COPYING) Copyright 1991 Free Software Foundation, Inc..
#  10. python (https://docs.python.org/3/license.html#psf-license) Copyright 2001 python software foundation.
#
#  DM20-1149
#
# ==========================================================================================================================================================

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
    ac = AcceptanceChecker(mm, max_epochs=5)
    ac._save_model = MagicMock()

    value = 0.1
    while not ac.done:
        ac.add_checkpoint(None, value)
        value += 0.1

    # The epoch based on should get to the end and NEVER call with a revision
    assert ac.current_epoch == 5
    ac._save_model.assert_called_with(None)


def test_threshold_checker():
    mm = ModelManager("test_model")
    ac = AcceptanceChecker(mm, max_epochs=10, threshold=0.6, comparator=acc_fun)
    ac._save_model = MagicMock()

    value = 0.1
    while not ac.done:
        ac.add_checkpoint(None, value)
        value += 0.1

    # This should reach the sixth epoch because of THRESHOLD
    assert ac.current_epoch == 6
    ac._save_model.assert_called_with(None)


def test_threshold_epoch_checker():
    mm = ModelManager("test_model")
    ac = AcceptanceChecker(mm, max_epochs=10, threshold=100, comparator=acc_fun)
    ac._save_model = MagicMock()

    value = 0.1
    while not ac.done:
        ac.add_checkpoint(None, value)
        value += 0.1

    # This should reach the TENTH epoch because of EPOCH
    assert ac.current_epoch == 10
    ac._save_model.assert_called_with(None)


def test_plateau_checker():
    mm = ModelManager("test_model")
    ac = AcceptanceChecker(mm, max_epochs=10, plateau_count=3, comparator=acc_fun)
    ac._save_model = MagicMock()

    #          1    2    3    4    5    6    7
    values = [0.1, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3]
    value_idx = 0
    while not ac.done:
        ac.add_checkpoint(None, values[value_idx])
        value_idx += 1

    # This should be the 5th because we have a plateau of 3. Then we should see a rename and some deletes.
    assert ac.current_epoch == 5
    ac._save_model.assert_called_with(None)


def test_plateau_epoch():
    mm = ModelManager("test_model")
    ac = AcceptanceChecker(mm, max_epochs=5, plateau_count=5, comparator=acc_fun)
    ac._save_model = MagicMock()

    #          1    2    3    4    5    6    7
    values = [0.1, 0.2, 0.3, 0.4, 0.4, 0.4, 0.4]
    value_idx = 0
    while not ac.done:
        ac.add_checkpoint(None, values[value_idx])
        value_idx += 1

    # This should be the 5th because we have a plateau of 3. Then we should see a rename and some deletes.
    assert ac.current_epoch == 5
    ac._save_model.assert_called_with(None)


def test_plateau_threshold():
    mm = ModelManager("test_model")
    ac = AcceptanceChecker(mm, max_epochs=5, threshold=0.2, plateau_count=5, comparator=acc_fun)
    ac._save_model = MagicMock()

    #          1    2    3    4    5    6    7
    values = [0.1, 0.2, 0.3, 0.4, 0.4, 0.4, 0.4]
    value_idx = 0
    while not ac.done:
        ac.add_checkpoint(None, values[value_idx])
        value_idx += 1

    # This should be the 5th because we have a plateau of 3. Then we should see a rename and some deletes.
    assert ac.current_epoch == 2
    ac._save_model.assert_called_with(None)


def test_comparator():
    # Used for loss and other comparator things
    mm = ModelManager("test_model")
    ac = AcceptanceChecker(mm, max_epochs=10, threshold=0.7, comparator=loss_fn)
    ac._save_model = MagicMock()

    value = 1
    while not ac.done:
        ac.add_checkpoint(None, value)
        value -= 0.1

    # This should reach the sixth epoch because of THRESHOLD
    assert ac.current_epoch == 4
    ac._save_model.assert_called_with(None)
