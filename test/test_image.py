#! /usr/bin/env python3

"""
Unit tests for core_utils for use by pytest.
"""

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

import juneberry.image as jbimage
import numpy as np

from PIL import Image


def make_raw_images():
    images = []
    for gray in range(0, 10):
        gray_frac = gray * 10
        color = f"rgb({gray_frac}%, {gray_frac}%, {gray_frac}%)"
        images.append(Image.new('RGB', (10, 10), color))

    return images


def test_compute_elementwise_mean() -> None:
    """
    We do one simple 2x2 test to make sure we get basic results.
    """
    a = np.array(range(1, 5), dtype='uint8')
    b = np.array(range(10, 50, 10), dtype='uint8')
    raw_correct = []
    for i in range(0, 4):
        raw_correct.append(int((a[i] + b[i]) / 2))

    a = a.reshape((2, 2))
    b = b.reshape((2, 2))
    correct = np.array(raw_correct).reshape((2, 2))

    results = jbimage.compute_elementwise_mean(np.array([a, b]))

    for i in range(0, 2):
        for j in range(0, 2):
            assert correct[i][j] == results[i][j]


def test_channel_means() -> None:
    images = make_raw_images()
    images = [np.array(image) for image in images]
    results = jbimage.compute_channel_means(images)
    assert results[0] == 0.45098039215686275
    assert results[1] == 0.45098039215686275
    assert results[2] == 0.45098039215686275
