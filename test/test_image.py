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

"""
Unit tests for core_utils for use by pytest.
"""

import numpy as np
from PIL import Image

import juneberry.image as jb_image


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

    results = jb_image.compute_elementwise_mean(np.array([a, b]))

    for i in range(0, 2):
        for j in range(0, 2):
            assert correct[i][j] == results[i][j]


def test_channel_means() -> None:
    images = make_raw_images()
    images = [np.array(image) for image in images]
    results = jb_image.compute_channel_means(images)
    assert results[0] == 0.45098039215686275
    assert results[1] == 0.45098039215686275
    assert results[2] == 0.45098039215686275
