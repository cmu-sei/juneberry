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
The vast majority of transformers just wrap specific functional calls such
as calls in the image support. We expect those tests to cover that part of the
functionality.

These tests are to make sure that the transforms have the proper signature
and can be loaded by the transform_manager. Thus, in most cases all we need
to do is to pass the configuration into the transform manager.

"""

import juneberry.transforms.transform_manager


def test_load_random_crop():
    config = [
        {
            'fqcn': 'juneberry.transforms.random_crop_mirror.RandomCropMirror',
            'kwargs': {"width_pixels": 0, "height_pixels": 0, "mirror": 0}
        }
    ]

    jtm = juneberry.transforms.transform_manager.TransformManager(config)
    assert len(jtm) == 1


def test_load_mirror_flip():
    config = [
        {
            'fqcn': 'juneberry.transforms.random_mirror_flip.RandomMirrorFlip',
            'kwargs': {"mirror_chance": 0.0, "flip_chance": 0.0}
        }
    ]

    jtm = juneberry.transforms.transform_manager.TransformManager(config)
    assert len(jtm) == 1


def test_load_random_shift():
    config = [
        {
            'fqcn': 'juneberry.transforms.random_shift.RandomShift',
            'kwargs': {"max_width": 0.0, "max_height": 0.0}
        }
    ]

    jtm = juneberry.transforms.transform_manager.TransformManager(config)
    assert len(jtm) == 1
