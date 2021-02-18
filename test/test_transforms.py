#! /usr/bin/env python3

"""
The vast majority of transformers just wrap specific functional calls such
as calls in the image support. We expect those tests to cover that part of the
functionality.

These tests are to make sure that the transforms have the proper signature
and can be loaded by the transform_manager. Thus, in most cases all we need
to do is to pass the configuration into the transform manager.

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

import juneberry.transform_manager


def test_load_random_crop():
    config = [
        {
            'fullyQualifiedClass': 'juneberry.transforms.random_crop_mirror.RandomCropMirror',
            'kwargs': { "width_pixels": 0, "height_pixels": 0, "mirror": 0 }
        }
    ]

    jtm = juneberry.transform_manager.TransformManager(config)
    assert len(jtm) == 1


def test_load_mirror_flip():
    config = [
        {
            'fullyQualifiedClass': 'juneberry.transforms.random_mirror_flip.RandomMirrorFlip',
            'kwargs': {"mirror_chance": 0.0, "flip_chance": 0.0 }
        }
    ]

    jtm = juneberry.transform_manager.TransformManager(config)
    assert len(jtm) == 1


def test_load_random_shift():
    config = [
        {
            'fullyQualifiedClass': 'juneberry.transforms.random_shift.RandomShift',
            'kwargs': {"max_width": 0.0, "max_height": 0.0 }
        }
    ]

    jtm = juneberry.transform_manager.TransformManager(config)
    assert len(jtm) == 1
