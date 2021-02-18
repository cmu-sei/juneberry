#! /usr/bin/env python3

"""
Simple transformer for mirroring or shifting an image.  The JSON configuration accept two arguments for chance to flip.

"config": { "mirror_chance": 0.0, "flip_chance": 0.0 }

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

import logging
import sys

import juneberry.image as iutils


class RandomMirrorFlip:
    def __init__(self, mirror_chance=None, flip_chance=None):
        self.mirror_chance = mirror_chance
        self.flip_chance = flip_chance

        if self.mirror_chance is None and self.flip_chance is None:
            logging.error(f"Neither 'mirror_chance' or 'flip_chance' found in specified. "
                          f"EXITING")
            sys.exit(-1)

        if self.mirror_chance is not None and not 0 <= self.mirror_chance <= 1:
            logging.error(f"mirror_chance must be a value in range (0,1)."
                          f"mirror_chance value was {self.mirror_chance}"
                          f"EXITING")
            sys.exit(-1)

        if self.flip_chance is not None and not 0 <= self.flip_chance <= 1:
            logging.error(f"flip_chance must be a value in range (0,1)."
                          f"flip_chance value was {self.flip_chance}"
                          f"EXITING")
            sys.exit(-1)

    def __call__(self, image):
        """
        Transformation function that is provided a PIL image.
        :param image: The source PIL image.
        :return: The transformed PIL image.
        """
        return iutils.random_mirror_flip(image, self.mirror_chance, self.flip_chance)
