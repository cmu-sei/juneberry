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

import logging

import juneberry.loader as loader


class TransformManager:
    """
    This class manages, constructs and executes a series of data transformers.
    Each transformer should be a class with the following structure. The first
    transformer must accept a PIL Image and must return a PIL Image or Pytorch Tensor.
    Subsequent transformers must accept the previous type.

    ```
    class <MyTransformerClass>:
        def __init__(self, <config expanded from kwargs>):
            ... initialization code ...

        def __call__(self, image_or_tensor):
            ... transformation ...
            return image_or_tensor
    ```

    These are specified in a config structure such as:
    ```
    [
        {
            "fullyQualifiedClass": <fully qualified name of transformer class that supports __call__(image_or_tensor)>,
            "kwargs": { <kwargs to be passed (expanded) to __init__ on construction> }
        }
    ]
    ```
    """

    def __init__(self, config):
        """
        Initializer that takes the augmentations stanza as configuration
        :param config: A configuration list of dicts of name, args.
        """
        self.config = config.copy()

        for entry in config:
            if 'kwargs' not in entry:
                entry['kwargs'] = None

            logging.info(f"Constructing transform: {entry['fullyQualifiedClass']} with args: {entry['kwargs']}")
            entry['transform'] = loader.construct_instance(entry['fullyQualifiedClass'], entry['kwargs'])

    def transform(self, image):
        """
        Performs all the transformations, in sequence, on the input and returns the last output.
        :param image: The image to transform.
        :return: The transformed image as a tensor.
        """
        for entry in self.config:
            image = entry['transform'](image)

        return image

    def __len__(self):
        """
        :return: The number of transformers in the list.
        """
        return len(self.config)

    def get_fqn(self, index: int) -> str:
        """
        Returns the fully qualified class name for the transformer at this index.
        :param index:  The index
        :return: The fully qualified class name
        """
        return self.config[index]['fullyQualifiedClass']

    def transform_image_and_target(self, image, target):
        """
        Performs all the transformations, in sequence, on the input image and target and returns the last output.
        :param image: The image to transform.
        :param target: The target to transform.
        :return: The transformed image and target.
        """
        for entry in self.config:
            image, target = entry['transform'](image, target)

        return image, target
