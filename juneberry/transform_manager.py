#! /usr/bin/env python3

# ======================================================================================================================
#  Copyright 2021 Carnegie Mellon University.
#
#  NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS"
#  BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER
#  INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED
#  FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM
#  FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.
#
#  Released under a BSD (SEI)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.
#
#  [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.
#  Please see Copyright notice for non-US Government use and distribution.
#
#  This Software includes and/or makes use of the following Third-Party Software subject to its own license:
#
#  1. PyTorch (https://github.com/pytorch/pytorch/blob/master/LICENSE) Copyright 2016 facebook, inc..
#  2. NumPY (https://github.com/numpy/numpy/blob/master/LICENSE.txt) Copyright 2020 Numpy developers.
#  3. Matplotlib (https://matplotlib.org/3.1.1/users/license.html) Copyright 2013 Matplotlib Development Team.
#  4. pillow (https://github.com/python-pillow/Pillow/blob/master/LICENSE) Copyright 2020 Alex Clark and contributors.
#  5. SKlearn (https://github.com/scikit-learn/sklearn-docbuilder/blob/master/LICENSE) Copyright 2013 scikit-learn 
#      developers.
#  6. torchsummary (https://github.com/TylerYep/torch-summary/blob/master/LICENSE) Copyright 2020 Tyler Yep.
#  7. pytest (https://docs.pytest.org/en/stable/license.html) Copyright 2020 Holger Krekel and others.
#  8. pylint (https://github.com/PyCQA/pylint/blob/main/LICENSE) Copyright 1991 Free Software Foundation, Inc..
#  9. Python (https://docs.python.org/3/license.html#psf-license) Copyright 2001 python software foundation.
#  10. doit (https://github.com/pydoit/doit/blob/master/LICENSE) Copyright 2014 Eduardo Naufel Schettino.
#  11. tensorboard (https://github.com/tensorflow/tensorboard/blob/master/LICENSE) Copyright 2017 The TensorFlow 
#                  Authors.
#  12. pandas (https://github.com/pandas-dev/pandas/blob/master/LICENSE) Copyright 2011 AQR Capital Management, LLC,
#             Lambda Foundry, Inc. and PyData Development Team.
#  13. pycocotools (https://github.com/cocodataset/cocoapi/blob/master/license.txt) Copyright 2014 Piotr Dollar and
#                  Tsung-Yi Lin.
#  14. brambox (https://gitlab.com/EAVISE/brambox/-/blob/master/LICENSE) Copyright 2017 EAVISE.
#  15. pyyaml  (https://github.com/yaml/pyyaml/blob/master/LICENSE) Copyright 2017 Ingy döt Net ; Kirill Simonov.
#  16. natsort (https://github.com/SethMMorton/natsort/blob/master/LICENSE) Copyright 2020 Seth M. Morton.
#  17. prodict  (https://github.com/ramazanpolat/prodict/blob/master/LICENSE.txt) Copyright 2018 Ramazan Polat
#               (ramazanpolat@gmail.com).
#  18. jsonschema (https://github.com/Julian/jsonschema/blob/main/COPYING) Copyright 2013 Julian Berman.
#
#  DM21-0689
#
# ======================================================================================================================

import logging
import sys

import juneberry.loader as loader

logger = logging.getLogger(__name__)


class TransformManager:
    """
    This class manages, constructs and executes a chain of transform objects constructed
    from classes of the following structure:

    ```
    class <MyTransformerClass>:
        def __init__(self, <config expanded from kwargs>):
            ... initialization code ...

        def __call__(self, object_to_transform):
            ... transformation ...
            return transformed_object
    ```

    These are specified in a config structure such as:
    ```
    [
        {
            "fqcn": <fully qualified name of transformer class that supports __call__(object)>,
            "kwargs": { <kwargs to be passed (expanded) to __init__ on construction> }
        }
    ]
    ```
    """

    class Entry:
        def __init__(self, fqcn: str, kwargs: dict = None):
            self.fqcn = fqcn
            self.kwargs = kwargs
            self.transform = None

    def __init__(self, config: list):
        """
        Initializer that takes the augmentations stanza as configuration
        :param config: A configuration list of dicts of name, args.
        """
        self.config = []

        for i in config:
            if 'fqcn' not in i:
                logger.error(f"Transform entry does not have required key 'fqcn' {i}")
                sys.exit(-1)

            entry = TransformManager.Entry(i['fqcn'], i.get('kwargs', None))

            logger.info(f"Constructing transform: {entry.fqcn} with args: {entry.kwargs}")
            entry.transform = loader.construct_instance(entry.fqcn, entry.kwargs)

            self.config.append(entry)

    def __call__(self, obj):
        """
        Performs all the transformations, in sequence, on the input and returns the last output.
        :param obj: The object to transform.
        :return: The transformed object.
        """
        for entry in self.config:
            if obj is not None:
                obj = entry.transform(obj)

        return obj

    def transform(self, obj):
        """
        Deprecated API for transforming the object.  Use __call__(obj) instead.
        :param obj: The object to transform.
        :return: The transformed object.
        """
        return self(obj)

    def __len__(self) -> int:
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
        return self.config[index].fqcn

    def transform_pair(self, a, b):
        # TODO: We can replace this just by passing in a tuple...
        """
        Performs all the transformations, in sequence, on the input pair and returns the last output.
        :param a: The first component
        :param b: The second component.
        :return: The transformed image and target.
        """
        for entry in self.config:
            a, b = entry.transform(a, b)

        return a, b

    def get_transforms(self) -> list:
        """ :return: The transforms as a list. """
        return [x.transform for x in self.config]

    def __str__(self) -> str:
        return f"TransformManager len={len(self.config)}"

    def __repr__(self) -> str:
        # Note for repr we should have more data
        return self.__str__()

