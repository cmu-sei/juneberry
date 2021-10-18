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

    def __init__(self, config: list, opt_args: dict = None):
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
            entry.transform = loader.construct_instance(entry.fqcn, entry.kwargs, opt_args)

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
