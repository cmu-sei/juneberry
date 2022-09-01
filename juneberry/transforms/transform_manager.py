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
from typing import Any

import juneberry.loader as jb_loader
import juneberry.utils as jb_utils

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

    We assume a call's signature where the first element is always the thing
    to be transformed, and all other arguments are by name.  So
    __call__(x, *, opt_1=None, opt2_=None, etc)
    ```
    """

    class Entry:
        def __init__(self, fqcn: str, kwargs: dict = None, opt_args: dict = None):
            """
            Initializes an entry which construct an instance with the kwargs and
            optional args for the function.
            :param fqcn: Fully Qualified Class Name
            :param kwargs: The keyword args from the config
            :param opt_args: Optional construction args
            """
            self.fqcn = fqcn
            self.kwargs = kwargs

            # The transform - Do NOT call this directly. Use the __call__ API.
            self.transform = None
            self.extra_params = []

            # Load the instance
            logger.info(f"Constructing transform: {fqcn} with args: {kwargs}")
            self.transform = jb_loader.construct_instance(fqcn, kwargs, opt_args)

            # Grab all the extra arguments so we can call it with those
            self.extra_params = jb_loader.extract_kwarg_names(self.transform)
            logger.info(f"... wants extra params {self.extra_params}")

        def __call__(self, obj: Any, **kwargs) -> Any:
            if obj is None:
                return None

            # Prune down the big pile of kwargs to what they want
            pruned_args = {}
            for k in self.extra_params:
                if k in kwargs:
                    pruned_args[k] = kwargs[k]
                else:
                    logger.error(f"Failed to find arg '{k}' in kwargs. {kwargs} {self.extra_params}")
                    raise RuntimeError("See log for details.")
            # pruned_args = {k: kwargs[k] for k in self.extra_params}
            return self.transform(obj, **pruned_args)

    def __init__(self, config: list, opt_args: dict = None):
        """
        Initializer that takes the transform stanza as configuration
        :param config: A configuration list of dicts of name, args.
        :param opt_args: A series of optional arguments to pass in during CONSTRUCTION.
        """
        self.config = []

        # A set of all the optional arguments wanted by all the transforms for preflight verification
        self.all_opt_args = set()

        # TODO: Switch to prodict
        for i in config:
            if 'fqcn' not in i:
                logger.error(f"Transform entry does not have required key 'fqcn' {i}")
                sys.exit(-1)

            entry = TransformManager.Entry(i['fqcn'], i.get('kwargs', None), opt_args)

            # Keep a master list (set) of all the extra params wanted
            self.all_opt_args.update(entry.extra_params)

            self.config.append(entry)

    def __call__(self, obj: Any, **kwargs) -> Any:
        """
        Performs all the transformations, in sequence, on the input and returns the last output.

        :param obj: The object to transform.
        :return: The transformed object.
        """
        for entry in self.config:
            obj = entry(obj, **kwargs)

        return obj

    def transform(self, obj: Any, **kwargs) -> Any:
        """
        Deprecated API for transforming the object.  Use __call__(obj) instead.
        :param obj: The object to transform.
        :return: The transformed object.
        """
        return self(obj, **kwargs)

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

    def get_transforms(self) -> list:
        """ :return: The transforms as a list. """
        return [x.transform for x in self.config]

    def __str__(self) -> str:
        return f"TransformManager len={len(self.config)}"

    def __repr__(self) -> str:
        # Note for repr we should have more data
        return self.__str__()


def unpack_return_values(retval, label, fqcn):
    """
    Transforms may return either (item) or (item, label). This function determines which and
    unpack accordingly.
    :param retval: The value(s) that came back.
    :param label: A label that was originally provided
    :param fqcn: Optional FQCN of the transform for debugging
    :return:
    """
    # TODO: Pre-determine if we should unpack or not and make a functor
    # Unpack the return value. If a tuple we assume it is (object, label).  We do NOT
    # do a list, because that could be real data.
    if isinstance(retval, tuple):
        if len(retval) != 2:
            logger.error(f"When running transform {fqcn} received a tuple with length {len(retval)} "
                         f"when expected two values.")
            raise RuntimeError("See log for details.")

        # Split the values and update the label
        obj, label = retval
    else:
        obj = retval

    return obj, label


class LabeledTransformManager(TransformManager):
    """
    This transform manager understand that labels CAN BE returned along with the object.
    Labels are passed in as an optional argument so they are placed into the proper spot,
    which is usually the second argument. However, we assume correctly called label
    not by position.
    """

    def __init__(self, config: list, opt_args: dict = None):
        """
        Initializer that takes the transform stanza as configuration
        :param config: A configuration list of dicts of name, args.
        :param opt_args: A series of optional arguments to pass in during CONSTRUCTION.
        """
        super().__init__(config, opt_args)

    def __call__(self, obj, **kwargs):
        """
        Performs all the transformations, in sequence, on the input and returns the last output
        along with the label.
        In this version the label is required in the opt_args.
        :param obj: The object to transform.
        :return: The transformed object and final label
        """
        if 'label' not in kwargs:
            logger.error(f"The LabeledTransformManager requires that 'label' be passed in as a kwarg. "
                         f"'{kwargs}' were provided.")
            raise RuntimeError("See log for details.")

        for entry in self.config:
            retval = entry(obj, **kwargs)
            obj, kwargs["label"] = unpack_return_values(retval, kwargs["label"], entry.fqcn)

        # We give back the object and accumulate label
        return obj, kwargs['label']


class StagedTransformManager:
    """
    A callable transform manager that manages the random seed state in a predictable fashion based
    on an initial seed state and the seed index.
    """

    def __init__(self, consistent_seed: int, consistent, per_epoch_seed: int, per_epoch):
        """
        Initialize the two stage manager with seeds and transforms for two stages.  The first
        stage, "consistent" is handled the same way for each epoch.  The second stage
        "epoch" is set differently for each epoch.
        Each seed will also be set differently based on the index of each element. Thus,
        regardless of the order the inputs are retrieved, they should provide the same value.
        :param consistent_seed: A seed for the 'con
        :param consistent: The transforms to be run consistently per epoch.
        :param per_epoch_seed: The BASE seed to be used for each epoch. It will be incremented every epoch.
        :param per_epoch: The transforms to be applied with the per-epoch seed.
        """
        # TODO: Check that the wrapped transforms take kwargs so we can pass anything through
        self.consistent_transform = consistent
        self.consistent_seed = consistent_seed
        self.per_epoch_transform = per_epoch
        self.per_epoch_seed = per_epoch_seed

    def __call__(self, item, **kwargs):
        if 'index' not in kwargs or 'epoch' not in kwargs:
            logger.error("The StagedTransformManager required 'index' and 'epoch' to be passed in as kwargs")
            raise RuntimeError("See log for details.")

        index = kwargs['index']
        epoch = kwargs['epoch']

        # Capture the random state
        self.save_random_state()

        # Set the random seed based on index only
        seed = jb_utils.wrap_seed(self.consistent_seed + index)
        self.set_seeds(seed)
        item = self.consistent_transform(item, **kwargs)

        # We might need to unpack the kwargs if label was available
        if 'label' in kwargs:
            item, kwargs['label'] = unpack_return_values(item, kwargs['label'], "Unknown")

        # Now, execute the per epoch transform
        seed = jb_utils.wrap_seed(self.per_epoch_seed + index + epoch)
        self.set_seeds(seed)
        item = self.per_epoch_transform(item, **kwargs)

        # We might need to unpack the kwargs if label was available
        if 'label' in kwargs:
            item, kwargs['label'] = unpack_return_values(item, kwargs['label'], "Unknown")

        # Restore the state
        self.restore_random_state()

        if 'label' in kwargs:
            return item, kwargs['label']
        else:
            return item

    # Extension points
    def save_random_state(self):
        pass

    def restore_random_state(self):
        pass

    def set_seeds(self, seed):
        pass
