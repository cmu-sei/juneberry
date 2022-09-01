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

from functools import reduce
import logging
from pathlib import Path
import sys

import torch
from torchsummary import summary

import juneberry.filesystem as jb_fs
from juneberry.pytorch.utils import PyTorchPlatformDefinitions

logger = logging.getLogger(__name__)


class LoadModel:
    """
    Loads models from existing files or an URL.
    """

    def __init__(self, *, modelName=None, modelPath=None, modelURL=None, strict=True, loadKeyPath=None,
                 includePatterns=None, excludePatterns=None, renamePatterns=None):
        self.model_name = modelName
        self.model_path = modelPath
        self.model_url = modelURL
        self.strict = strict

        # We accept None (don't do), single argument, or lists of arguments.
        # Internally we like lists of arguments so promote if single argument.
        self.load_key_path = _ensure_list(loadKeyPath)
        self.include_patterns = _ensure_list(includePatterns)
        self.exclude_patterns = _ensure_list(excludePatterns)
        self.rename_patterns = _ensure_list_of_lists(renamePatterns)

        if self.model_name is None and self.model_path is None and self.model_url is None:
            logger.error("The LoadModel model transform requires either a modelName, modelPath, or modelURL. EXITING.")
            sys.exit(-1)

    def __call__(self, model):

        state_dict = {}
        if self.model_name is not None:
            logger.info(f"LoadModel transform loading model: {self.model_name}.")
            # Construct a model manager so we can get the model path.
            model_manager = jb_fs.ModelManager(self.model_name)
            model_path = model_manager.get_model_path(PyTorchPlatformDefinitions())
            if not Path(model_path).exists():
                logger.error(f"Model path {model_path} does not exist! EXITING.")
                sys.exit(-1)
            state_dict = torch.load(str(model_path))
        elif self.model_path is not None:
            logger.info(f"LoadModel transform loading model from Path: {self.model_path}.")
            if not Path(self.model_path).exists():
                logger.error(f"Model path {self.model_path} does not exist! EXITING.")
                sys.exit(-1)
            state_dict = torch.load(self.model_path)
        elif self.model_url is not None:
            logger.info(f"LoadModel transform loading model from URL: {self.model_url}.")
            state_dict = torch.hub.load_state_dict_from_url(self.model_url, progress=False)

        # The first step is to dig down to the path they want.
        if self.load_key_path is not None:
            for key in self.load_key_path:
                state_dict = state_dict[key]

        # DESIGN NOTE: This design focuses on identifying the keys we actually want,
        # then pulling over just that data to operations
        keep_map = self.filter_keys(state_dict.keys())
        logger.info(f"LoadModel loading keys: {keep_map}")

        # Construct the new state dict from the old one
        state_dict = {keep_map[k]: state_dict[k] for k in keep_map.keys()}

        # Shove whatever we have into the state dict of the model
        model.load_state_dict(state_dict, strict=self.strict)

        return model

    def filter_keys(self, input_keys):
        """
        Filters the input key set based on the patterns that were set during construction.
        We expose this for unit testing.
        :param input_keys: The keys to filter.
        :return: The filtered keys as a map of input key name to kept key name.
        """
        # NOTE: We are order preserving, so we can't use sets.
        keep_keys = [*input_keys]

        # Filter based on a positive include pattern, if we have any
        if self.include_patterns is not None:
            keep_keys = [k for k in keep_keys for p in self.include_patterns if p in k]

        # Move the keys to a new keys if we do NOT exclude
        if self.exclude_patterns is not None:
            tmp_keys = keep_keys
            keep_keys = []
            for key in tmp_keys:
                found_it = False
                for p in self.exclude_patterns:
                    if p in key:
                        found_it = True
                        break
                if not found_it:
                    keep_keys.append(key)

        # Build a key conversion map of names
        keep_map = {k: k for k in keep_keys}
        if self.rename_patterns is not None:
            for k in keep_keys:
                new_name = k
                for p in self.rename_patterns:
                    new_name = new_name.replace(p[0], p[1])

                keep_map[k] = new_name

        return keep_map


class SaveModel:
    """
    A transform for saving the model to a file specified by model name or file path.
    """

    def __init__(self, *, modelName=None, modelPath=None, overwrite=False):
        self.model_name = modelName
        self.model_path = modelPath
        self.overwrite = overwrite

        if self.model_name is None and self.model_path is None:
            logger.error("The SaveModel model transform requires either a modelName or modelPath. EXITING.")
            sys.exit(-1)

    def __call__(self, model):
        if self.model_name is not None:
            logger.info(f"SaveModel transform saving model: {self.model_name}.")
            # Construct a model manager so we can get the model path.
            model_manager = jb_fs.ModelManager(self.model_name)
            model_dir = model_manager.get_model_dir()
            if not Path(model_dir).exists():
                logger.error(f"Model directory {model_dir} does not exist! EXITING.")
                sys.exit(-1)

            model_path = model_manager.get_model_path(PyTorchPlatformDefinitions())
            if Path(model_path).exists() and not self.overwrite:
                logger.error(
                    f"Model file already exists and 'overwrite' setting is False. Model: {model_path} EXITING.")
                sys.exit(-1)

            torch.save(model.state_dict(), model_path)

        elif self.model_path is not None:
            logger.info(f"LoadModel transform loading model from Path: {self.model_path}.")
            if Path(self.model_path).exists() and not self.overwrite:
                logger.error(
                    f"Model file already exists and 'overwrite' setting is False. Model: {self.model_path} EXITING.")
                sys.exit(-1)

            torch.save(model.state_dict(), self.model_path)

        return model


class LogModelSummary:
    """
    Transform used to output the model summary to the console for debugging.
    """

    def __init__(self, image_shape):
        """
        Logs the image summary.
        :param image_shape: Shape in a tuple of C, W, H
        """
        # Can we make up an image shape?
        self.image_shape = image_shape

    def __call__(self, model):
        orig = sys.stdout
        sys.stdout.write = logger.info
        with torch.no_grad():
            summary(model, self.image_shape)
        sys.stdout = orig
        return model


class PrintModel:
    """
    Transform to do a plain print of the model object
    """

    def __init__(self):
        """
        Prints the model object
        """

    def __call__(self, model):
        with torch.no_grad():
            print(model)
        return model


class ReplaceFC:
    """
    A transform for replacing the fully connected layer. Useful for pre-trained models.
    """

    def __init__(self, num_classes, fc_name='fc', fc_bias=True):
        self.num_classes = num_classes
        self.fc_name = fc_name
        self.fc_bias = fc_bias

    @staticmethod
    def get_module_by_name(module, access_string):
        names = access_string.split(sep='.')
        return reduce(getattr, names, module)

    @staticmethod
    def set_module_by_name(module, access_string, value):
        names = access_string.split(sep='.')
        x = module
        for (i, name) in enumerate(names):
            if i == len(names) - 1:
                setattr(x, name, value)
            else:
                x = getattr(x, name)
        return module

    def __call__(self, model):

        # Find the last linear layer
        original_layer = self.get_module_by_name(model, self.fc_name)
        in_features = original_layer.in_features
        new_layer = torch.nn.modules.linear.Linear(in_features=in_features, out_features=self.num_classes,
                                                   bias=self.fc_bias)
        model = self.set_module_by_name(model, self.fc_name, new_layer)

        return model


class Freeze:
    """
    Transform used to freeze a pre-trained model
    """

    def __init__(self):
        pass

    def __call__(self, model):
        for param in model.parameters():
            param.requires_grad = False
        return model


class EmptyTransform:
    def __init__(self):
        pass

    def __call__(self, model):
        return model


# Utilities
def _ensure_list(args):
    if args is None:
        return None

    if isinstance(args, list):
        return args

    return [args]


def _ensure_list_of_lists(args):
    if args is None:
        return None

    if isinstance(args[0], list):
        return args

    return [args]
