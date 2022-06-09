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

from argparse import Namespace
import logging

from juneberry.config.model import ModelConfig
import juneberry.filesystem as jb_fs
import juneberry.loader as jb_loader
import juneberry.training.utils as jb_training_utils

logger = logging.getLogger(__name__)


class Sprout:

    def __init__(self):
        self.workspace_dir = None
        self.dataroot_dir = None
        self.tensorboard_dir = None
        self.log_dir = None

        self.silent = None
        self.log_level = None

        self.profile_name = None

    def grow_from_args(self, args: Namespace):
        self.workspace_dir = args.workspace
        self.dataroot_dir = args.dataRoot
        self.tensorboard_dir = args.tensorboard
        self.log_dir = args.logDir
        self.silent = True if args.silent else False
        self.log_level = logging.DEBUG if args.verbose else logging.INFO
        self.profile_name = args.profileName


class TrainingSprout(Sprout):

    def __init__(self):
        super().__init__()
        self.model_name = None

        self.num_gpus = 0
        self.dryrun = None
        self.resume = None

        self.native_output_format = None
        self.onnx_output_format = None

        self.model_manager = None
        self.model_config = ModelConfig()

    def grow_from_args(self, args: Namespace):
        super().grow_from_args(args)

        self.model_name = args.modelName
        if self.model_name is not None:
            self.model_manager = jb_fs.ModelManager(self.model_name)

        self.num_gpus = args.num_gpus
        self.dryrun = args.dryrun
        self.resume = args.resume

        self._determine_output_format(args.skipNative, args.onnx)

    def _determine_output_format(self, native_arg, onnx_arg):
        self.native_output_format = not native_arg
        self.onnx_output_format = onnx_arg

        if not (self.onnx_output_format or self.native_output_format):
            logger.warning(f"An output format was not set. Defaulting to the native format.")
            self.native_output_format = True

    def set_model_config(self, model_config: ModelConfig = None):
        if model_config is None:
            if self.model_manager is not None:
                self.model_config = ModelConfig.load(self.model_manager.get_model_config())
                return
            else:
                logger.warning(f"Attempted to set a model config in the TrainingSprout, but no model config "
                               f"was provided.")
                return

        self.model_config = model_config

    def get_trainer(self):
        if self.model_config is not None:
            if self.model_config.trainer is None:
                return jb_training_utils.assemble_trainer_stanza(self.model_config)
            else:
                return jb_loader.construct_instance(self.model_config.trainer.fqcn, self.model_config.trainer.kwargs)
        else:
            logger.warning(f"There is no model config associated with the training sprout. Unable to "
                           f"determine which type of trainer to build.")
            return None


class EvaluationSprout(Sprout):

    def __init__(self):
        super().__init__()
        self.model_name = None

        self.eval_dataset_name = None
        self.use_train_split = None
        self.use_val_split = None

        self.dryrun = None
        self.num_gpus = 0

        self.top_k = 0

