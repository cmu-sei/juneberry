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
import datetime
import logging
import os
from pathlib import Path

from juneberry.config.model import ModelConfig
from juneberry.config.tuning import TuningConfig
import juneberry.filesystem as jb_fs
from juneberry.lab import Lab
import juneberry.loader as jb_loader
import juneberry.logging as jb_logging
import juneberry.scripting.utils as jb_scripting_utils
import juneberry.training.utils as jb_training_utils

logger = logging.getLogger(__name__)


class Sprout:

    def __init__(self, **kwargs):
        self.workspace_dir = None
        self.dataroot_dir = None
        self.tensorboard_dir = None
        self.log_dir = None

        self.silent = None
        self.log_level = None

        self.profile_name = None

        self.num_gpus = None
        self.dryrun = None

        self.model_name = None
        self.model_manager = None
        self.model_config = None
        self.lab = None

    def grow_from_args(self, args: Namespace):
        self.workspace_dir = getattr(args, "workspace", None)
        self.dataroot_dir = getattr(args, "dataRoot", None)
        self.tensorboard_dir = getattr(args, "tensorboard", None)
        self.log_dir = getattr(args, "logDir", None)
        self.silent = True if getattr(args, "silent", None) else False
        self.log_level = logging.DEBUG if getattr(args, "verbose", None) else logging.INFO
        self.profile_name = getattr(args, "profileName", None)

    def log_sprout_directories(self):
        logger.info(f"Directories associated with this sprout:")
        logger.info(f"  Workspace dir: {self.workspace_dir}")
        logger.info(f"  Dataroot dir: {self.dataroot_dir}")
        logger.info(f"  Tensorboard dir: {self.tensorboard_dir}")
        logger.info(f"  Log dir: {self.log_dir}")

    def set_model_config(self, model_config: ModelConfig = None):
        if model_config is None:
            if self.model_manager is not None:
                self.model_config = ModelConfig.load(self.model_manager.get_model_config())
                if self.lab is None:
                    self.lab = self.initialize_lab()
                else:
                    self.lab.setup_lab_profile(model_config=self.model_config)
                return
            else:
                logger.warning(f"Attempted to set a model config in the TrainingSprout, but no model config "
                               f"was provided.")
                return

        self.model_config = model_config
        if self.lab is None:
            self.lab = self.initialize_lab()
        else:
            self.lab.setup_lab_profile(model_config=self.model_config)

    def initialize_lab(self):
        lab_args = Namespace(workspace=self.workspace_dir, dataRoot=self.dataroot_dir, tensorboard=self.tensorboard_dir,
                             profileName=self.profile_name)
        resolved_args = jb_scripting_utils.resolve_lab_args(lab_args)
        Lab.validate_args(**resolved_args)
        lab = Lab(**resolved_args)

        logger.info(f"Changing directory to workspace: '{lab.workspace()}'")
        os.chdir(lab.workspace())

        lab.setup_lab_profile(model_name=self.model_name, model_config=self.model_config)

        if self.num_gpus is not None:
            lab.profile.num_gpus = self.num_gpus

        logger.info(f"Using lab profile: {lab.profile}")

        return lab

    def initialize_logging(self, log_file: Path = None, banner: str = ">>> New Section <<<"):
        log_prefix = "<<DRY_RUN>> " if self.dryrun else ""

        logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)
        # TODO: Maybe this isn't the right spot for the banner?
        banner_msg = banner
        jb_logging.log_banner(logger, banner_msg)

        if log_file is not None and log_file.exists():
            if int(os.environ.get('JUNEBERRY_REMOVE_OLD_LOGS', 0)) == 1:
                log_file.unlink()
            else:
                logger.info("Keeping old log files. Specify 'JUNEBERRY_REMOVE_OLD_LOGS=1' to remove them.")
                time_val = datetime.datetime.fromtimestamp(os.path.getmtime(log_file)).strftime("%m%d%y_%H%M")
                new_file_path = Path(log_file.parent, f"{log_file.stem}_{time_val}.txt")
                os.rename(log_file, new_file_path)

        jb_logging.setup_logger(log_file, log_prefix=log_prefix, log_to_console=not self.silent, level=self.log_level,
                                name="juneberry")

        logger.debug(f"DEBUG messages enabled.")

        self.log_sprout_directories()

    def build_trainer_from_model_config(self):
        if self.model_config is not None:
            if self.model_config.trainer is None:
                trainer = jb_training_utils.assemble_trainer_stanza(self.model_config)
            else:
                trainer = jb_loader.construct_instance(self.model_config.trainer.fqcn, self.model_config.trainer.kwargs)

            trainer.inherit_from_sprout(self)
            return trainer
        else:
            logger.warning(f"There is no model config associated with the training sprout. Unable to "
                           f"determine which type of trainer to build.")
            return None


class TrainingSprout(Sprout):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.num_gpus = 0
        self.resume = None

        self.native_output_format = None
        self.onnx_output_format = None

    def grow_from_args(self, args: Namespace, init_logging: bool = True):
        super().grow_from_args(args)

        self.model_name = args.modelName
        self.num_gpus = getattr(args, "num_gpus", None)
        self.dryrun = args.dryrun
        self.resume = getattr(args, "resume", None)

        skip_native_arg = getattr(args, "skipNative", False)
        onnx_arg = getattr(args, "onnx", False)

        self._determine_output_format(skip_native_arg, onnx_arg)

        self.model_manager = jb_fs.ModelManager(self.model_name)
        # TODO: How should JB log if there's no ModelManager?
        # TODO: self.log_dir should influence the location of the log.
        log_file = self.model_manager.get_training_dryrun_log_path() if self.dryrun \
            else self.model_manager.get_training_log()
        if init_logging:
            self.initialize_logging(log_file=log_file, banner=">>> Juneberry Trainer <<<")
        self.set_model_config()

    def _determine_output_format(self, native_arg, onnx_arg):
        self.native_output_format = not native_arg
        self.onnx_output_format = onnx_arg

        if not (self.onnx_output_format or self.native_output_format):
            logger.warning(f"An output format was not set. Defaulting to the native format.")
            self.native_output_format = True


class EvaluationSprout(Sprout):

    def __init__(self):
        super().__init__()
        self.model_name = None

        self.eval_dataset_name = None
        self.use_train_split = None
        self.use_val_split = None

        self.dryrun = None

        self.top_k = 0


class TuningSprout(TrainingSprout):

    def __init__(self):
        super().__init__()
        self.tuning_config_str = None
        self.tuning_config = TuningConfig()

    def grow_from_args(self, args: Namespace, init_logging: bool = True):
        super().grow_from_args(args, init_logging=False)
        self.model_name = args.modelName
        self.dryrun = args.dryrun
        self.tuning_config_str = args.tuningConfig

        self.model_manager = jb_fs.ModelManager(self.model_name)
        # TODO: How should JB log if there's no ModelManager?
        # TODO: Figure out where/how to save Tuning log messages
        log_file = None
        if init_logging:
            self.initialize_logging(log_file=log_file, banner=">>> Juneberry Tuner <<<")
        self.set_model_config()
        self.set_tuning_config()

    def set_tuning_config(self, tuning_config_str: str = None):
        target_string = self.tuning_config_str if tuning_config_str is None else tuning_config_str
        self.tuning_config = TuningConfig.load(target_string)

    def build_tuner(self):
        tuner = jb_loader.construct_instance("juneberry.tuning.tuner.Tuner", kwargs={})
        tuner.inherit_from_sprout(self)
        return tuner
