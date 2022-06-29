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
from dataclasses import asdict, dataclass
import datetime
import logging
import os
from pathlib import Path

from prodict import Prodict

from juneberry.config.model import ModelConfig
from juneberry.config.tuning import TuningConfig
import juneberry.filesystem as jb_fs
from juneberry.lab import Lab
import juneberry.loader as jb_loader
import juneberry.jb_logging as jb_logging
import juneberry.scripting as jb_scripting_utils
import juneberry.training.utils as jb_training_utils

logger = logging.getLogger(__name__)


@dataclass
class Sprout:
    # ========== SCRIPT ARGS ==========
    # ===== DIRECTORY ARGS =====
    workspace_dir: str = None
    dataroot_dir: str = None
    tensorboard_dir: str = None
    log_dir: str = None

    # ===== LOGGING ARGS =====
    silent: bool = None
    log_level: int = None

    # ===== LAB ARGS =====
    profile_name: str = None

    def __repr__(self):
        return Prodict(asdict(self))

    def grow_from_args(self, args: Namespace):
        self.workspace_dir = getattr(args, "workspace", None)
        self.dataroot_dir = getattr(args, "dataRoot", None)
        self.tensorboard_dir = getattr(args, "tensorboard", None)
        self.log_dir = getattr(args, "logDir", None)
        self.silent = getattr(args, "silent", False)
        self.log_level = logging.DEBUG if getattr(args, "verbose", None) else logging.INFO
        self.profile_name = getattr(args, "profileName", None)

    def log_sprout_directories(self):
        logger.info(f"Directories associated with this sprout:")
        logger.info(f"  Workspace dir: {self.workspace_dir}")
        logger.info(f"  Dataroot dir: {self.dataroot_dir}")
        logger.info(f"  Tensorboard dir: {self.tensorboard_dir}")
        logger.info(f"  Log dir: {self.log_dir}")


@dataclass()
class TrainingSprout(Sprout):
    # ========== SCRIPT ARGS ==========
    # ===== EXECUTION MODE ARGS =====
    dryrun: bool = None
    num_gpus: int = None
    resume: bool = None

    # ===== OUTPUT FORMAT ARGS =====
    onnx: bool = None
    skip_native: bool = None

    # ===== MODEL ARGS =====
    model_name: str = None

    # ========== DERIVED FROM SCRIPT ARGS ==========
    # ===== JUNEBERRY ATTRIBUTES =====
    model_manager: jb_fs.ModelManager = None
    model_config: ModelConfig = None
    lab: Lab = None

    # ===== LOGGING ATTRIBUTES =====
    log_prefix = None
    log_file = None

    # ===== OUTPUT FORMAT ATTRIBUTES =====
    native_output_format: bool = None
    onnx_output_format: bool = None

    def __repr__(self):
        return Prodict(asdict(self))

    def grow_from_args(self, args: Namespace, init_logging: bool = True, derive_attr: bool = True):
        super().grow_from_args(args)

        self.model_name = getattr(args, "modelName", None)
        self.num_gpus = getattr(args, "num_gpus", None)
        self.dryrun = getattr(args, "dryrun", False)
        self.resume = getattr(args, "resume", False)

        self.skip_native = getattr(args, "skipNative", False)
        self.onnx = getattr(args, "onnx", False)

        if derive_attr:
            self._derive_attributes(init_logging=init_logging)

    def log_sprout_directories(self):
        if self.lab is not None:
            logger.info(f"Directories associated with this sprout:")
            logger.info(f"  Workspace dir: {self.lab.workspace()}")
            logger.info(f"  Dataroot dir: {self.lab.data_root()}")
            logger.info(f"  Tensorboard dir: {self.lab.tensorboard}")
            logger.info(f"  Log dir: {self.log_dir}")
        else:
            super().log_sprout_directories()

    def _derive_attributes(self, init_logging: bool):
        self.model_manager = jb_fs.ModelManager(self.model_name, validate_dir=True)
        # TODO: How should JB log if there's no ModelManager?
        # TODO: self.log_dir should influence the location of the log.
        if init_logging:
            if self.dryrun:
                self.log_file = self.model_manager.get_training_dryrun_log_path()
            else:
                self.log_file = self.model_manager.get_training_log()

            self._initialize_logging(banner=">>> Juneberry Trainer <<<")

        self._determine_output_format()
        self.set_model_config()

        self.log_sprout_directories()

    def _determine_output_format(self):
        self.native_output_format = not self.skip_native
        self.onnx_output_format = self.onnx

        if not (self.onnx_output_format or self.native_output_format):
            logger.warning(f"An output format was not set. Defaulting to the native format.")
            self.native_output_format = True

    def set_model_config(self, model_config: ModelConfig = None):
        if model_config is None:
            if self.model_manager is not None:
                self.model_config = ModelConfig.load(self.model_manager.get_model_config())
                if self.lab is None:
                    self.lab = self._initialize_lab()
                else:
                    self.lab.setup_lab_profile(model_config=self.model_config)
                return
            else:
                logger.warning(f"Attempted to set a model config in the Sprout, but no model config "
                               f"was provided.")
                return

        self.model_config = model_config
        if self.lab is None:
            self.lab = self._initialize_lab()
        else:
            self.lab.setup_lab_profile(model_config=self.model_config)

    def _initialize_lab(self):
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

    def _initialize_logging(self, banner: str = ">>> New Section <<<"):
        self.log_prefix = "<<DRY_RUN>> " if self.dryrun else ""

        logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)

        if self.log_file is not None and self.log_file.exists():
            if int(os.environ.get('JUNEBERRY_REMOVE_OLD_LOGS', 0)) == 1:
                self.log_file.unlink()
            else:
                logger.info("Keeping old log files. Specify 'JUNEBERRY_REMOVE_OLD_LOGS=1' to remove them.")
                time_val = datetime.datetime.fromtimestamp(os.path.getmtime(self.log_file)).strftime("%m%d%y_%H%M")
                timestamped_log_filename = Path(self.log_file.parent, f"{self.log_file.stem}_{time_val}.txt")
                self.log_file.rename(timestamped_log_filename)

        jb_logging.setup_logger(self.log_file, log_prefix=self.log_prefix, log_to_console=not self.silent,
                                level=self.log_level, name="juneberry")

        jb_logging.log_banner(logger, banner)

        logger.debug(f"DEBUG messages enabled.")

    def get_trainer(self):
        if self.model_config is None:
            logger.warning(f"There is no model config associated with the sprout. Unable to "
                           f"determine which type of trainer to build.")
            return None

        else:
            return self._assemble_trainer()

    def _assemble_trainer(self):
        if self.model_config.trainer is None:
            trainer_fqcn = jb_training_utils.assemble_stanza_and_construct_trainer(self.model_config)
            trainer_kwargs = {}
        else:
            trainer_fqcn = self.model_config.trainer.fqcn
            trainer_kwargs = self.model_config.trainer.kwargs if self.model_config.trainer.kwargs is not None else {}

        trainer_kwargs['lab'] = self.lab
        trainer_kwargs['model_manager'] = self.model_manager
        trainer_kwargs['model_config'] = self.model_config
        trainer_kwargs['log_level'] = self.log_level

        dataset_config_path = self.lab.workspace() / self.model_config.training_dataset_config_path
        trainer_kwargs['dataset_config'] = self.lab.load_dataset_config(dataset_config_path)

        trainer = jb_loader.construct_instance(trainer_fqcn, trainer_kwargs)

        trainer.onnx = self.onnx_output_format
        trainer.native = self.native_output_format

        return trainer


@dataclass
class TuningSprout(TrainingSprout):
    # ========== SCRIPT ARGS ==========
    # ===== TUNING ARGS =====
    tuning_config_str: str = None

    # ========== DERIVED FROM SCRIPT ARGS ==========
    # ===== TUNING ATTRIBUTES =====
    tuning_config: TuningConfig = None

    def __repr__(self):
        return Prodict(asdict(self))

    def grow_from_args(self, args: Namespace, init_logging: bool = True, derive_attr: bool = True):
        super().grow_from_args(args, init_logging=False, derive_attr=False)

        self.model_name = getattr(args, "modelName", None)
        self.dryrun = getattr(args, "dryrun", False)
        self.tuning_config_str = getattr(args, "tuningConfig", None)

        if derive_attr:
            self._derive_attributes(init_logging=init_logging)

    def _derive_attributes(self, init_logging: bool):
        self.model_manager = jb_fs.ModelManager(self.model_name)
        self.model_manager.setup_tuning()

        if init_logging:
            self._initialize_logging()

        self.set_model_config()
        self.set_tuning_config()

        self.log_sprout_directories()

    def _initialize_logging(self, banner: str = ">>> Juneberry Tuner <<<"):
        self.log_file = self.log_dir / self.model_manager.get_tuning_log()
        super()._initialize_logging(banner=banner)
        jb_logging.setup_logger(log_file=self.log_file, log_prefix="", name="ray", level=self.log_level)

    def set_tuning_config(self, tuning_config_str: str = None):
        if tuning_config_str is not None:
            self.tuning_config_str = tuning_config_str

        self.tuning_config = TuningConfig.load(self.tuning_config_str)

    def build_tuner(self):
        tuner = jb_loader.construct_instance("juneberry.tuning.tuner.Tuner", kwargs={})
        tuner = self._pollinate_tuner(tuner)
        return tuner

    def _pollinate_tuner(self, tuner):
        tuner.trial_resources = self.tuning_config.trial_resources
        tuner.metric = self.tuning_config.tuning_parameters.metric
        tuner.mode = self.tuning_config.tuning_parameters.mode
        tuner.num_samples = self.tuning_config.sample_quantity
        tuner.scope = self.tuning_config.tuning_parameters.scope
        tuner.checkpoint_interval = self.tuning_config.tuning_parameters.checkpoint_interval

        tuner.baseline_model_config = self.lab.load_model_config(self.model_name)

        tuner.tuning_sprout = self

        return tuner


@dataclass
class EvaluationSprout(Sprout):
    eval_dataset_name: str = None
    use_train_split: bool = None
    use_val_split: bool = None
    top_k: int = None
