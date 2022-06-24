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
    workspace_dir: str = None
    dataroot_dir: str = None
    tensorboard_dir: str = None
    log_dir: str = None

    silent: bool = None
    log_level: int = None

    profile_name: str = None

    num_gpus: int = None
    dryrun: bool = None

    model_name: str = None
    model_manager: jb_fs.ModelManager = None
    model_config: ModelConfig = None
    lab: Lab = None

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
                logger.warning(f"Attempted to set a model config in the Sprout, but no model config "
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


@dataclass()
class TrainingSprout(Sprout):
    resume: bool = None

    native_output_format: bool = None
    onnx_output_format: bool = None

    def __repr__(self):
        return Prodict(asdict(self))

    def grow_from_args(self, args: Namespace, init_logging: bool = True):
        super().grow_from_args(args)

        self.model_name = getattr(args, "modelName", None)
        self.num_gpus = getattr(args, "num_gpus", None)
        self.dryrun = getattr(args, "dryrun", False)
        self.resume = getattr(args, "resume", False)

        skip_native_arg = getattr(args, "skipNative", False)
        onnx_arg = getattr(args, "onnx", False)

        self._determine_output_format(skip_native_arg, onnx_arg)

        self.model_manager = jb_fs.ModelManager(self.model_name, validate_dir=True)
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

    def get_trainer(self):
        if self.model_config is None:
            logger.warning(f"There is no model config associated with the sprout. Unable to "
                           f"determine which type of trainer to build.")
            trainer = None

        else:
            trainer = self._assemble_trainer()

        return trainer

    def _assemble_trainer(self):
        if self.model_config.trainer is None:
            trainer_fqcn = jb_training_utils.assemble_stanza_and_construct_trainer(self.model_config)
            trainer_kwargs = {}
        else:
            trainer_fqcn = self.model_config.trainer.fqcn
            trainer_kwargs = self.model_config.trainer.kwargs

        if trainer_kwargs is None:
            trainer_kwargs = {}

        trainer_kwargs['lab'] = self.lab
        trainer_kwargs['model_manager'] = self.model_manager
        trainer_kwargs['model_config'] = self.model_config
        trainer_kwargs['log_level'] = self.log_level

        dataset_config_path = self.lab.workspace() / self.model_config.training_dataset_config_path
        trainer_kwargs['dataset_config'] = self.lab.load_dataset_config(dataset_config_path)

        trainer = jb_loader.construct_instance(trainer_fqcn, trainer_kwargs)

        trainer.onnx = self.onnx_output_format
        trainer.native = self.native_output_format

        # trainer = self._pollinate_trainer(trainer)

        return trainer

    # def _pollinate_trainer(self, trainer: Trainer):
    #     if isinstance(trainer, ClassifierTrainer):
    #         return self._pytorch_classifier_pollination(trainer)
    #     if isinstance(trainer, Detectron2Trainer):
    #         return self._detectron2_trainer_pollination(trainer)
    #     if isinstance(trainer, MMDTrainer):
    #         return self._mmdetection_trainer_pollination(trainer)
    #     if isinstance(trainer, TFTrainer):
    #         return self._tensorflow_trainer_pollination(trainer)
    #
    # def _base_trainer_pollination(self, trainer: Trainer):
    #     trainer.model_manager = self.model_manager
    #     trainer.model_config = self.model_config
    #     trainer.log_level = self.log_level
    #     trainer.num_gpus = self.num_gpus
    #     trainer.dryrun = self.dryrun
    #     trainer.native_output_format = self.native_output_format
    #     trainer.onnx_output_format = self.onnx_output_format
    #     trainer.lab = self.lab
    #     trainer.dataset_config = trainer.lab.load_dataset_config(trainer.model_config.training_dataset_config_path)
    #
    #     return trainer
    #
    # def _epoch_trainer_pollination(self, trainer: EpochTrainer):
    #     trainer = self._base_trainer_pollination(trainer)
    #
    #     trainer.max_epochs = self.model_config.epochs
    #     trainer.done = False if trainer.max_epochs > trainer.epoch else True
    #
    #     return trainer
    #
    # def _pytorch_classifier_pollination(self, trainer: ClassifierTrainer):
    #     trainer = self._epoch_trainer_pollination(trainer)
    #
    #     trainer.data_version = self.model_manager.model_version
    #     trainer.binary = trainer.dataset_config.is_binary
    #     trainer.pytorch_options = self.model_config.pytorch
    #
    #     trainer.no_paging = False
    #     if "JB_NO_PAGING" in os.environ and os.environ['JB_NO_PAGING'] == "1":
    #         logger.info("Setting to no paging mode.")
    #         trainer.no_paging = True
    #
    #     trainer.memory_summary_freq = int(os.environ.get("JUNEBERRY_CUDA_MEMORY_SUMMARY_PERIOD", 0))
    #
    #     trainer.lr_step_frequency = LRStepFrequency.EPOCH
    #
    #     return trainer
    #
    # def _detectron2_trainer_pollination(self, trainer: Detectron2Trainer):
    #     trainer = self._base_trainer_pollination(trainer)
    #     trainer.output_builder.set_from_model_config(self.model_manager.model_name, self.model_config)
    #
    #     trainer.resume = self.resume
    #     trainer.output_dir = self.model_manager.get_train_scratch_path()
    #     trainer.final_model_path = self.model_manager.get_detectron2_model_path()
    #
    #     return trainer
    #
    # def _mmdetection_trainer_pollination(self, trainer: MMDTrainer):
    #     trainer = self._base_trainer_pollination(trainer)
    #     trainer.working_dir = self.model_manager.get_train_scratch_path() if self.model_manager is not None else None
    #     trainer.dryrun = self.dryrun
    #
    #     # Fill out some of the output fields using the model name / model config.
    #     trainer.output_builder.set_from_model_config(self.model_manager.model_name, self.model_config)
    #
    #     return trainer
    #
    # def _tensorflow_trainer_pollination(self, trainer: TFTrainer):
    #     trainer = self._base_trainer_pollination(trainer)
    #     self.width = self.model_config.model_architecture.args['img_width']
    #     self.height = self.model_config.model_architecture.args['img_height']
    #     self.channels = self.model_config.model_architecture.args['channels']
    #
    #     return trainer


@dataclass
class TuningSprout(TrainingSprout):
    tuning_config_str: str = None
    tuning_config: TuningConfig = None

    def __repr__(self):
        return Prodict(asdict(self))

    def grow_from_args(self, args: Namespace, init_logging: bool = True):
        super().grow_from_args(args, init_logging=False)
        self.model_name = getattr(args, "modelName", None)
        self.dryrun = getattr(args, "dryrun", False)
        self.tuning_config_str = getattr(args, "tuningConfig", None)

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
