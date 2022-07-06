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
from dataclasses import dataclass
import datetime
import logging
import os
from pathlib import Path
from typing import Union

from juneberry.config.model import ModelConfig
import juneberry.filesystem as jb_fs
from juneberry.lab import Lab
import juneberry.loader as jb_loader
import juneberry.jb_logging as jb_logging
import juneberry.scripting as jb_scripting_utils
from juneberry.script_tools.sprout import Sprout
from juneberry.trainer import Trainer
import juneberry.training.utils as jb_training_utils

logger = logging.getLogger(__name__)


@dataclass()
class TrainingSprout(Sprout):
    """
    The TrainingSprout class extends the base Sprout class to include attributes related to training
    models in Juneberry.
    """
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

    def grow_from_args(self, args: Namespace, init_logging: bool = True, derive_attr: bool = True) -> None:
        """
        This method reads a Namespace of arguments and sets the corresponding attributes in the Sprout.
        :param args: A Namespace of arguments, typically created by passing arguments to a Juneberry script.
        :param init_logging: A boolean which controls whether or not to initialize logging when deriving attributes.
        :param derive_attr: A boolean which controls whether or not to derive attributes from the args.
        :return: Nothing.
        """
        # Start by setting the attributes in the base Sprout.
        super().grow_from_args(args)

        # Now set the attributes stored in the TrainingSprout.
        self.model_name = getattr(args, "modelName", None)
        self.num_gpus = getattr(args, "num_gpus", None)
        self.dryrun = getattr(args, "dryrun", False)
        self.resume = getattr(args, "resume", False)
        self.skip_native = getattr(args, "skipNative", False)
        self.onnx = getattr(args, "onnx", False)

        # Finally, derive the remaining TrainingSprout attributes.
        if derive_attr:
            self._derive_attributes(init_logging=init_logging)

    def log_sprout_directories(self) -> None:
        """
        This method can be used to log the various directories that are associated with the Sprout. It is slightly
        different than the method in the base Sprout class, due to the presence of the lab attribute.
        :return: Nothing.
        """
        if self.lab is not None:
            logger.info(f"Directories associated with this sprout:")
            logger.info(f"  Workspace dir: {self.lab.workspace()}")
            logger.info(f"  Dataroot dir: {self.lab.data_root()}")
            logger.info(f"  Tensorboard dir: {self.lab.tensorboard}")
            logger.info(f"  Log dir: {self.log_dir}")
        else:
            super().log_sprout_directories()

    def _derive_attributes(self, init_logging: bool) -> None:
        """
        This method is responsible for deriving the remaining attributes in the TrainingSprout that are not
        set directly via script args.
        :param init_logging: This boolean controls if logging should be initialized.
        :return:
        """
        # Set the model_manager.
        self.model_manager = jb_fs.ModelManager(self.model_name, validate_dir=True)
        # TODO: How should JB log if there's no ModelManager?
        # TODO: self.log_dir should influence the location of the log.
        # Once a model_manager exists, logging can be initialized (if requested).
        if init_logging:
            # Determine which file to log to.
            if self.dryrun:
                self.log_file = self.model_manager.get_training_dryrun_log_path()
            else:
                self.log_file = self.model_manager.get_training_log()

            # Initialize logging.
            self._initialize_logging(banner=">>> Juneberry Trainer <<<")

        # Set the output format attributes.
        self._determine_output_format()

        # Set the TrainingSprout's ModelConfig.
        self.set_model_config()

        # Log the current values for the Sprout directories.
        self.log_sprout_directories()

    def _determine_output_format(self) -> None:
        """
        This method is responsible for converting the arguments related to output formats into attributes
        which indicate what output formats to use for the trained model.
        :return: Nothing.
        """
        # Set the booleans for each output format appropriately.
        self.native_output_format = not self.skip_native
        self.onnx_output_format = self.onnx

        # At least one output format must be chosen, so if neither were selected then choose the
        # native format by default.
        if not (self.onnx_output_format or self.native_output_format):
            logger.warning(f"An output format was not set. Defaulting to the native format.")
            self.native_output_format = True

    def set_model_config(self, model_config: ModelConfig = None) -> None:
        """
        This method is responsible for setting the model_config attribute for the TrainingSprout. When a
        ModelConfig is not provided to the method, the model config.json will be loaded from the
        ModelManager's model directory.
        :param model_config: A Juneberry ModelConfig object to associate with the TrainingSprout.
        :return: Nothing.
        """
        # Handle the situation where a ModelConfig was not provided to the method.
        if model_config is None:

            # If the TrainingSprout has a ModelManager, use it to load the model config from the model directory.
            if self.model_manager is not None:
                self.model_config = ModelConfig.load(self.model_manager.get_model_config())
                self._set_lab_attribute()
                return

            # When there is no ModelManager associated with the TrainingSprout, the Sprout is unable
            # to determine which model config file to load.
            else:
                logger.warning(f"Attempted to set a model config in the Sprout, but no model config "
                               f"was provided.")
                logger.warning(f"  Either provide a ModelConfig to set_model_config() or associate a ModelManager "
                               f"with the Sprout.")

        # If a ModelConfig was provided to the method, set the model_config and lab attributes.
        else:
            self.model_config = model_config
            self._set_lab_attribute()

    def _set_lab_attribute(self) -> None:
        """
        This method is responsible for setting the TrainingSprout's lab attribute. The lab attribute can be
        initialized if it has never been set, or it can be adjusted to match the profile in the TrainingSprout's
        model_config attribute.
        :return:
        """
        # If a Lab hasn't been associated with the TrainingSprout yet, initialize one.
        if self.lab is None:
            self._initialize_lab()
        # Otherwise adjust the lab profile using the profile in the model config (if one exists).
        else:
            self.lab.setup_lab_profile(model_config=self.model_config)

    def _initialize_lab(self) -> None:
        """
        This method is responsible for initializing the Juneberry Lab object associated with the TrainingSprout.
        :return: Nothing
        """
        # Isolate the Lab args, resolve them, and then create a Lab object after the args have been validated.
        lab_args = Namespace(workspace=self.workspace_dir, dataRoot=self.dataroot_dir, tensorboard=self.tensorboard_dir,
                             profileName=self.profile_name)
        resolved_args = jb_scripting_utils.resolve_lab_args(lab_args)
        Lab.validate_args(**resolved_args)
        lab = Lab(**resolved_args)

        # Change to the workspace directory.
        logger.info(f"Changing directory to workspace: '{lab.workspace()}'")
        os.chdir(lab.workspace())

        # Set up the lab profile.
        lab.setup_lab_profile(model_name=self.model_name, model_config=self.model_config)

        # If the TrainingSprout requires GPUs, set the Lab GPUs to the same amount.
        if self.num_gpus is not None:
            lab.profile.num_gpus = self.num_gpus

        # Log the lab profile
        logger.info(f"Using lab profile: {lab.profile}")

        self.lab = lab

    def _initialize_logging(self, banner: str = ">>> New Section <<<") -> None:
        """
        This method is responsible for setting up logging.
        :param banner: A string indicating which text to include in the banner message for a log section.
        :return: Nothing.
        """
        # Choose a string to use for the log prefix.
        self.log_prefix = "<<DRY_RUN>> " if self.dryrun else ""

        # Set up basic logging for now.
        logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)

        # Determine if the desired log file exists.
        if self.log_file is not None and self.log_file.exists():

            # Delete the log file if the environment variable indicates Juneberry should delete old logs.
            if int(os.environ.get('JUNEBERRY_REMOVE_OLD_LOGS', 0)) == 1:
                self.log_file.unlink()

            # Otherwise preserve the existing log file. Get the current time, use it to construct a new name
            # for the existing log file, then rename the existing log file.
            else:
                logger.info("Keeping old log files. Specify 'JUNEBERRY_REMOVE_OLD_LOGS=1' to remove them.")
                time_val = datetime.datetime.fromtimestamp(os.path.getmtime(self.log_file)).strftime("%m%d%y_%H%M")
                timestamped_log_filename = Path(self.log_file.parent, f"{self.log_file.stem}_{time_val}.txt")
                self.log_file.rename(timestamped_log_filename)

        # Set up the "juneberry" root logger.
        jb_logging.setup_logger(self.log_file, log_prefix=self.log_prefix, log_to_console=not self.silent,
                                level=self.log_level, name="juneberry")

        # Log the desired banner.
        jb_logging.log_banner(logger, banner)

        # Log a simple test message to indicate when verbose logging has been enabled.
        logger.debug(f"DEBUG messages enabled.")

    def get_trainer(self) -> Union[None, Trainer]:
        """
        This method is responsible for using the attributes in the TrainingSprout to build and return a
        Juneberry Trainer object.
        :return: A Juneberry Trainer, or None when a Trainer could not be built.
        """
        # If a ModelConfig is not associated with the TrainingSprout, the Sprout can't determine
        # the type of Trainer to build.
        if self.model_config is None:
            logger.warning(f"There is no model config associated with the sprout. Unable to "
                           f"determine which type of trainer to build.")
            return None

        # When a ModelConfig does exist, ModelConfig properties can be used to determine which type
        # of Trainer to build. Return the constructed Trainer.
        else:
            return self._assemble_trainer()

    def _assemble_trainer(self) -> Trainer:
        """
        This method is responsible for constructing a Juneberry Trainer object using the TrainingSprout's
        attributes.
        :return: A Juneberry Trainer object.
        """
        # If the model config doesn't have a "trainer" stanza, then it's likely an older version. The
        # correct trainer fqcn will need to be retrieved via a task/platform mapping. There will be no kwargs.
        if self.model_config.trainer is None:
            trainer_fqcn = jb_training_utils.assemble_stanza_and_construct_trainer(self.model_config)
            trainer_kwargs = {}

        # Otherwise, retrieve the Trainer fqcn and kwargs from the ModelConfig's Trainer stanza.
        else:
            trainer_fqcn = self.model_config.trainer.fqcn
            trainer_kwargs = self.model_config.trainer.kwargs if self.model_config.trainer.kwargs is not None else {}

        # Set some additional Trainer kwargs to reflect TrainingSprout attributes. These kwargs are needed
        # to construct the Trainer instance.
        trainer_kwargs['lab'] = self.lab
        trainer_kwargs['model_manager'] = self.model_manager
        trainer_kwargs['model_config'] = self.model_config
        trainer_kwargs['log_level'] = self.log_level

        # The "dataset_config" kwarg must also be set, but it requires an extra step to determine the correct
        # location of the dataset config file within the workspace.
        dataset_config_path = self.lab.workspace() / self.model_config.training_dataset_config_path
        trainer_kwargs['dataset_config'] = self.lab.load_dataset_config(dataset_config_path)

        # Construct the Trainer object.
        trainer = jb_loader.construct_instance(trainer_fqcn, trainer_kwargs)

        # Set the output formats in the Trainer.
        trainer.onnx = self.onnx_output_format
        trainer.native = self.native_output_format

        return trainer
