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
import logging

from juneberry.config.tuning import TuningConfig
import juneberry.filesystem as jb_fs
import juneberry.loader as jb_loader
import juneberry.jb_logging as jb_logging
from juneberry.script_tools.training_sprout import TrainingSprout
from juneberry.tuning.tuner import Tuner

logger = logging.getLogger(__name__)


@dataclass
class TuningSprout(TrainingSprout):
    """
    The TuningSprout class extends the TrainingSprout class to include args related to model tuning.
    """
    # ========== SCRIPT ARGS ==========
    # ===== TUNING ARGS =====
    tuning_config_str: str = None

    # ========== DERIVED FROM SCRIPT ARGS ==========
    # ===== TUNING ATTRIBUTES =====
    tuning_config: TuningConfig = None

    def grow_from_args(self, args: Namespace, init_logging: bool = True, derive_attr: bool = True) -> None:
        """
        This method reads a Namespace of arguments and sets the corresponding attributes in the Sprout.
        :param args: A Namespace of arguments, typically created by passing arguments to a Juneberry script.
        :param init_logging: A boolean which controls whether or not to initialize logging when deriving attributes.
        :param derive_attr: A boolean which controls whether or not to derive attributes from the args.
        :return: Nothing.
        """
        # Start by setting the attributes in the TrainingSprout. Since logging and attribute derivation
        # will be handled by the TuningSprout, those aspects are not necessary in the TrainingSprout.
        super().grow_from_args(args, init_logging=False, derive_attr=False)

        # Now set the attributes listed in the TuningSprout.
        self.tuning_config_str = getattr(args, "tuningConfig", None)

        # Finally, derive the remaining TrainingSprout attributes.
        if derive_attr:
            self._derive_attributes(init_logging=init_logging)

    def _derive_attributes(self, init_logging: bool) -> None:
        """
        This method is responsible for deriving the remaining attributes in the TuningSprout that are not
        set directly via script args.
        :param init_logging: This boolean controls if logging should be initialized.
        :return:
        """
        # Set the model_manager and set up the model directory for tuning.
        self.model_manager = jb_fs.ModelManager(self.model_name)
        self.model_manager.setup_tuning()

        # Once the model_manager exists, logging can be initialized (if requested).
        if init_logging:
            self._initialize_logging()

        # Set the TuningSprout's ModelConfig.
        self.set_model_config()

        # Set the TuningSprout's TuningConfig.
        self.set_tuning_config()

        # Log the current values for the Sprout directories.
        self.log_sprout_directories()

    def _initialize_logging(self, banner: str = ">>> Juneberry Tuner <<<") -> None:
        """
        This method is responsible for setting up logging in a TuningSprout.
        :param banner: A string indicating which text to include in the banner message for a log section.
        :return: Nothing.
        """
        # Set the Tuning log file.
        self.log_file = self.log_dir / self.model_manager.get_tuning_log()

        # With the log file set in the Sprout set to a Tuning-specific log, the rest of the logging
        # initialization can just follow the same steps from the TrainingSprout class.
        super()._initialize_logging(banner=banner)

        # An additional root logger needs to be set to capture messages from ray.tune during tuning.
        jb_logging.setup_logger(log_file=self.log_file, log_prefix="", name="ray", level=self.log_level)

    def set_tuning_config(self, tuning_config_str: str = None) -> None:
        """
        This method is responsible for setting the tuning_config attribute for the TuningSprout.
        :param tuning_config_str: (Optional) A string indicating the location of the tuning config to load.
        :return: Nothing.
        """
        # If a tuning_config_str was provided, set the tuning_config_str attribute in the TuningSprout to the
        # provided value.
        if tuning_config_str is not None:
            self.tuning_config_str = tuning_config_str

        # Load the TuningConfig and set the tuning_config attribute.
        self.tuning_config = TuningConfig.load(self.tuning_config_str)

    def build_tuner(self) -> Tuner:
        """
        This method is responsible for constructing a Juneberry Tuner object.
        :return: A Juneberry Tuner object.
        """
        # Construct the Tuner object.
        tuner = jb_loader.construct_instance("juneberry.tuning.tuner.Tuner", kwargs={})

        # Use the TuningSprout attributes to set attributes in the Tuner.
        return self._pollinate_tuner(tuner)

    def _pollinate_tuner(self, tuner) -> Tuner:
        """
        This method is responsible for populating attributes in the Tuner based on attributes in the
        TuningSprout.
        :param tuner: The Tuner object receiving the attributes.
        :return: The Tuner object whose attributes have been adjusted to match the values in the
        TuningSprout.
        """
        # Set various attributes in the Tuner.
        tuner.trial_resources = self.tuning_config.trial_resources
        tuner.metric = self.tuning_config.tuning_parameters.metric
        tuner.mode = self.tuning_config.tuning_parameters.mode
        tuner.num_samples = self.tuning_config.sample_quantity
        tuner.scope = self.tuning_config.tuning_parameters.scope
        tuner.checkpoint_interval = self.tuning_config.tuning_parameters.checkpoint_interval

        # Set the Tuner's baseline model config, which is the starting point for tuning.
        tuner.baseline_model_config = self.lab.load_model_config(self.model_name)

        # Set the Tuner's tuning_sprout attribute to this TuningSprout.
        tuner.tuning_sprout = self

        return tuner
