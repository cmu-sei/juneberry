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

"""
This base class provides a skeleton loop for supervised training and evaluating classifiers
and object detectors.
It makes the following assumptions:
- Data comes with targets/labels.

It provides the following features:
- A variety of extension points to customize the process.
- Data can be provided in batches.
- Training can be terminated early based on extensible stopping criteria.
- Each stage is independently timed.
"""

import datetime
import logging

from tqdm import tqdm

from juneberry.config.dataset import DatasetConfig
from juneberry.config.model import ModelConfig
from juneberry.config.training_output import TrainingOutputBuilder
from juneberry.filesystem import ModelManager
import juneberry.filesystem as jb_fs
from juneberry.logging import log_banner
from juneberry.lab import Lab
from juneberry.timing import Berryometer

logger = logging.getLogger(__name__)


class Trainer:
    """
    This is the base class for all trainers and organizes the model we are training the config associated
    with that model, and the dataset uses to train this model.
    """

    def __init__(self, lab: Lab, model_manager: ModelManager, model_config: ModelConfig, dataset_config: DatasetConfig,
                 log_level):
        """
        Constructs a basic trainer component.
        :param lab: The Juneberry lab in which to run the trainer.
        :param model_manager: The model manager of where to save model outputs.
        :param model_config: The model config to use in training.
        :param dataset_config: The dataset config to use when training.
        :param log_level: The message level to capture in the logging.
        """
        # This is the model we are associated with
        self.lab = lab
        self.model_manager = model_manager

        # The model and dataset configurations.
        self.model_config = model_config
        self.dataset_config = dataset_config

        # The level of logging for the Trainer.
        self.log_level = log_level

        # Store the time that training started / ended.
        self.train_start_time = None
        self.train_end_time = None

        # This is the GPU (if any) associated with this training process. None indicates CPU.
        self.gpu = None

        # Flag indicating if we are to run in distributed mode and the number of GPUs to use.
        self.distributed = False
        self.num_gpus = 0

        # A Berryometer object we use to track the times of phases. Can be used by the
        # subclasses for tracking the time of particular tasks.
        self.timer = Berryometer()

        # This is where all the results of the training process are placed
        # that we can serialize at the end.
        self.results_builder = TrainingOutputBuilder()
        self.results = self.results_builder.output

        # These booleans control which format(s) will be used when saving the trained model. All Trainers support
        # some kind of native format, but not all trainers will support the ONNX format.
        self.native = True
        self.onnx = False

        self.metrics_plugins = self.model_config.training_metrics

    # ==========================

    @classmethod
    def get_platform_defs(cls):
        """ :return: An object (PlatformDefinitions) that contains methods for various platform details. """
        logger.error(f"get_platform_defs() must be defined as a static method on the trainer")
        raise RuntimeError(f"get_platform_defs() must be defined as a static method on the trainer")

    # ==========================

    @staticmethod
    def get_training_output_files(model_mgr: ModelManager, dryrun: bool):
        """
        Returns a list of files to clean from the training directory. This list should contain ONLY
        files or directories that were produced by the training command. Directories in this list
        will be deleted even if they are not empty.
        :param model_mgr: A ModelManager to help locate files.
        :param dryrun: When True, returns a list of files created during a dryrun of the Trainer.
        :return: The files to clean from the training directory.
        """
        logger.error(f"get_training_output_files() must be defined as a static method on the trainer")
        raise RuntimeError(f"get_training_output_files() must be defined as a static method on the trainer")

    @staticmethod
    def get_training_clean_extras(model_mgr: ModelManager, dryrun: bool):
        """
        Returns a list of extra "training" files/directories to clean. Directories in this list will NOT
        be deleted if they are not empty.
        :param model_mgr: A ModelManager to help locate files.
        :param dryrun: When True, returns a list of files created during a dryrun of the Trainer.
        :return: The extra files to clean from the training directory.
        """
        logger.error(f"get_training_clean_extras() must be defined as a static method on the trainer")
        raise RuntimeError(f"get_training_clean_extras() must be defined as a static method on the trainer")

    def get_default_metric_value(self):
        """ :return: The value of the default metric in the results structure """
        logger.error(f"get_default_metric_value() not implemented in {self.__class__}")
        raise RuntimeError(f"get_default_metric_value() not implemented in {self.__class__}")

    # ==========================

    def set_output_format(self, native: bool = True, onnx: bool = False):
        """
        This method can be used to set the Trainer's output format attributes. If no output format is
        detected, then the "native" format will be chosen by default.
        :param native: A boolean indicating if the Trainer should save the trained model file in the
        platform's native format. The default value is True.
        :param onnx: A boolean indicating if the Trainer should save the trained model file in ONNX
        format, if the training platform supports saving ONNX model files. The default values is False.
        :return: Nothing.
        """
        # Set the attributes.
        self.native = native
        self.onnx = onnx

        # If the trainer has no output format, choose the native format by default even if
        # the user requested it to be skipped.
        if not (self.onnx or self.native):
            logger.warning(f"An output format was not set for the Trainer. Enabling the native format.")
            self.native = True

    # ==========================

    def dry_run(self) -> None:
        """
        Executes a "dryrun" of the training checking for model viability, data set properties, etc.
        :return: None
        """
        logger.warning("No Dry Run implemented in base Trainer")

    # ==========================

    def train_model(self, gpu: int = None) -> None:
        """
        Executes construction, training and serialization of the model.
        :param gpu: The gpu/process number to use for training.  None indicates CPU only.
        :return: None
        """
        self.gpu = gpu

        # This allows each platform to have its own particular way of setting up logging.
        # Logging must be set up prior to the first logging banner.
        self.establish_loggers()

        log_banner(logger, "Setup")
        with self.timer("setup"):
            self.setup()

        log_banner(logger, "Training")
        self.train_start_time = datetime.datetime.now().replace(microsecond=0)
        self.train()
        self.train_end_time = datetime.datetime.now().replace(microsecond=0)

        log_banner(logger, "Finalizing")
        with self.timer("finalize"):
            self.finish()

    # ==========================

    def node_setup(self) -> None:
        """ Called to prepare the node for either single process or distributed training. """
        pass

    def establish_loggers(self) -> None:
        logger.warning("establish_loggers() not implemented in base Trainer.")

    def setup(self) -> None:
        logger.warning("setup() not implemented in base Trainer")

    def train(self) -> None:
        logger.warning("train() not implemented in base Trainer")

    def tune(self) -> None:
        logger.warning("tune() not implemented in base Trainer")

    def finish(self) -> None:
        logger.warning("finish() not implemented in base Trainer")

    # ==========================

    def check_gpu_availability(self, required: int = None):
        """
        This allows the particular backend to use its own method of determining resource
        availability.
        :param required: The number of required gpus. 'None' will use the maximum available.
        :return: The number of gpus the trainer can use.
        """
        return 0

    def train_distributed(self, num_gpus) -> None:
        """
        Executes the training of the model in a distributed fashion.
        :param num_gpus: The number of gpus to use for training.
        :return: None
        """
        logger.warning("train_distributed() not implemented in the base Trainer.")

    # ==========================
    # Some utility methods

    def _finalize_results_prep(self):
        end_time = datetime.datetime.now().replace(microsecond=0)

        # Record the training times in the training output.
        self.results_builder.set_times(start_time=self.train_start_time, end_time=end_time)

        # Capture the relevant properties from the model config in the training options.
        self.results_builder.set_from_model_config(model_name=self.model_manager.model_name,
                                                   model_config=self.model_config)

        # Capture the relevant properties from the dataset config in the training options.
        self.results_builder.set_from_dataset_config(dataset_config=self.dataset_config)

    def _serialize_results(self):

        if self.gpu:
            logger.info("Only the rank 0 process is responsible for writing the training results to file.")

        else:
            logger.info(f"Writing output file: {self.model_manager.get_training_out_file()}")
            jb_fs.save_json(self.results.to_json(), self.model_manager.get_training_out_file())


#  _____                  _   _____          _
# | ____|_ __   ___   ___| |_|_   _| __ __ _(_)_ __   ___ _ __
# |  _| | '_ \ / _ \ / __| '_ \| || '__/ _` | | '_ \ / _ \ '__|
# | |___| |_) | (_) | (__| | | | || | | (_| | | | | |  __/ |
# |_____| .__/ \___/ \___|_| |_|_||_|  \__,_|_|_| |_|\___|_|
#       |_|

class EpochTrainer(Trainer):
    """
    This class encapsulates the process of training a supervised model epoch by epoch.
    """

    def __init__(self, lab, model_manager, model_config, dataset_config, log_level):
        """
        Construct an epoch trainer based on a model config and dataset config.
        :param lab: The Juneberry lab in which to run the trainer.
        :param model_config: A Juneberry ModelConfig object that describes how to construct and train the model.
        """
        # Set to true for dry run mode
        super().__init__(lab, model_manager, model_config, dataset_config, log_level)

        # Maximum number of epochs.  We may stop earlier if we meet other criteria
        self.max_epochs = model_config.epochs

        # A 1-based counter showing the current epoch. A value of zero means training has not begun.
        self.epoch = 0

        # Is the training done? Set to True to exit training loop.
        self.done = False if self.max_epochs > self.epoch else True

        # Iterables that provides batches (lists) of pairs of (data, targets) for training and evaluation.
        # These must be initialized during setup.
        self.training_iterable = None
        self.evaluation_iterable = None

        # Establish an empty list to track how long it takes to train each epoch. Failing to do this
        # will result in the EpochTrainer attempting to append duration values to a 'NoneType' object.
        self.results.times.epoch_duration_sec = []

    # -----------------------------------------------
    #  _____     _                 _              ______     _       _
    # |  ___|   | |               (_)             | ___ \   (_)     | |
    # | |____  _| |_ ___ _ __  ___ _  ___  _ __   | |_/ /__  _ _ __ | |_ ___
    # |  __\ \/ / __/ _ \ '_ \/ __| |/ _ \| '_ \  |  __/ _ \| | '_ \| __/ __|
    # | |___>  <| ||  __/ | | \__ \ | (_) | | | | | | | (_) | | | | | |_\__ \
    # \____/_/\_\\__\___|_| |_|___/_|\___/|_| |_| \_|  \___/|_|_| |_|\__|___/

    # ==========================

    def dry_run(self) -> None:
        """ Output all the dry run information about this model."""
        logger.warning("No Dry Run implemented in EpochTrainer")

    # ==========================

    def setup(self) -> None:
        """ Setup all data loaders, model and supporting functions."""
        logger.warning("setup() not implemented in base Trainer")

    def train(self) -> None:
        # For each epoch we need to do our basic training loops
        logger.info(f"Starting Training...")
        while not self.done:
            self._train_one_epoch()

    def tune(self):
        # For each epoch we need to complete a tuning interval.
        logger.info(f"Calculating the latest training metrics and sending them to the tuner...")
        while not self.done:
            yield self._tune_one_interval()

    def finish(self) -> None:
        """
        Called to finalize and close all resources.
        """
        logger.info(f"Finalizing results")
        # Add all our bits first
        self._finalize_results_prep()
        # Now let them fix-up as appropriate
        self.finalize_results()

        # Save it to a file.  NOTEBOOKS: We'll want to make this a separate step
        self._serialize_results()

    # ==========================

    def start_epoch_phase(self, train: bool):
        """
        Called at the beginning of the phase of an epoch.
        This should return a metrics object that will be passed to each batch
        for producing batch metrics. This object will also be passed to the
        summarization routine to summarize the phase.
        :param train: True if training, else evaluation
        :return: The metrics object to be tracked across the epoch
        """
        return {}

    def process_batch(self, train: bool, data, targets):
        """
        Process the batch of data and targets, returning a results object to be
        passed to update metrics.
        :param train: True if training, else evaluation
        :param data: The data to process.
        :param targets: The associated targets.
        :return: Results from the batch to be applied to epoch metrics.
        """
        return None

    def update_metrics(self, train: bool, metrics, results) -> None:
        """
        Called on every batch to update the epoch metrics with the specific batch results.
        :param train: True if training, else evaluating.
        :param metrics: The ongoing metrics to roll into results.
        :param results: The batch results.
        """
        pass

    def update_model(self, results) -> None:
        """
        Update the model based on the corresponding results.
        :param results: The results from processing the batch.
        """
        pass

    def summarize_metrics(self, train: bool, metrics) -> None:
        """
        Summarize the metrics captured across the epoch to the internal model
        for later use. Printing, showing as appropriate.
        :param train: True if training, else evaluating.
        :param metrics: Summarize the metrics across the batch.
        """
        pass

    def end_epoch(self, tuning_mode: bool = False) -> str:
        """
        Called at the end of epoch for model saving, external telemetry, etc.
        :param tuning_mode: Boolean indicating if the epoch is being used to tune the model.
        """
        return ""

    def finalize_results(self) -> None:
        """
        Add final results to the results structure. Create any needed
        diagrams, models, etc.
        """
        pass

    # -----------------------------------------------
    # ______     _            _
    # | ___ \   (_)          | |
    # | |_/ / __ ___   ____ _| |_ ___
    # |  __/ '__| \ \ / / _` | __/ _ \
    # | |  | |  | |\ V / (_| | ||  __/
    # \_|  |_|  |_| \_/ \__,_|\__\___|

    def _train_one_epoch(self) -> None:
        """
        Iterates the training set performing the forward function and updating
        the model after each batch.  Then evaluates the model using the training
        batch.

        Metrics are produced and updated for each batch.

        Performs timing instrumentation as appropriate.
        """
        self.epoch += 1

        with self.timer("epoch"):
            # Process all the data from the data loader
            self._process_one_iterable(True, self.training_iterable)

            # Process all the data from the data loader
            self._process_one_iterable(False, self.evaluation_iterable)

        # End of epoch, check for acceptance model checkpointing, etc.
        # This is outside of the epoch timer so the end has access to the time.
        with self.timer("end_epoch"):
            epoch_tracker = self.timer('epoch')
            msg = self.end_epoch()
            self.results_builder.record_epoch_duration(epoch_tracker.last_elapsed())

        # Deal with timing and logs
        epoch_tracker = self.timer('epoch')
        elapsed = epoch_tracker.last_elapsed()

        remaining_seconds = (self.max_epochs - self.epoch) * epoch_tracker.weighted_mean()
        eta = datetime.datetime.now() + datetime.timedelta(seconds=remaining_seconds)

        # Calculate the remaining time as if we get to max epoch.
        hours, remainder = divmod(remaining_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        logger.info(f"{self.epoch}/{self.max_epochs}, "
                    f"time_sec: {elapsed:.3f}, eta: {eta.strftime('%H:%M:%S')}, "
                    f"remaining: {int(hours):d}:{int(minutes):02d}:{int(seconds):02d}, "
                    f"{msg}")

    def _tune_one_interval(self):
        """
        Iterates the training set performing the forward function and returns metric data
        to the tuner. Then evaluates the model using the validation batch.
        Metrics are produced and updated for each batch.
        Performs timing instrumentation as appropriate.
        """
        self.epoch += 1

        with self.timer("epoch"):
            # Process all the data from the data loader
            self._process_one_iterable(True, self.training_iterable)

            # Process all the data from the data loader
            self._process_one_iterable(False, self.evaluation_iterable)

        with self.timer("end_epoch"):
            epoch_tracker = self.timer('epoch')
            return_val = self.end_epoch(tuning_mode=True)
            return return_val

    def _process_one_iterable(self, train: bool, data_iterable):
        """
        Iterates the specified data iterable providing each batch to process_batch.
        If in training mode (train is True) then the model is also updated after
        each batch.
        Performs timing instrumentation as appropriate.
        :param train: Set to true for training mode.
        :param data_iterable: The object to iterate that provides data and targets.
        :return:
        """
        label = "train" if train else "eval"
        metrics = self.start_epoch_phase(train)

        # Process each batch
        with self.timer(label):
            try:
                for data, targets in tqdm(data_iterable):
                    # Forwards
                    with self.timer(f"{label}_batch"):
                        results = self.process_batch(train, data, targets)

                    # Update the metrics (e.g. loss) based on this batch
                    with self.timer(f"update_{label}_metrics"):
                        self.update_metrics(train, metrics, results)

                    if train:
                        # Backprop
                        with self.timer("update_model"):
                            self.update_model(results)
            except FileNotFoundError:
                if self.dataset_config['url'] is not None:
                    raise FileNotFoundError(f"Data not found! Check your datasets folder and the pathing information "
                                            f"specified. If needed, download here {self.dataset_config.url}")
                else:
                    raise

        # Now that we have finished all the batches, summarize the metrics.
        with self.timer("summarize_train"):
            self.summarize_metrics(train, metrics)


def main():
    print("Nothing to see here.")


if __name__ == "__main__":
    main()
