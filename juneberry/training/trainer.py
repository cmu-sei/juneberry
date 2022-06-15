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

from juneberry.config.dataset import DatasetConfig
from juneberry.config.model import ModelConfig
from juneberry.config.training_output import TrainingOutput
import juneberry.config.training_output
import juneberry.filesystem as jb_fs
from juneberry.lab import Lab
from juneberry.logging import log_banner
from juneberry.scripting.sprout import TrainingSprout
from juneberry.timing import Berryometer

logger = logging.getLogger(__name__)


class Trainer:

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # All attributes
        self.dataset_config = DatasetConfig()
        self.gpu = None
        self.distributed = False

        # Unique to trainer
        self.train_start_time = None
        self.train_end_time = None
        self.timer = Berryometer()
        self.results = TrainingOutput()
        self.results.options = juneberry.config.training_output.Options()
        self.results.times = juneberry.config.training_output.Times()
        self.results.times.epoch_duration_sec = []
        self.results.results = juneberry.config.training_output.Results()

        # Defined in Sprout
        self.lab = Lab()
        self.dryrun = False
        self.model_manager = None
        self.model_config = ModelConfig()
        self.log_level = None
        self.num_gpus = 0
        self.native_output_format = None
        self.onnx_output_format = None
        self.lab = None

    def inherit_from_sprout(self, sprout: TrainingSprout):
        self.model_manager = sprout.model_manager
        self.model_config = sprout.model_config
        self.log_level = sprout.log_level
        self.num_gpus = sprout.num_gpus
        self.dryrun = sprout.dryrun
        self.native_output_format = sprout.native_output_format
        self.onnx_output_format = sprout.onnx_output_format
        self.lab = sprout.lab
        self.dataset_config = self.lab.load_dataset_config(self.model_config.training_dataset_config_path)

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

        if self.dryrun:
            self.dry_run()
        else:
            self.num_gpus = self.check_gpu_availability(self.lab.profile.num_gpus)

            if self.lab.profile.max_gpus is not None:
                if self.num_gpus > self.lab.profile.max_gpus:
                    logger.info(
                        f"Maximum numbers of GPUs {self.num_gpus} being capped to {self.lab.profile.max_gpus} "
                        f"because of lab profile.")
                    self.num_gpus = self.lab.profile.max_gpus

            # No matter the number of GPUs, setup the node for training
            self.node_setup()

            if self.num_gpus == 0:
                self.gpu = None
            elif self.num_gpus == 1:
                self.gpu = 0
            else:
                self.train_distributed(self.num_gpus)

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

    def tuning_round(self) -> dict:
        logger.warning("tuning_round() not implemented in base Trainer.")
        import random
        loss = random.randint(0, 10)
        accuracy = random.randint(0, 100)
        val_loss = random.randint(0, 10)
        val_accuracy = random.randint(0, 100)

        return {"loss": loss, "accuracy": accuracy, "val_loss": val_loss, "val_accuracy": val_accuracy}

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

        duration = end_time - self.train_start_time

        model_config = self.model_config
        dataset_config = self.dataset_config

        # Add all the times.
        # TODO Switch to use the new training_output script
        self.results['times']['start_time'] = self.train_start_time.isoformat()
        self.results['times']['end_time'] = end_time.isoformat()
        self.results['times']['duration'] = duration.total_seconds()

        # Copy in relevant parts of our training options.
        self.results['options']['training_dataset_config_path'] = str(model_config.training_dataset_config_path)
        self.results['options']['model_architecture'] = model_config.model_architecture
        self.results['options']['epochs'] = model_config.epochs
        self.results['options']['batch_size'] = model_config.batch_size
        self.results['options']['seed'] = model_config.seed

        # This couples us to one dataset.
        self.results['options']['data_type'] = dataset_config.data_type

        # This should move into the training options.
        self.results['results']['model_name'] = self.model_manager.model_name

        self.results['format_version'] = TrainingOutput.FORMAT_VERSION

    def _serialize_results(self):

        if self.gpu:
            logger.info("Only the rank 0 process is responsible for writing the training results to file.")

        else:
            logger.info(f"Writing output file: {self.model_manager.get_training_out_file()}")
            jb_fs.save_json(self.results.to_json(), self.model_manager.get_training_out_file())


class EpochTrainer(Trainer):
    """
    This class encapsulates the process of training a supervised model epoch by epoch.
    """

    def __init__(self, **kwargs):
        """
        Construct an epoch trainer based on a model config and dataset config.
        :param model_config: A Juneberry ModelConfig object that describes how to construct and train the model.
        """
        # Set to true for dry run mode
        super().__init__(**kwargs)

        # A 1-based counter showing the current epoch. A value of zero means training has not begun.
        self.epoch = 0
        #
        # # TODO: New material
        self.max_epochs = None
        self.done = None
        self.training_iterable = None
        self.evaluation_iterable = None

    def inherit_from_sprout(self, sprout: TrainingSprout):
        super().inherit_from_sprout(sprout)

        self.max_epochs = self.model_config.epochs
        self.done = False if self.max_epochs > self.epoch else True

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

    def end_epoch(self) -> str:
        """
        Called at the end of epoch for model saving, external telemetry, etc.
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
            self.results['times']['epoch_duration_sec'].append(epoch_tracker.last_elapsed())

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
                for data, targets in data_iterable:
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
