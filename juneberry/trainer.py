#! /usr/bin/env python3

"""
This base class provides a skeleton loop for supervised training and evaluating classifiers.
It makes the following assumptions:
- Data comes with targets/labels.

It provides the following features:
- A variety of extension points to customize the process.
- Data can be provided in batches.
- Training can be terminated early based on extensible stopping criteria.
- Each stage is independently timed.
"""

# ==========================================================================================================================================================
#  Copyright 2021 Carnegie Mellon University.
#
#  NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS"
#  BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER
#  INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED
#  FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM
#  FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT. Released under a BSD (SEI)-style license, please see license.txt
#  or contact permission@sei.cmu.edu for full terms.
#
#  [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see
#  Copyright notice for non-US Government use and distribution.
#
#  This Software includes and/or makes use of the following Third-Party Software subject to its own license:
#  1. Pytorch (https://github.com/pytorch/pytorch/blob/master/LICENSE) Copyright 2016 facebook, inc..
#  2. NumPY (https://github.com/numpy/numpy/blob/master/LICENSE.txt) Copyright 2020 Numpy developers.
#  3. Matplotlib (https://matplotlib.org/3.1.1/users/license.html) Copyright 2013 Matplotlib Development Team.
#  4. pillow (https://github.com/python-pillow/Pillow/blob/master/LICENSE) Copyright 2020 Alex Clark and contributors.
#  5. SKlearn (https://github.com/scikit-learn/sklearn-docbuilder/blob/master/LICENSE) Copyright 2013 scikit-learn
#      developers.
#  6. torchsummary (https://github.com/TylerYep/torch-summary/blob/master/LICENSE) Copyright 2020 Tyler Yep.
#  7. adversarial robust toolbox (https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/LICENSE)
#      Copyright 2018 the adversarial robustness toolbox authors.
#  8. pytest (https://docs.pytest.org/en/stable/license.html) Copyright 2020 Holger Krekel and others.
#  9. pylint (https://github.com/PyCQA/pylint/blob/master/COPYING) Copyright 1991 Free Software Foundation, Inc..
#  10. python (https://docs.python.org/3/license.html#psf-license) Copyright 2001 python software foundation.
#
#  DM20-1149
#
# ==========================================================================================================================================================

import datetime
import logging

from juneberry.timing import Berryometer


class EpochTrainer:
    """
    This class encapsulates the process of training a supervised model epoch by epoch.
    """

    def __init__(self, training_config, data_set_config, *, dry_run=False):
        """
        Construct an epoch trainer based on a training config and data set config.
        :param training_config: The configuration that describes how to construct and train the model.
        :param data_set_config: The configuration that describes the data set used during training and validation.
        """
        # Set to true for dry run mode
        self.do_dry_run = dry_run

        # The training and data set configurations.
        self.training_config = training_config
        self.data_set_config = data_set_config

        # Is the training done? Set to True to exit training loop.
        self.done = False

        # Maximum number of epochs.  We may stop earlier if we meet other criteria
        self.max_epochs = training_config.epochs

        # A 1-based counter showing the current epoch. A value of zero means training has not begun.
        self.epoch = 0

        # Iterables that provides batches (lists) of pairs of (data, targets) for training and evaluation.
        # These must be initialized during setup.
        self.training_iterable = None
        self.evaluation_iterable = None

        # A Berryometer object we use to track the times of phases. Can be used by the
        # subclasses for tracking the time of particular tasks.
        self.timer = None

        # Logger that can be used for training.
        self.logger = logging.getLogger()

        # When did the training start
        self.train_start_time = None

    def train_model(self):
        # Capture the time training started
        self.train_start_time = datetime.datetime.now().replace(microsecond=0)

        self.timer = Berryometer()

        # Read the configs and setup the models, loaders, functions, etc.
        self.logger.info("Performing setup")
        with self.timer("setup"):
            self.setup()

        # If we have done setup, we can now dump the dry run data
        if self.do_dry_run:
            self.logger.info("Performing dry run")
            self.dry_run()
            return

        # For each epoch we need to do our basic training loops
        self.logger.info(f"Starting Training...")
        while not self.done:
            self._train_one_epoch()

        # Finalize the model, results and everything else
        with self.timer("finalize_results"):
            self.logger.info(f"Finalizing results")
            self.finalize_results()

        self.close()

    # -----------------------------------------------
    #  _____     _                 _              ______     _       _
    # |  ___|   | |               (_)             | ___ \   (_)     | |
    # | |____  _| |_ ___ _ __  ___ _  ___  _ __   | |_/ /__  _ _ __ | |_ ___
    # |  __\ \/ / __/ _ \ '_ \/ __| |/ _ \| '_ \  |  __/ _ \| | '_ \| __/ __|
    # | |___>  <| ||  __/ | | \__ \ | (_) | | | | | | | (_) | | | | | |_\__ \
    # \____/_/\_\\__\___|_| |_|___/_|\___/|_| |_| \_|  \___/|_|_| |_|\__|___/

    def setup(self) -> None:
        """ Setup all data loaders, model and supporting functions."""
        pass

    def dry_run(self) -> None:
        """ Output all the dry run information about this model."""
        pass

    def start_epoch_phase(self, train: bool):
        """
        Called at the beginning of the phase of an epoch.
        This should return a metrics object that will be passed to each batch
        for producing batch metrics. This object will also be passed to the
        summarization routine to summarize the phase.
        :param train: True if training, else evaluation
        :return: The metrics object
        """
        return {}

    def process_batch(self, train: bool, data, targets):
        """
        Process the batch of data and targets, returning a results object to be
        passed to update metrics.
        :param train: True if training, else evaluation
        :param data: The data to process.
        :param targets: The associated targets.
        :return: A results object for updating the metrics.
        """
        return None

    def update_metrics(self, train: bool, metrics, results) -> None:
        """
        Called to update the metrics with the specific batch results.
        :param train: True if training, else evaluating.
        :param metrics: The metrics to roll into results.
        :param results: The results.
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
        Summarize the metrics captured on a per batch basis to the internal model
        for later use. Printing, showing as appropriate.
        :param train: True if training, else evaluating.
        :param metrics: Summarize the metrics across the batch.
        :result: A message to produce at the end of each epoch
        """
        # TODO: Should it keep/retain the final results/history?
        pass

    def end_epoch(self, elapsed_secs: float) -> str:
        """
        Called at the end of epoch for model saving, external telemetry, etc.
        :param elapsed_secs: Number of seconds elapsed during the epoch
        """
        return ""

    def finalize_results(self) -> None:
        """
        Save results to appropriate files, models, etc.
        """
        pass

    def close(self) -> None:
        """
        Called to finalize and close all resources.
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
            self._process_one_iteration(True, self.training_iterable)

            # Process all the data from the data loader
            self._process_one_iteration(False, self.evaluation_iterable)

        # End of epoch, check for acceptance model checkpointing, etc.
        # This is outside of the epoch timer so the end has access to the time.
        with self.timer("end_epoch"):
            epoch_tracker = self.timer('epoch')
            msg = self.end_epoch(epoch_tracker.last_elapsed())

        # Deal with timing and logs
        epoch_tracker = self.timer('epoch')
        elapsed = epoch_tracker.last_elapsed()

        remaining_seconds = (self.max_epochs - self.epoch) * epoch_tracker.weighted_mean()
        eta = datetime.datetime.now() + datetime.timedelta(seconds=remaining_seconds)

        # Calculate the remaining time as if we get to max epoch.
        hours, remainder = divmod(remaining_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        logging.info(f"{self.epoch}/{self.max_epochs}, "
                     f"time_sec: {elapsed:.3f}, eta: {eta.strftime('%H:%M:%S')}, "
                     f"remaining: {int(hours):d}:{int(minutes):02d}:{int(seconds):02d}, "
                     f"{msg}")

    def _process_one_iteration(self, train: bool, data_iterable):
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

        # Now that we have finished all the batches, summarize the metrics.
        with self.timer("summarize_train"):
            self.summarize_metrics(train, metrics)


def main():
    print("Nothing to see here.")


if __name__ == "__main__":
    main()
