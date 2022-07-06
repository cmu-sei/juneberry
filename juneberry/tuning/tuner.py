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
import logging
import os
from pathlib import Path
import sys

from ray import tune
from ray.tune.integration.torch import DistributedTrainableCreator, distributed_checkpoint_dir
import torch

from juneberry.config.model import ModelConfig
import juneberry.jb_logging as jb_logging
import juneberry.loader as jb_loader
from juneberry.trainer import Trainer
from juneberry.tuning.reporter import CustomReporter

logger = logging.getLogger(__name__)


class Tuner:
    """
    This class is responsible for tuning models in Juneberry.
    """

    def __init__(self):
        # This model config represents the base state of the model whose hyperparameters
        # are being adjusted during tuning.
        self.baseline_model_config = ModelConfig()

        # The TuningSprout stores information about the conditions involved in the current
        # tuning run. Properties like workspace directories, number of GPUs available, etc.
        self.tuning_sprout = None

        # Attributes derived from the tuning config.

        # This will be a dictionary that defines which hyperparameters will be adjusted,
        # as well as what values are possible for each adjustment.
        self.search_space = None

        # The search algo is responsible for suggesting hyperparameter configurations out of
        # the search space.
        self.search_algo = None

        # This scheduler is used during the tuning run to determine when to stop a trial
        # of hyperparameters early if they're not performing well.
        self.scheduler = None

        # Some more properties from the tuning config.
        self.trial_resources = None
        self.metric = None
        self.mode = None
        self.num_samples = None
        self.scope = None
        self.checkpoint_interval = None

        # Attribute to capture the best tuning result.
        self.best_result = None

    def _build_tuning_components(self) -> None:
        """
        This method is responsible for assembling various tuning components.
        :return:
        """
        # Extract the search_space from the tuning config.
        if self.tuning_sprout.tuning_config.search_space is not None:
            self._build_search_space()

        # Exit if the tuning config does not define a search space.
        else:
            logger.error(f"The tuning config {self.tuning_sprout.tuning_config_str} does not "
                         f"define a search space. The Tuner cannot determine which hyperparameters "
                         f"to tune. Exiting.")
            sys.exit(-1)

        # Extract the scheduler from the tuning config and then build the desired scheduler.
        if self.tuning_sprout.tuning_config.scheduler is not None:
            self._build_scheduler()
        else:
            logger.warning(f"The tuning config {self.tuning_sprout.tuning_config_str} does not "
                           f"define a scheduler. Ray Tune uses "
                           f"'ray.tune.schedulers.FIFOScheduler' by default.")

        # Extract the search algorithm for the search space and then build it.
        if self.tuning_sprout.tuning_config.search_algorithm is not None:
            self._build_search_algo()
        else:
            logger.warning(f"The tuning config {self.tuning_sprout.tuning_config_str} does not "
                           f"define a search algorithm. Ray Tune uses "
                           f"'ray.tune.suggest.basic_variant.BasicVariantGenerator' by default.")

    def _build_search_space(self) -> None:
        """
        This method is responsible for constructing the search space for the Tuner.
        :return: Nothing.
        """
        search_space = {}

        # Retrieve the desired search space from the tuning config.
        search_space_dict = self.tuning_sprout.tuning_config.search_space

        # For each item listed in the tuning config's search space, construct the desired
        # sampling function.
        for key, plugin in search_space_dict.items():
            search_space[key] = jb_loader.construct_instance(plugin.fqcn, plugin.kwargs)

        # Once the entire search space has been assembled, assign it to the Tuner.
        self.search_space = search_space

    def _build_scheduler(self) -> None:
        """
        This method is responsible for constructing the scheduler for the Tuner.
        :return: Nothing.
        """
        # Retrieve the desired scheduler from the tuning config.
        scheduler_dict = self.tuning_sprout.tuning_config.scheduler
        logger.info(f"Constructing tuning scheduler using fqcn: {scheduler_dict.fqcn}")

        # All schedulers in Ray Tune use 'self.metric' and 'self.mode' to make decisions about
        # terminating bad trials, altering parameters in a running trial, etc. However,
        # tune.run() receives both of these properties directly and will complain when they are
        # also provided to the scheduler.
        if "metric" in scheduler_dict.kwargs:
            logger.warning(f"The scheduler does not need a 'metric' parameter since the 'metric' arg is "
                           f"passed to tune.run(). Remove the 'metric' kwarg from the 'scheduler' section "
                           f"in '{self.tuning_sprout.tuning_config_str}' to eliminate this warning.")
            scheduler_dict.kwargs.pop("metric")

        if "mode" in scheduler_dict.kwargs:
            logger.warning(f"The scheduler does not need a 'mode' parameter since the 'mode' arg is "
                           f"passed to tune.run(). Remove the 'mode' kwarg from the 'scheduler' section "
                           f"in '{self.tuning_sprout.tuning_config_str}' to eliminate this warning.")
            scheduler_dict.kwargs.pop("mode")

        # Construct the scheduler and assign it to the scheduler attribute.
        logger.info(f"  kwargs for tuning scheduler: {scheduler_dict.kwargs}")
        self.scheduler = jb_loader.construct_instance(scheduler_dict.fqcn, scheduler_dict.kwargs)
        logger.info(f"  Tuning scheduler built.")

    def _build_search_algo(self) -> None:
        """
        This method is responsible for constructing the search algorithm for the Tuner.
        :return: Nothing.
        """
        # Retrieve the desired search algorithm from the tuning config.
        algo_dict = self.tuning_sprout.tuning_config.search_algorithm
        logger.info(f"Constructing tuning search_algo using fqcn: {algo_dict.fqcn}")

        # Construct the search algorithm and assign it to the search_algo attribute.
        logger.info(f"  kwargs for tuning search_algo: {algo_dict.kwargs}")
        self.search_algo = jb_loader.construct_instance(algo_dict.fqcn, algo_dict.kwargs)
        logger.info(f"  Tuning search_algo built.")

    def _tuning_attempt(self, config: dict, checkpoint_dir: str = None) -> None:
        """
        This method represents the 'Trainable' that Ray Tune will attempt to optimize turing the tuning run.
        :param config: A dictionary containing the chosen hyperparameter values for the current tuning trial.
        :param checkpoint_dir: A string indicating the name of the model checkpoint directory to use.
        :return: Nothing.
        """
        # Ray Tune runs this function on a separate thread in a Ray actor process.

        # Set up the root juneberry logger for this thread.
        jb_logging.setup_logger(self.tuning_sprout.log_file, log_prefix=self.tuning_sprout.log_prefix,
                                log_to_console=not self.tuning_sprout.silent, level=self.tuning_sprout.log_level,
                                name="juneberry")

        # This will substitute the current set of hyperparameters for the trial into the baseline config.
        # trial_model_config is a ModelConfig.
        trial_model_config = self.baseline_model_config.adjust_attributes(config)

        # Once the trial model config has been created, adjust the model config in the TuningSprout and
        # use the Sprout to fetch a Juneberry Trainer.
        self.tuning_sprout.set_model_config(trial_model_config)
        trainer = self.tuning_sprout.get_trainer()

        # Initialize the current epoch to zero.
        cur_epoch = 0

        # "Many Tune features rely on checkpointing, including certain Trial Schedulers..."
        # Retrieve the checkpoint if one was provided.
        if checkpoint_dir:
            logger.info(f"Loading from checkpoint. Checkpoint dir - {checkpoint_dir}")
            checkpoint_path = os.path.join(checkpoint_dir, "checkpoint")
            checkpoint = torch.load(checkpoint_path)
            trainer.model.load_state_dict(checkpoint["model_state_dict"])
            cur_epoch = checkpoint["cur_epoch"]

        # Perform any tuning setup steps defined in the Trainer.
        trainer.tuning_setup()

        # Continue fetching metrics until the maximum number of training epochs has
        # been achieved (unless the trial gets stopped early by the Tuner).
        while cur_epoch < trial_model_config.epochs:
            # Fetch the latest metrics.
            returned_metrics = trainer.tune()

            # Determine if model checkpointing was desired.
            if self.checkpoint_interval:
                # This ensures a model checkpoint will be saved every after every N epochs, where
                # N is equal to the checkpoint_interval.
                if not (cur_epoch + 1) % self.checkpoint_interval:
                    # Distributed training requires a unique type of checkpoint directory.
                    if self.trial_resources.gpu > 1:
                        # TODO: Shouldn't reach here until distributed tuning works.
                        with distributed_checkpoint_dir(step=cur_epoch) as checkpoint_dir:
                            self._save_model_checkpoint(trainer, checkpoint_dir)
                    # Non-distributed training requires the more typical checkpoint directory.
                    else:
                        with tune.checkpoint_dir(step=cur_epoch) as checkpoint_dir:
                            self._save_model_checkpoint(trainer, checkpoint_dir)

            # Retrieve the latest metrics, report the intermediate value, and increment the epoch number.
            metrics = next(returned_metrics)
            yield metrics
            cur_epoch += 1

    @staticmethod
    def _save_model_checkpoint(trainer: Trainer, checkpoint_dir: str) -> None:
        """
        This method is responsible for saving a model checkpoint to a checkpoint directory.
        :param trainer: A Juneberry Trainer object containing the trained model to be saved.
        :param checkpoint_dir: A string indicating where to save the model checkpoint.
        :return: Nothing.
        """
        path = os.path.join(checkpoint_dir, "checkpoint")
        logger.info(f"Saving checkpoint to {path}")
        torch.save((trainer.model.state_dict(), trainer.optimizer.state_dict()), path)

    def tune(self) -> None:
        """
        This method performs a tuning run based on the conditions set by all of the various
        attributes of the Tuner.
        :return: Nothing.
        """
        # Construct the various components required for tuning.
        self._build_tuning_components()

        # Determine which Trainable to use for the tuning run. The main decision in this step is
        # to figure out if a distributed trainable is needed.
        trainable, trial_resources = self._determine_trainable()

        logger.info(f"Starting the tuning run.")
        # Perform the tuning run.
        result = tune.run(
            trainable,
            resources_per_trial=trial_resources,
            config=self.search_space,
            search_alg=self.search_algo,
            metric=self.metric,
            mode=self.mode,
            num_samples=self.num_samples,
            scheduler=self.scheduler,
            local_dir=str(self.tuning_sprout.model_manager.get_tuning_dir()),
            progress_reporter=CustomReporter()
        )
        logger.info(f"The tuning run is complete. Storing the best result.")
        #
        # Once tuning is complete, store the best result.
        self.best_result = result.get_best_trial(self.metric, self.mode, self.scope)

        # Perform any final tuning steps, such as indicating the "best result".
        self.finish_tuning()

    def _determine_trainable(self) -> tuple:
        """
        The purpose of this method is to make adjustments to the Trainable if distributed
        training has been requested.
        :return: A tuple with the Trainable, and the trial_resources
        """
        # Make the necessary adjustments to support tuning a model that requires distributed training.
        if self.trial_resources.gpu > 1:
            # TODO: Implement distributed tuning.
            # For now, log a warning that distributed tuning is not supported and lower the number of
            # GPU resources to 1.
            logger.warning(f"Distributed tuning not implemented yet. Setting trial GPU resources "
                           f"to 1.")
            trainable = self._tuning_attempt
            self.trial_resources.gpu = 1
            trial_resources = self.trial_resources

            # This is what would probably be needed to get distributed tuning working:
            # trainable = DistributedTrainableCreator(
            #     self._tuning_attempt,
            #     num_workers=1,
            #     num_cpus_per_worker=self.trial_resources.cpu,
            #     num_gpus_per_worker=self.trial_resources.gpu
            # )
            # tune.run complains if trial resources are still set if they've been defined
            # inside the DistributedTrainableCreator
            # trial_resources = None
        # Otherwise, use the standard _tuning_attempt function as the Trainable and don't make any
        # adjustments to the requested trial_resources.
        else:
            trainable = self._tuning_attempt
            trial_resources = self.trial_resources

        return trainable, trial_resources

    def finish_tuning(self):
        # TODO: Once a tuning run is complete, there are various things that can be done with
        #  the best tuning result.
        #  Maybe save the best model config?
        logger.info(f"Best trial config: {self.best_result.config}")
        logger.info(f"Best trial final '{self.metric}': {self.best_result.last_result[self.metric]}")

        # Move the Juneberry tuning log into the tuning directory for this run.
        new_log_path = Path(self.best_result.local_dir) / "log.txt"
        self.tuning_sprout.model_manager.get_tuning_log().rename(new_log_path)
