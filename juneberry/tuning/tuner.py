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

from ray import tune
from ray.tune.integration.torch import DistributedTrainableCreator, distributed_checkpoint_dir
import torch

from juneberry.config.model import ModelConfig
from juneberry.config.plugin import Plugin
import juneberry.jb_logging as jb_logging
import juneberry.loader as jb_loader
from juneberry.tuning.reporter import CustomReporter

logger = logging.getLogger(__name__)


class Tuner:

    def __init__(self):
        self.baseline_model_config = ModelConfig()
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

    def _build_tuning_components(self):
        """
        This appears to be working properly.
        :return:
        """
        # Extract the search_space from the tuning config.
        if self.tuning_sprout.tuning_config.search_space is not None:
            self._build_search_space(self.tuning_sprout.tuning_config.search_space)

        # Extract the scheduler from the tuning config and then build the desired scheduler.
        if self.tuning_sprout.tuning_config.scheduler is not None:
            self._build_scheduler(self.tuning_sprout.tuning_config.scheduler)

        # Extract the search algorithm for the search space and then build it.
        if self.tuning_sprout.tuning_config.search_algorithm is not None:
            self._build_search_algo(self.tuning_sprout.tuning_config.search_algorithm)

    def _build_search_space(self, search_space_dict: dict):
        """
        This appears to be working properly.
        :param search_space_dict:
        :return:
        """
        search_space = {}

        for key, plugin in search_space_dict.items():
            search_space[key] = jb_loader.construct_instance(plugin.fqcn, plugin.kwargs)

        self.search_space = search_space

    def _build_scheduler(self, scheduler_dict: Plugin):
        """
        This appears to be working properly.
        :param scheduler_dict:
        :return:
        """

        # All schedulers in Ray Tune will need the 'self.metric' and 'self.mode' in order to make
        # decisions about terminating bad trials, altering parameters in a running trial, etc.
        logger.info(f"Constructing tuning scheduler using fqcn: {scheduler_dict.fqcn}")

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

        logger.info(f"  kwargs for tuning scheduler: {scheduler_dict.kwargs}")
        self.scheduler = jb_loader.construct_instance(scheduler_dict.fqcn, scheduler_dict.kwargs)
        logger.info(f"  Tuning scheduler built!")

    def _build_search_algo(self, algo_dict: Plugin):
        logger.info(f"Constructing tuning search_algo using fqcn: {algo_dict.fqcn}")
        logger.info(f"  kwargs for tuning search_algo: {algo_dict.kwargs}")
        self.search_algo = jb_loader.construct_instance(algo_dict.fqcn, algo_dict.kwargs)
        logger.info(f"  Tuning search_algo built!")

    def _tuning_attempt(self, config, checkpoint_dir=None):
        # Ray Tune runs this function on a separate thread in a Ray actor process.

        jb_logging.setup_logger(self.tuning_sprout.log_file, log_prefix=self.tuning_sprout.log_prefix,
                                log_to_console=not self.tuning_sprout.silent, level=self.tuning_sprout.log_level,
                                name="juneberry")

        # This will substitute the current set of hyperparameters for the trial into the baseline config.
        # trial_model_config is a ModelConfig.
        trial_model_config = self.baseline_model_config.adjust_attributes(config)

        self.tuning_sprout.set_model_config(trial_model_config)
        trainer = self.tuning_sprout.get_trainer()

        # trainer.tuning_setup()
        cur_epoch = 0

        # "Many Tune features rely on checkpointing, including certain Trial Schedulers..."
        # Retrieve the checkpoint if one was provided.
        if checkpoint_dir:
            logger.info(f"Loading from checkpoint. Checkpoint dir - {checkpoint_dir}")
            checkpoint_path = os.path.join(checkpoint_dir, "checkpoint")
            checkpoint = torch.load(checkpoint_path)
            trainer.model.load_state_dict(checkpoint["model_state_dict"])
            cur_epoch = checkpoint["cur_epoch"]

        trainer.tuning_setup()

        while cur_epoch < trial_model_config.epochs:
            returned_metrics = trainer.tune()

            # Save a checkpoint
            if self.checkpoint_interval:
                if not (cur_epoch + 1) % self.checkpoint_interval:

                    if self.trial_resources.gpu > 1:
                        # TODO: Shouldn't reach here until distributed tuning works.
                        with distributed_checkpoint_dir(step=cur_epoch) as checkpoint_dir:
                            self._save_model_checkpoint(trainer, checkpoint_dir)
                    else:
                        with tune.checkpoint_dir(step=cur_epoch) as checkpoint_dir:
                            self._save_model_checkpoint(trainer, checkpoint_dir)

            # Retrieve the latest metrics, report the intermediate value, and increment the epoch number.
            metrics = next(returned_metrics)
            yield metrics
            cur_epoch += 1

    @staticmethod
    def _save_model_checkpoint(trainer, checkpoint_dir):
        path = os.path.join(checkpoint_dir, "checkpoint")
        logger.info(f"Saving checkpoint to {path}")
        torch.save((trainer.model.state_dict(), trainer.optimizer.state_dict()), path)

    def tune(self):
        # Methods for setting attributes.
        self._build_tuning_components()

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
        # Once the tuning is complete, store the best result.
        self.best_result = result.get_best_trial(self.metric, self.mode, self.scope)

    def _determine_trainable(self):
        if self.trial_resources.gpu > 1:
            logger.warning(f"Distributed tuning not implemented yet. Setting trial GPU resources "
                           f"to 1.")
            trainable = self._tuning_attempt
            self.trial_resources.gpu = 1
            trial_resources = self.trial_resources

            # TODO: Implement distributed tuning.
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
        else:
            trainable = self._tuning_attempt
            trial_resources = self.trial_resources

        return trainable, trial_resources

    # TODO: Better name for this method.
    def process_best_result(self):
        # TODO: Once a tuning run is complete, there are various things that can be done with
        #  the best tuning result. This is where we'd do something with self.best_result.
        #  Maybe save the best model config?
        logger.info(f"Best trial config: {self.best_result.config}")
        logger.info(f"Best trial final '{self.metric}': {self.best_result.last_result[self.metric]}")

        # Move the Juneberry tuning log into the tuning directory for this run.
        new_log_path = Path(self.best_result.local_dir) / "log.txt"
        self.tuning_sprout.model_manager.get_tuning_log().rename(new_log_path)
