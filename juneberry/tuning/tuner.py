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

from ray import tune

from juneberry.config.model import ModelConfig
from juneberry.config.plugin import Plugin
from juneberry.config.tuning import TuningConfig
from juneberry.lab import Lab
import juneberry.loader as jb_loader
from juneberry.scripting.sprout import TuningSprout

logger = logging.getLogger(__name__)


class Tuner:

    def __init__(self, **kwargs):
        self.model_name = None
        self.model_manager = None
        self.baseline_model_config = ModelConfig()

        self.tuning_config_str = None
        self.tuning_config = TuningConfig()

        self.lab = Lab()

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
        self.trial_resources = self.tuning_config.trial_resources
        self.metric = None
        self.mode = None
        self.num_samples = self.tuning_config.sample_quantity
        self.scope = None
        self.checkpoint_interval = None

        # Attribute to capture the best tuning result.
        self.best_result = None

        # Methods for setting attributes.
        self._build_tuning_components()

        self.sprout = None

    def inherit_from_sprout(self, sprout: TuningSprout):
        self.sprout = sprout

        self.model_name = sprout.model_name
        self.model_manager = sprout.model_manager
        self.lab = sprout.lab
        self.tuning_config_str = sprout.tuning_config_str
        self.tuning_config = sprout.tuning_config

        self.trial_resources = self.tuning_config.trial_resources
        self.metric = self.tuning_config.tuning_parameters.metric
        self.mode = self.tuning_config.tuning_parameters.mode
        self.num_samples = self.tuning_config.sample_quantity
        self.scope = self.tuning_config.tuning_parameters.scope
        self.checkpoint_interval = self.tuning_config.tuning_parameters.checkpoint_interval

        self.baseline_model_config = self.lab.load_model_config(self.model_name)

        # Methods for setting attributes.
        self._build_tuning_components()

    def _build_tuning_components(self):
        """
        This appears to be working properly.
        :return:
        """
        # Extract the search_space from the tuning config.
        if self.tuning_config.search_space is not None:
            self._build_search_space(self.tuning_config.search_space)

        # Extract the scheduler from the tuning config and then build the desired scheduler.
        if self.tuning_config.scheduler is not None:
            self._build_scheduler(self.tuning_config.scheduler)

        # Extract the search algorithm for the search space and then build it.
        if self.tuning_config.search_algorithm is not None:
            self._build_search_algo(self.tuning_config.search_algorithm)

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
                           f"in '{self.tuning_config_name}' to eliminate this warning.")
            scheduler_dict.kwargs.pop("metric")

        if "mode" in scheduler_dict.kwargs:
            logger.warning(f"The scheduler does not need a 'mode' parameter since the 'mode' arg is "
                           f"passed to tune.run(). Remove the 'mode' kwarg from the 'scheduler' section "
                           f"in '{self.tuning_config_name}' to eliminate this warning.")
            scheduler_dict.kwargs.pop("mode")

        logger.info(f"  kwargs for tuning scheduler: {scheduler_dict.kwargs}")
        self.scheduler = jb_loader.construct_instance(scheduler_dict.fqcn, scheduler_dict.kwargs)
        logger.info(f"  Tuning scheduler built!")

    def _build_search_algo(self, algo_dict: Plugin):
        logger.info(f"Constructing tuning search_algo using fqcn: {algo_dict.fqcn}")
        logger.info(f"  kwargs for tuning search_algo: {algo_dict.kwargs}")
        self.search_algo = jb_loader.construct_instance(algo_dict.fqcn, algo_dict.kwargs)
        logger.info(f"  Tuning search_algo built!")

    def _train_fn(self, config, checkpoint_dir=None):
        # TODO: Lots of work here...the real "meat" of the problem. May need to refactor various Juneberry
        #  components.
        # Ray Tune runs this function on a separate thread in a Ray actor process.

        # This will substitute the current set of hyperparameters for the trial into the baseline config.
        """ The model config substitution process (line below) appears to be working as expected."""
        trial_model_config = self.baseline_model_config.adjust_attributes(config)
        # trial_model_config is a ModelConfig

        self.sprout.set_model_config(trial_model_config)
        trainer = self.sprout.build_trainer_from_model_config()

        # "Many Tune features rely on checkpointing, including certain Trial Schedulers..."
        if checkpoint_dir:
            pass
            # Load from checkpoint.
            # checkpoint_path = self.model_manager.get_tuning_checkpoint_dir() / checkpoint_name
            # checkpoint = torch.load(checkpoint_path)
            # model.load_state_dict(checkpoint["model_state_dict"])
            # cur_epoch = checkpoint["cur_epoch"]

        cur_epoch = 0

        trainer.tuning_setup()

        while cur_epoch < trial_model_config.epochs:
            latest_metrics = trainer.tune()
            # latest_metrics = trainer.tuning_round()
            # latest_metrics = trainer.tuning_generator()
            # Assuming latest_metrics is a dictionary, use tune.report() like this:
            # tune.report(**latest_metrics)
            thing = next(latest_metrics)
            # print(f"thing{cur_epoch} is {thing}")
            yield thing
            cur_epoch += 1

        # Would something like this make sense? Would require Trainer refactoring, especially for the
        # metrics = trainer._train_one_epoch() (or equivalent) line.
        # while cur_epoch < max_epochs:
        #   metrics = trainer._train_one_epoch()
        #   if cur_epoch % self.checkpoint_interval == 0:
        #       with tune.checkpoint_dir(step=cur_epoch) as checkpoint_dir:
        #           path = Path(checkpoint_dir) / "checkpoint"
        #           torch.save({
        #               "cur_epoch": cur_epoch,
        #               "model_state_dict": model.state_dict(),
        #               "metric": metrics.metric
        #           }, path)
        #   tune.report(metric=metrics.metric)

    def tune(self):
        # Perform the tuning run.
        result = tune.run(
            self._train_fn,
            resources_per_trial=self.trial_resources,
            config=self.search_space,
            search_alg=self.search_algo,
            metric=self.metric,
            mode=self.mode,
            num_samples=self.num_samples,
            scheduler=self.scheduler,
            local_dir=str(self.model_manager.get_tuning_dir())
        )
        #
        # Once the tuning is complete, store the best result.
        self.best_result = result.get_best_trial(self.metric, self.mode, self.scope)

    # TODO: Better name for this method.
    def process_best_result(self):
        # TODO: Once a tuning run is complete, there are various things that can be done with
        #  the best tuning result. This is where we'd do something with self.best_result.
        logger.info(f"Best trial config: {self.best_result.config}")
        logger.info(f"Best trial final '{self.metric}': {self.best_result.last_result[self.metric]}")


    