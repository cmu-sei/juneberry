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
import juneberry.filesystem as jb_fs
import juneberry.loader as jb_loader

logger = logging.getLogger(__name__)


class Tuner:

    def __init__(self, model_name: str, tuning_config_name: str):
        self.model_name = model_name
        self.model_manager = jb_fs.ModelManager(self.model_name)
        self.baseline_model_config = ModelConfig.load(self.model_name)
        self.trial_model_config = None

        self.tuning_config_name = tuning_config_name
        self.tuning_config = TuningConfig.load(self.tuning_config_name)

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
        self.checkpoint_interval = 1

        # Attribute to capture the best tuning result.
        self.best_result = None

        # Methods for setting attributes.
        self._setup_tuning_parameters()

    def _setup_tuning_parameters(self):
        pass

        # TODO: This is responsible for extracting the tuning parameters from the tuning config.

        #    TODO: Extract the search_space from the tuning config.

        #    TODO: Extract the scheduler from the tuning config and then build a scheduler.
        self._build_scheduler(self.tuning_config.scheduler)

        #    TODO: Extract the trial resources from the tuning config. Set reasonable defaults
        #     if they're not present in the tuning config. Trial resources are the machine resources
        #     to allocate per trial. Defaults to 1 CPU and 0 GPUs.

        #    TODO: Extract the metric (str), mode (str), and scope (str). Set reasonable defaults
        #     if they're not present in the tuning config.
        #       metric - Metric to optimize. Should be reported with tune.report()
        #       mode - Either 'min' or 'max'. Determines whether the objective is to 'min' or 'max'
        #         the metric attribute.
        #       scope - One of [all, last, avg, last-5-avg, last-10-avg]. Indicates how to calculate
        #         a trial's score for the metric. 'last' looks at the final reported metric value for
        #         the trial. 'all' would look at the metric value reported after every step (epoch?)
        #         and choose the min/max out of all those values. avgs are averages across N steps.

        #    TODO: Extract the search algorithm for the search space. Set a reasonable default if
        #     it's not present in the tuning config.
        self._build_search_algo(self.tuning_config.search_algorithm)

    def _build_scheduler(self, scheduler_dict: Plugin):
        # TODO: This is responsible for constructing the search algorithm to be used during the tuning run.
        #  Ultimately this will looks something like:

        # All schedulers in Ray Tune will need the 'self.metric' and 'self.mode' in order to make
        # decisions about terminating bad trials, altering parameters in a running trial, etc.

        self.scheduler = jb_loader.construct_instance(scheduler_dict.fqcn, scheduler_dict.kwargs)

    def _build_search_algo(self, algo_dict: Plugin):
        # TODO: This is responsible for constructing the search algorithm to be used during the tuning run.
        #  Ultimately this will looks something like:
        self.search_algo = jb_loader.construct_instance(algo_dict.fqcn, algo_dict.kwargs)

    def _train_fn(self, config, checkpoint_dir=None):
        # Ray Tune runs this function on a separate thread in a Ray actor process.

        # This will substitute the current set of hyperparameters for the trial into the baseline config.
        self.trial_model_config = self.baseline_model_config.adjust_attributes(config)

        # Now that we have a new model config, eventually we want to get to a spot where we can do a
        # trainer.train_model()

        # This function either needs to report the target metric (self.metric) after every unit of time
        # (epoch?) or return the metric value at the end of the function. I think if the scheduler is
        # going to be important to us for enforcing early stopping of a bad trial, then the best option
        # is to report the metric on a per-epoch basis. You can either report the metric to tune via
        # tune.report() or with a Python yield statement.

        # cur_epoch = 0

        # "Many Tune features rely on checkpointing, including certain Trial Schedulers..."
        if checkpoint_dir:
            pass
            # Load from checkpoint.
            # checkpoint_path = self.model_manager.get_tuning_checkpoint_dir() / checkpoint_name
            # checkpoint = torch.load(checkpoint_path)
            # model.load_state_dict(checkpoint["model_state_dict"])
            # cur_epoch = checkpoint["cur_epoch"]

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
        pass

        # TODO: Perform the tuning run.

        # result = tune.run(
        #     tune_with_parameters(self._train_fn),
        #     resources_per_trial=self.trial_resources,
        #     config=self.search_space,
        #     search_alg=self.search_algo,
        #     metric=self.metric,
        #     mode=self.mode,
        #     num_samples=self.num_samples,
        #     scheduler=self.scheduler
        # )
        #
        # TODO: Once the tuning is complete, store the best result.
        # self.best_result = result.get_best_trial(self.metric, self.mode, self.scope)

    # TODO: Better name for this method.
    def process_best_result(self):
        pass

        # TODO: Once a tuning run is complete, there are various things that can be done with
        #  the best tuning result. This is where we'd do something with self.best_result.
    