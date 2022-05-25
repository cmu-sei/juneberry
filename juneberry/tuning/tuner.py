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

from juneberry.config.model import ModelConfig
from juneberry.config.plugin import Plugin
from juneberry.config.tuning import TuningConfig
import juneberry.loader as jb_loader

logger = logging.getLogger(__name__)


class Tuner:

    def __init__(self, model_name: str, tuning_config_name: str):
        self.model_name = model_name
        self.model_config = ModelConfig.load(self.model_name)

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

        # Attributes derived from the model config.

        # Attributes related to model training.
        self.training_data = None
        self.validation_data = None
        self.train_fn = None

        # Attributes related to model evaluation.
        self.evaluation_data = None
        self.eval_fn = None

        # Attribute to capture the best tuning result.
        self.best_result = None

        # Methods for setting attributes.
        self.setup_tuning_parameters()
        self.setup_model()
        self.setup_data()
        self.setup_training()
        self.setup_evaluation()

    def setup_tuning_parameters(self):
        pass

        # TODO: This is responsible for extracting the tuning parameters from the tuning config.

        #    TODO: Extract the search_space from the tuning config.

        #    TODO: Extract the scheduler from the tuning config and then build a scheduler.

        #    TODO: Extract the trial resources from the tuning config. Set reasonable defaults
        #     if they're not present in the tuning config. Trial resources are things like # CPUs,
        #     # GPUs, etc.

        #    TODO: Extract the metric (str), mode (str), and scope (str). Set reasonable defaults
        #     if they're not present in the tuning config.

        #    TODO: Extract the search algorithm for the search space. Set a reasonable default if
        #     it's not present in the tuning config.
        self.build_search_algo(self.tuning_config.search_algorithm)

    def build_search_algo(self, algo_dict: Plugin):
        # TODO: This is responsible for constructing the search algorithm to be used during the tuning run.
        #  Ultimately this will looks something like:
        self.search_algo = jb_loader.construct_instance(algo_dict.fqcn, algo_dict.kwargs)

    def setup_model(self):
        pass
        # TODO: This is responsible for setting up the model to be tuned. Start with reading the model config.
        #  This may be very closely coupled to the 'setup_training' method below. It may be possible to rely
        #  on most trainer's 'setup_model' method to a certain extent.
        #
        #    TODO: The model architecture must also be set up. It may have tunable properties, so this should
        #     happen after self.search_space is known.

    def setup_data(self):
        pass
        # TODO: This is responsible for setting up the training_data, validation_data, and evaluation_data.

    def setup_training(self):
        pass
        # TODO: This sets up the training loop used by the tuning run.

        #    TODO: Build the correct trainer.
        #    TODO: Set the correct train_fn. The train_fn may look a lot like trainer.train(), but it
        #     will probably need to end with a tune.report() call which captures metrics (loss,
        #     accuracy, etc.) from the current trial.

    def setup_evaluation(self):
        pass
        # TODO: This sets up the evaluation loop used by the "best" model.

        #    TODO: Build the correct evaluator.
        #    TODO: Set the correct eval_fn.

    def tune(self):
        pass

        # TODO: Perform the tuning run.

        # result = tune.run(
        #     tune_with_parameters(self.train_fn),
        #     resources_per_trial=self.trial_resources,
        #     config=self.search_space,
        #     search_alg=self.search_algo,
        #     metric=self.metric,
        #     mode=self.mode,
        #     num_samples=self.num_samples,
        #     scheduler=self.scheduler
        # )
        #
        # TODO: Once the tuning is complete, you fetch the best trial and perform an eval.
        # self.best_result = result.get_best_trial(self.metric, self.mode, self.scope)

    # TODO: Better name for this method.
    def process_best_result(self):
        pass

        # TODO: Once a tuning run is complete, there are various things that can be done with
        #  the tuning result. Those would be done here. Maybe the tuning config specifies the course
        #  of action somehow? Do we spit out a model config with the best parameters?
        # self.eval_fn(best_trial)
    