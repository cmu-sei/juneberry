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

import datetime
import logging
import os

from ray import tune
from ray.tune.integration.torch import distributed_checkpoint_dir
import torch

from juneberry.config.tuning_output import TuningOutputBuilder
import juneberry.filesystem as jb_fs
import juneberry.loader as jb_loader
import juneberry.logging as jb_logging
import juneberry.scripting.utils as jb_scripting_utils
from juneberry.training.trainer import Trainer
from juneberry.tuning.reporter import CustomReporter

logger = logging.getLogger(__name__)


class Tuner:
    """
    This class is responsible for tuning models in Juneberry.
    """

    def __init__(self):
        # The trainer factory produces Juneberry trainers for the Tuner.
        self.trainer_factory = None

        # The TuningSprout contains any script args that may have been passed to
        # a Juneberry Tuning script.
        self.sprout = None

        # The TuningConfig describes properties that affect tuning.
        self.tuning_config = None

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

        # Attribute to capture the best tuning result.
        self.best_result = None

        # These attributes are responsible for producing the tuning output.json.
        self.output_builder = TuningOutputBuilder()
        self.output = self.output_builder.output

        # Store the time that tuning started / ended.
        self.tuning_start_time = None
        self.tuning_end_time = None

    # ==========================

    @staticmethod
    def get_tuning_output_files(model_mgr: jb_fs.ModelManager, dryrun: bool = False):
        if dryrun:
            return []
        else:
            return [model_mgr.get_tuning_dir()]

    @staticmethod
    def get_tuning_clean_extras(model_mgr: jb_fs.ModelManager, dryrun: bool = False):
        if dryrun:
            return []
        else:
            return [model_mgr.get_train_root_dir()]

    # ==========================

    def _build_tuning_components(self) -> None:
        """
        This method is responsible for assembling various tuning components and setting the
        corresponding Tuner attribute.
        :return:
        """
        # Extract the search_space from the tuning config.
        self._build_search_space()

        # Extract the scheduler from the tuning config and then build the desired scheduler.
        if self.tuning_config.scheduler is not None:
            self._build_scheduler()
        else:
            logger.warning(f"The tuning config does not define a scheduler. Ray Tune uses "
                           f"'ray.tune.schedulers.FIFOScheduler' by default.")

        # Extract the search algorithm for the search space and then build it.
        if self.tuning_config.search_algorithm is not None:
            self._build_search_algo()
        else:
            logger.warning(f"The tuning config does not define a search algorithm. Ray Tune uses "
                           f"'ray.tune.suggest.basic_variant.BasicVariantGenerator' by default.")

    def _build_search_space(self) -> None:
        """
        This method is responsible for constructing the search space for the Tuner.
        :return: Nothing.
        """
        # The goal is to create a dictionary that tune.run() will use to select hyperparameters
        # for each tuning trial. In this dictionary, the keys are the hyperparameter to adjust, and
        # the values are functions that define the allowed range of sample values for the hyperparameter.
        search_space = {}

        # Convert the search space defined in the tuning config into the format tune.run() expects.
        for variable in self.tuning_config.search_space:
            search_space[variable.hyperparameter_name] = jb_loader.construct_instance(variable.fqcn, variable.kwargs)

        # Once the entire search space has been assembled, assign it to the Tuner attribute.
        self.search_space = search_space

    def _build_scheduler(self) -> None:
        """
        This method is responsible for constructing the scheduler for the Tuner.
        :return: Nothing.
        """
        # Retrieve the desired scheduler from the tuning config.
        scheduler_dict = self.tuning_config.scheduler
        logger.info(f"Constructing tuning scheduler using fqcn: {scheduler_dict.fqcn}")

        # All schedulers in Ray Tune use 'self.metric' and 'self.mode' to make decisions about
        # terminating bad trials, altering parameters in a running trial, etc. However,
        # tune.run() receives both of these properties directly and will complain when they are
        # also provided to the scheduler.
        if "metric" in scheduler_dict.kwargs:
            logger.warning(f"The scheduler does not need a 'metric' parameter since the 'metric' arg is "
                           f"passed to tune.run(). Remove the 'metric' kwarg from the 'scheduler' section "
                           f"in the tuning config file to eliminate this warning.")
            scheduler_dict.kwargs.pop("metric")

        if "mode" in scheduler_dict.kwargs:
            logger.warning(f"The scheduler does not need a 'mode' parameter since the 'mode' arg is "
                           f"passed to tune.run(). Remove the 'mode' kwarg from the 'scheduler' section "
                           f"in the tuning config file to eliminate this warning.")
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
        algo_dict = self.tuning_config.search_algorithm
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

        # Locate the Tuning log file and set up the root "juneberry" logger for this thread.
        log_file = self.trainer_factory.lab.workspace() / self.trainer_factory.model_manager.get_tuning_log()
        jb_logging.setup_logger(log_file=log_file,
                                log_prefix=jb_scripting_utils.standard_line_prefix(self.sprout.dryrun),
                                log_to_console=not self.sprout.silent,
                                level=self.sprout.log_level,
                                name="juneberry")

        # This will substitute the set of hyperparameters chosen for the trial into the ModelConfig stored in the
        # TrainerFactory.
        # trial_model_config is a ModelConfig object.
        trial_model_config = self.trainer_factory.model_config.adjust_attributes(config)

        # Once the trial's ModelConfig has been created, adjust the store it in the TrainerFactory and retrieve
        # the resulting Trainer object.
        self.trainer_factory.set_model_config(trial_model_config)
        trainer = self.trainer_factory.get_trainer()

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

        # Check how many GPUs are available.
        trainer.num_gpus = trainer.check_gpu_availability(self.trainer_factory.lab.profile.num_gpus)

        # Check if the lab profile places any constraints on the number of GPUs to use.
        if self.trainer_factory.lab.profile.num_gpus is not None:
            if trainer.num_gpus > self.trainer_factory.lab.profile.num_gpus:
                logger.info(f"Maximum numbers of GPUs {trainer.num_gpus} being capped to "
                            f"{self.trainer_factory.lab.profile.num_gpus} because of lab profile.")
                trainer.num_gpus = self.trainer_factory.lab.profile.num_gpus

        # Assign the GPU if one was requested or initiate distributed training if multiple GPUs were requested.
        if trainer.num_gpus == 0:
            trainer.gpu = None
        elif trainer.num_gpus == 1:
            trainer.gpu = 0
        else:
            # TODO: Figure out the right way to do this.
            # trainer.train_distributed(trainer.num_gpus)
            # return
            pass

        # No matter the number of GPUs, setup the node for training.
        trainer.node_setup()
        trainer.setup()

        # Continue fetching metrics until the maximum number of training epochs has
        # been achieved (unless the trial gets stopped early by the Tuner).
        while cur_epoch < trial_model_config.epochs:
            # Fetch the latest metrics.
            returned_metrics = trainer.tune()

            # Determine if model checkpointing was desired.
            if self.tuning_config.tuning_parameters.checkpoint_interval:
                # This ensures a model checkpoint will be saved every after every N epochs, where
                # N is equal to the checkpoint_interval.
                if not (cur_epoch + 1) % self.tuning_config.tuning_parameters.checkpoint_interval:
                    # Distributed training requires a unique type of checkpoint directory.
                    if self.tuning_config.trial_resources.gpu > 1:
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

    def tune(self, dryrun: bool = False) -> None:
        """
        This method performs a tuning run based on the conditions set by all of the various
        attributes of the Tuner.
        :param dryrun: A boolean indicating whether to tune in dryrun mode. Default is False.
        :return: Nothing.
        """
        # Record the Tuning options.
        self.output_builder.set_tuning_options(model_name=self.sprout.model_name,
                                               tuning_config=self.sprout.tuning_config)

        # Construct the various components required for tuning.
        self._build_tuning_components()

        # Determine which Trainable to use for the tuning run. The main decision in this step is
        # to figure out if a distributed trainable is needed.
        trainable, trial_resources = self._determine_trainable()

        if not dryrun:

            logger.info(f"Starting the tuning run.")

            # Capture the time when the tuning run started.
            self.tuning_start_time = datetime.datetime.now().replace(microsecond=0)

            # Perform the tuning run.
            result = tune.run(
                trainable,
                resources_per_trial=trial_resources,
                config=self.search_space,
                search_alg=self.search_algo,
                metric=self.tuning_config.tuning_parameters.metric,
                mode=self.tuning_config.tuning_parameters.mode,
                num_samples=self.tuning_config.num_samples,
                scheduler=self.scheduler,
                local_dir=str(self.trainer_factory.model_manager.get_tuning_dir()),
                progress_reporter=CustomReporter(),
                trial_dirname_creator=self._trial_dirname_string_creator
            )

            # Capture the time when the tuning run ended.
            self.tuning_end_time = datetime.datetime.now().replace(microsecond=0)

            # Once tuning is complete, store the best result.
            logger.info(f"The tuning run is complete. Storing the best result.")
            self.best_result = result.get_best_trial(self.tuning_config.tuning_parameters.metric,
                                                     self.tuning_config.tuning_parameters.mode,
                                                     self.tuning_config.tuning_parameters.scope)

            # Retrieve the data from inside each trial's result.json file.
            for trial in result.trials:
                result_path = self.trainer_factory.model_manager.get_tuning_result_file(trial.logdir)
                trial_data = jb_fs.load_json_lines(result_path)

                # Add the trial's result data to the tuning output.
                self.output_builder.append_trial_result(directory=trial.logdir, params=trial.config,
                                                        trial_data=trial_data)

            # Perform any final tuning steps, such as indicating the "best result".
            self.finish_tuning()

    def _determine_trainable(self) -> tuple:
        """
        The purpose of this method is to make adjustments to the Trainable if distributed
        training has been requested.
        :return: A tuple with the Trainable, and the trial_resources
        """
        # Make the necessary adjustments to support tuning a model that requires distributed training.
        if self.tuning_config.trial_resources.gpu > 1:
            # TODO: Implement distributed tuning.
            # For now, log a warning that distributed tuning is not supported and lower the number of
            # GPU resources to 1.
            logger.warning(f"Distributed tuning not implemented yet. Setting trial GPU resources "
                           f"to 1.")
            trainable = self._tuning_attempt
            self.tuning_config.trial_resources.gpu = 1
            trial_resources = self.tuning_config.trial_resources

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
            trial_resources = self.tuning_config.trial_resources

        return trainable, trial_resources

    @staticmethod
    def _trial_dirname_string_creator(trial) -> str:
        """
        The purpose of this method is to shorten the directory name used to contain the files for each trial.
        Without this, tuning runs involving many hyperparameters run the risk of generating directory names that
        may be too long on a Windows system.
        :param trial: A Ray Tune Trial object.
        :return: A string to use for the Trial's directory.
        """
        return f"{trial.trainable_name}_{trial.trial_id}"

    def finish_tuning(self):
        # TODO: Once a tuning run is complete, there are various things that can be done with
        #  the best tuning result.
        #  Maybe save the best model config?
        logger.info(f"Best trial config: {self.best_result.config}")
        logger.info(f"Best trial final '{self.tuning_config.tuning_parameters.metric}': "
                    f"{self.best_result.last_result[self.tuning_config.tuning_parameters.metric]}")

        # Move the Juneberry tuning log into the tuning directory for this run.
        log_dest = self.trainer_factory.model_manager.get_relocated_tuning_log(self.best_result.local_dir)
        logger.info(f"Moving log file to {log_dest}")
        self.trainer_factory.model_manager.get_tuning_log().rename(log_dest)

        # Save a copy of the tuning log to the log directory. The prefix helps identify when the
        # tuning run took place.
        tuning_log_dir = self.trainer_factory.lab.workspace() / self.trainer_factory.model_manager.get_tuning_log_dir()
        prefix = str(self.best_result.local_dir).split("/")[-1]
        log_copy = self.trainer_factory.model_manager.get_relocated_tuning_log(target_dir=tuning_log_dir, prefix=prefix)
        logger.info(f"Copying log file to {log_copy}")
        log_copy.write_text(log_dest.read_text())

        # Store some data in the tuning output, then save the tuning output file.
        logger.info(f"Generating Tuning Output...")

        logger.info(f"  Recording time spent tuning.")
        self.output_builder.set_times(start_time=self.tuning_start_time, end_time=self.tuning_end_time)

        logger.info(f"  Recording the ID and config params of best trial.")
        self.output_builder.output.results.best_trial_id = self.best_result.trial_id
        self.output_builder.output.results.best_trial_params = self.best_result.config

        logger.info(f"  Saving Tuning Output file.")
        output_path = self.trainer_factory.model_manager.get_relocated_tuning_output(self.best_result.local_dir)
        self.output_builder.save(output_path)
