#! /usr/bin/env python3

# ======================================================================================================================
# Juneberry - Release 0.5
#
# Copyright 2022 Carnegie Mellon University.
#
# NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS"
# BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER
# INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED
# FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM
# FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.
#
# Released under a BSD (SEI)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.
#
# [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution. Please see
# Copyright notice for non-US Government use and distribution.
#
# This Software includes and/or makes use of Third-Party Software each subject to its own license.
#
# DM22-0856
#
# ======================================================================================================================

"""
This base class provides a skeleton for evaluating models in Juneberry. Evaluation makes
the following assumptions:
  - There is a dataset file that the user would like to use for evaluation.
  - There is a model the user would like to evaluate.
  - There is a model config for the model the user would like to evaluate.
  - The model config describes how the evaluation should be performed and how the
    evaluation output should be formatted.

The class contains several extension points that provide opportunities for customization.
"""

import datetime
import logging
from types import SimpleNamespace

from juneberry.config.dataset import DatasetConfig
from juneberry.config.eval_output import EvaluationOutputBuilder, EvaluationOutput
from juneberry.config.model import ModelConfig
from juneberry.filesystem import EvalDirMgr, ModelManager
from juneberry.lab import Lab

logger = logging.getLogger(__name__)


class EvaluatorBase:
    """
    This class encapsulates the process of evaluating a model.
    """

    def __init__(self, model_config: ModelConfig, lab: Lab, model_manager: ModelManager, eval_dir_mgr: EvalDirMgr,
                 dataset: DatasetConfig = None, eval_options: SimpleNamespace = None, log_file: str = None, **kwargs):
        """
        Construct an Evaluator based on command line arguments and a Juneberry ModelManager object.
        :param model_config: The model config used to train the model.
        :param lab: The Juneberry Lab in which to run the evaluation.
        :param model_manager: A Juneberry ModelManager object responsible for managing operations involving the
        model to be evaluated.
        :param eval_dir_mgr: A Juneberry EvalDirMgr object responsible for managing file path operations
        within the model's eval directory.
        :param dataset: A Juneberry DatasetConfig object representing the dataset to be evaluated.
        :param eval_options: A SimpleNamespace containing various options for the evaluation. Expected options
        include the following: top_k, use_train_split, use_val_split.
        :param log_file: A string indicating the location of the current log file.
        """
        # TODO: Should we make a model manager or get passed one???

        # Stash the eval directory manager.
        self.eval_dir_mgr = eval_dir_mgr

        # Stash the location of the log file.
        self.log_file_path = log_file

        # Stash the lab off so everyone can use it
        self.lab = lab

        # Attribute that determines if a dry run of the evaluation is performed.
        self.dryrun = False

        # How many GPUs to use. 0 is CPU.
        self.num_gpus = 0

        # These attributes describe the model being evaluated. The "model_config" is a Juneberry ModelConfig
        # object. The "model" is the object used to perform the evaluation.
        # TODO: Should we load the model config now??
        self.model_manager = model_manager
        self.model_config_path = model_manager.get_model_config()
        self.model_config = model_config
        self.model = None

        # These attributes describe the dataset being evaluated. The "eval_dataset_config" is a Juneberry
        # DatasetConfig object.
        self.eval_dataset_config = dataset
        self.use_train_split = False
        self.use_val_split = False

        if dataset:
            # This is fragile. The file_path in the dataset is OPTIONAL
            # TODO: Find a way to make a data path if we don't have one for notebook support
            if dataset.file_path is None:
                logger.error("The evaluator requires an output file path.")
            # TODO: This is too long and really muddies up the code.
            self.eval_dataset_config_path = dataset.file_path

        #  A list of pairs of some item name or id and truth label (target)
        self.eval_name_targets = []

        # These attributes track whether any custom classes are required to load the eval data, conduct
        # the evaluation, or format the evaluation. Not all evaluators require these classes to be
        # specified in order for the evaluation to take place, but some, such as the ONNX evaluator, do.
        self.eval_data_loader_method = None
        self.eval_method = None
        self.eval_output_method = None

        # The loader kwarg provides users the ability to specify a custom class for building the evaluation
        # data loader. The output and procedure kwargs to evaluator allow for custom classes
        # to conduct the evaluation and/or format the eval results. When not provided, the evaluator subclasses
        # may set these to the default classes when required.
        if self.model_config.evaluator is not None and self.model_config.evaluator.kwargs is not None:
            if 'loader' in self.model_config.evaluator.kwargs:
                self.eval_data_loader_method = self.model_config.evaluator.kwargs['loader']
            if 'output' in self.model_config.evaluator.kwargs:
                self.eval_output_method = self.model_config.evaluator.kwargs['output']
            if 'procedure' in self.model_config.evaluator.kwargs:
                self.eval_method = self.model_config.evaluator.kwargs['procedure']

        # These attributes are all related to the output of the evaluation process. They contain the
        # raw data that resulted from the evaluation process, the formatted output data that will be
        # written to an output JSON file, and (if requested) the top-K classes predicted for each piece
        # of evaluated data.
        self.raw_output = None
        self.top_k = None

        # Set up the eval output
        self._init_output()

        # Check the eval_options for values related to the relevant attributes. If found, set the
        # attribute.
        for option in ['dryrun', 'use_train_split', 'top_k', 'use_val_split']:
            try:
                setattr(self, option, getattr(eval_options, option))
            except AttributeError:
                pass

        self.metrics_plugins = self.model_config.evaluation_metrics

    # -----------------------------------------------
    #  _____     _                 _              ______     _       _
    # |  ___|   | |               (_)             | ___ \   (_)     | |
    # | |____  _| |_ ___ _ __  ___ _  ___  _ __   | |_/ /__  _ _ __ | |_ ___
    # |  __\ \/ / __/ _ \ '_ \/ __| |/ _ \| '_ \  |  __/ _ \| | '_ \| __/ __|
    # | |___>  <| ||  __/ | | \__ \ | (_) | | | | | | | (_) | | | | | |_\__ \
    # \____/_/\_\\__\___|_| |_|___/_|\___/|_| |_| \_|  \___/|_|_| |_|\__|___/

    @classmethod
    def get_platform_defs(cls):
        """ :return: An object (PlatformDefinitions) containing methods for various platform details. """
        logger.error(f"get_platform_defs() must be defined as a static method on the evaluator.")
        raise RuntimeError(f"get_platform_defs() must be defined as a static method on the evaluator.")

    @classmethod
    def get_default_metric_value(cls, eval_data: EvaluationOutput):
        """ :return: The value of the Evaluator's default metric as found in the results structure """
        # TODO: This should be able to be replaced using the first metrics from the results
        #  when we change the evaluation output to be an array of results.
        logger.error(f"get_default_metric_value() not implemented in {cls}")
        raise RuntimeError(f"get_default_metric_value() not implemented in {cls}")

    @classmethod
    def get_eval_output_files(cls, model_mgr: ModelManager, dataset_path: str, dryrun: bool):
        """
        Returns a list of files to clean from the eval directory. This list should contain ONLY
        files or directories that were produced by the evaluate command. Directories in this list
        will be deleted even if they are not empty.
        :param model_mgr: A ModelManager to help locate files.
        :param dataset_path: A string indicating the name of the dataset being evaluated.
        :param dryrun: When True, returns a list of files created during a dryrun of the Evaluator.
        :return: The files to clean from the eval directory.
        """
        logger.error(f"get_evaluation_output_files() must be defined as a static method on the evaluator")
        raise RuntimeError(f"get_evaluation_output_files() must be defined as a static method on the evaluator")

    @classmethod
    def get_eval_clean_extras(cls, model_mgr: ModelManager, dataset_path: str, dryrun: bool):
        """
        Returns a list of extra "evaluation" files to clean. Directories in this list will NOT
        be deleted if they are not empty.
        :param model_mgr: A ModelManager to help locate files.
        :param dataset_path: A string indicating the name of the dataset being evaluated.
        :param dryrun: When True, returns a list of files created during a dryrun of the Trainer.
        :return: The extra files to clean from the training directory.
        """
        logger.error(f"get_evaluation_clean_extras() must be defined as a static method on the evaluator")
        raise RuntimeError(f"get_evaluation_clean_extras() must be defined as a static method on the evaluator")

    def dry_run(self) -> None:
        """
        Executes a "dryrun" of the evaluation, checking for model viability, dataset properties, etc.
        :return: None
        """
        pass

    def check_gpu_availability(self, required: int) -> int:
        """
        This allows the particular backend to use its own method of determining resource
        availability.
        :param required: The number of required gpus. 'None' will use the maximum available.
        :return: The number of gpus the evaluator can use.
        """
        return 0

    def setup(self) -> None:
        """
        The intent of this extension point is to perform any setup steps required by the evaluation process.
        :return: Nothing
        """
        pass

    def reset(self) -> None:
        pass

    def obtain_dataset(self) -> None:
        """
        The intent of this extension point is to perform the steps required to obtain the dataset being used
        for evaluation.
        :return: Nothing
        """
        pass

    def obtain_model(self) -> None:
        """
        The intent of this extension point is to perform the steps required to obtain the model being evaluated.
        :return: Nothing
        """
        pass

    def evaluate_data(self) -> None:
        """
        The intent of this extension point is to perform the evaluation. The dataset and model are used to
        generate raw evaluation data.
        :return: Nothing
        """
        pass

    def format_evaluation(self) -> None:
        """
        The intent of this extension point is to format the raw evaluation data into something human-readable.
        :return: Nothing
        """
        pass

    def populate_metrics(self) -> None:
        """
        The intent of this extension point is to populate the evaluation output with metrics.
        :return: Nothing
        """
        pass

    def perform_evaluation(self) -> None:
        """
        The order in which the extension points are called is described here. After setup occurs, the
        dataset is acquired, followed by the model. Then the dataset is evaluated in the model, which
        produces raw evaluation data, followed by a step to format the raw data.
        :return: Nothing
        """
        self.setup()

        # Report what the methods are we are finally using
        logger.info(f"Eval Data Loader FQCN: {self.eval_data_loader_method}")
        logger.info(f"Eval Procedure FQCN: {self.eval_method}")
        logger.info(f"Eval Output FQCN: {self.eval_output_method}")

        self.obtain_dataset()
        self.obtain_model()

        # Record the time the evaluation started.
        self.output.times.start_time = datetime.datetime.now().replace(microsecond=0)

        self.evaluate_data()

        self.populate_metrics()

        # Record the time the evaluation ended.
        self.output.times.end_time = datetime.datetime.now().replace(microsecond=0)

        # Format the evaluation times in the output and calculate the duration.
        self.output_builder.set_times(self.output.times.start_time, self.output.times.end_time)

        self.format_evaluation()

    def perform_additional_eval(self, log_file, dataset: DatasetConfig, eval_dir_mgr: EvalDirMgr,
                                dryrun: bool = False) -> None:
        """
        Allows the evaluation of a different dataset that has the same basic characteristics
        as the original data set, in terms of model layer sizes, etc.
        :param log_file: The file to log to
        :param dataset: The dataset to evaluate
        :param eval_dir_mgr: Where to place the evaluation
        :param dryrun: Boolean indicating whether or not to conduct a dryrun of the eval.
        :return: Nothing
        """
        # This is all reinit stuff that we could factor out of __init__
        # TODO: should go into a separate init function.

        # Reset the eval dataset, output directory, and log file location.
        self.eval_dataset_config = dataset
        self.eval_dir_mgr = eval_dir_mgr
        self.log_file_path = log_file

        # Perform the dry run if one was requested.
        if dryrun:
            self.dry_run()
            return

        # Set up the eval output
        self._init_output()

        self.reset()

        # ===

        self.obtain_dataset()
        # Record the time the evaluation started.
        self.output.times.start_time = datetime.datetime.now().replace(microsecond=0)

        self.evaluate_data()

        self.populate_metrics()

        # Record the time the evaluation ended.
        self.output.times.end_time = datetime.datetime.now().replace(microsecond=0)

        # Format the evaluation times in the output and calculate the duration.
        self.output_builder.set_times(self.output.times.start_time, self.output.times.end_time)

        self.format_evaluation()

    def _init_output(self):
        # Set up the evaluation output.
        self.output_builder = EvaluationOutputBuilder()
        self.output = self.output_builder.output

        # Record some initial information in the evaluation output, such as the model being
        # evaluated and the dataset used in the evaluation.
        self.output.options.model.name = self.model_manager.model_name
        self.output.options.dataset.config = self.eval_dataset_config.file_path if self.eval_dataset_config else None


def main():
    pass


if __name__ == "__main__":
    main()
