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
from types import SimpleNamespace

import tensorflow as tf

import juneberry.evaluation.evaluator
import juneberry.evaluation.utils
from juneberry.config.dataset import DatasetConfig
from juneberry.config.eval_output import EvaluationOutput
from juneberry.config.model import ModelConfig
from juneberry.evaluation.evaluator import EvaluatorBase
from juneberry.filesystem import ModelManager, EvalDirMgr
import juneberry.tensorflow.data as tf_data
import juneberry.tensorflow.utils as tf_utils
from juneberry.tensorflow.utils import TensorFlowPlatformDefinitions
import juneberry.utils

logger = logging.getLogger(__name__)


class Evaluator(EvaluatorBase):
    def __init__(self, model_config: ModelConfig, lab, model_manager: ModelManager, eval_dir_mgr: EvalDirMgr,
                 dataset: DatasetConfig, eval_options: SimpleNamespace = None, log_file: str = None):
        super().__init__(model_config, lab, model_manager, eval_dir_mgr, dataset, eval_options, log_file)

        # TODO: This should be in base
        self.dataset_config = dataset
        self.eval_loader = None
        self.eval_labels = None

        # TODO: Should these be in the training config?
        self.shape_hwc = self.model_config.model_architecture.get_shape_hwc()

        self.eval_results = None
        self.predictions = None

    # ==========================================================================

    @classmethod
    def get_platform_defs(cls):
        return TensorFlowPlatformDefinitions()

    # ==========================================================================

    @classmethod
    def get_eval_output_files(cls, model_mgr: ModelManager, dataset_path: str, dryrun: bool = False):
        """
        Returns a list of files to clean from the eval directory. This list should contain ONLY
        files or directories that were produced by the evaluate command. Directories in this list
        will be deleted even if they are not empty.
        :param model_mgr: A ModelManager to help locate files.
        :param dataset_path: A string indicating the name of the dataset being evaluated.
        :param dryrun: When True, returns a list of files created during a dryrun of the Evaluator.
        :return: The files to clean from the eval directory.
        """
        eval_dir_mgr = model_mgr.get_eval_dir_mgr(dataset_path)
        if dryrun:
            return [eval_dir_mgr.get_manifest_path(),
                    eval_dir_mgr.get_dir()]
        else:
            return [eval_dir_mgr.get_predictions_path(),
                    eval_dir_mgr.get_metrics_path(),
                    eval_dir_mgr.get_manifest_path(),
                    eval_dir_mgr.get_dir()]

    @classmethod
    def get_eval_clean_extras(cls, model_mgr: ModelManager, dataset_path: str, dryrun: bool = False):
        """
        Returns a list of extra "evaluation" files to clean. Directories in this list will NOT
        be deleted if they are not empty.
        :param model_mgr: A ModelManager to help locate files.
        :param dataset_path: A string indicating the name of the dataset being evaluated.
        :param dryrun: When True, returns a list of files created during a dryrun of the Trainer.
        :return: The extra files to clean from the training directory.
        """
        eval_dir_mgr = model_mgr.get_eval_dir_mgr(dataset_path)
        if dryrun:
            return [eval_dir_mgr.get_dir().parent]
        else:
            return [eval_dir_mgr.get_dir().parent]

    @classmethod
    def get_default_metric_value(cls, eval_data: EvaluationOutput):
        """ :return: The value of the Evaluator's default metric as found in the results structure """
        return eval_data.results.metrics.accuracy, "accuracy"

    # ==========================================================================
    def dry_run(self) -> None:
        self.setup()
        self.obtain_dataset()
        self.obtain_model()

        logger.info(f"Dryrun complete.")

    # ==========================================================================

    def check_gpu_availability(self, required: int):
        return 0

    def setup(self) -> None:
        # Set all seed values.
        tf_utils.set_tensorflow_seeds(self.model_config.seed)

        # Use default values if they were not provided in the model config.
        if self.eval_method is None:
            self.eval_method = "juneberry.tensorflow.evaluation.default.TFEvaluationProcedure"
        if self.eval_output_method is None:
            self.eval_output_method = "juneberry.tensorflow.evaluation.default.TFEvaluationOutput"

    def obtain_dataset(self) -> None:
        logger.info(f"Splitting the dataset according to the model's validation split instructions.")
        self.eval_loader, self.eval_labels = tf_data.load_eval_dataset(
            self.lab, self.eval_dataset_config, self.model_config, self.eval_dir_mgr,
            self.use_train_split, self.use_val_split)

    def obtain_model(self) -> None:
        hdf5_file = self.model_manager.get_model_path(TensorFlowPlatformDefinitions())
        if hdf5_file.exists():
            logger.info(f"Loading model {hdf5_file}...")
            self.model = tf.keras.models.load_model(hdf5_file)
            logger.info("...complete")
        else:
            logger.warning(f"Failed to load model. File may not exist: {hdf5_file}")

    def evaluate_data(self) -> None:
        logger.info(f"Generating EVALUATION data according to {self.eval_method}")
        logger.info(f"Will evaluate model {self.model_manager.model_name} using {self.eval_dataset_config_path}")

        juneberry.evaluation.utils.invoke_evaluator_method(self, self.eval_method)

        logger.info(f"EVALUATION COMPLETE.")

    def format_evaluation(self) -> None:
        logger.info(f"Formatting raw EVALUATION data according to {self.eval_output_method}")
        juneberry.evaluation.utils.invoke_evaluator_method(self, self.eval_output_method)
