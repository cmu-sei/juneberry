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
import random

import numpy as np

import tensorflow as tf

import juneberry.data as jb_data
import juneberry.evaluation.evaluator
import juneberry.evaluation.utils
from juneberry.config.dataset import DatasetConfig
from juneberry.config.model import ModelConfig
from juneberry.evaluation.evaluator import EvaluatorBase
from juneberry.filesystem import ModelManager, EvalDirMgr
from juneberry.tensorflow.data import TFImageDataSequence
import juneberry.tensorflow.utils as tf_utils
from juneberry.transform_manager import TransformManager
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
            self.eval_method = "juneberry.evaluation.evals.tensorflow.TFEvaluationProcedure"
        if self.eval_output_method is None:
            self.eval_output_method = "juneberry.evaluation.evals.tensorflow.TFEvaluationOutput"

    def obtain_dataset(self) -> None:
        logger.info(f"Splitting the dataset according to the model's validation split instructions.")
        splitting_config = self.model_config.get_validation_split_config()
        eval_list, split = jb_data.dataspec_to_manifests(self.lab,
                                                         dataset_config=self.eval_dataset_config,
                                                         splitting_config=splitting_config,
                                                         preprocessors=TransformManager(
                                                             self.model_config.preprocessors))

        if self.use_train_split:
            logger.info("Evaluating using ONLY the training portion of the split data.")

        elif self.use_val_split:
            logger.info("Evaluating using ONLY the validation portion of the split data.")
            eval_list = split

        transforms = TransformManager(self.model_config.evaluation_transforms)
        self.eval_loader = TFImageDataSequence(eval_list, self.model_config.batch_size, transforms, self.shape_hwc)

        # Extract all the labels for later.
        self.eval_labels = [x[1] for x in eval_list]

    def obtain_model(self) -> None:
        hdf5_file = self.model_manager.get_tensorflow_model_path()
        logger.info(f"Loading model {hdf5_file}...")
        self.model = tf.keras.models.load_model(hdf5_file)
        logger.info("...complete")

    def evaluate_data(self) -> None:
        logger.info(f"Generating EVALUATION data according to {self.eval_method}")
        logger.info(f"Will evaluate model {self.model_manager.model_name} using {self.eval_dataset_config_path}")

        juneberry.evaluation.utils.invoke_evaluator_method(self, self.eval_method)

        logger.info(f"EVALUATION COMPLETE.")

    def format_evaluation(self) -> None:
        logger.info(f"Formatting raw EVALUATION data according to {self.eval_output_method}")
        juneberry.evaluation.utils.invoke_evaluator_method(self, self.eval_output_method)
