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

import juneberry.evaluation.evaluator
from juneberry.config.dataset import DatasetConfig
from juneberry.config.model import ModelConfig
from juneberry.filesystem import ModelManager, EvalDirMgr
import juneberry.tensorflow.data as tf_data

logger = logging.getLogger(__name__)


class TFEvaluator(juneberry.evaluation.evaluator.Evaluator):
    def __init__(self, model_config: ModelConfig, lab, dataset: DatasetConfig, model_manager: ModelManager,
                 eval_dir_mgr: EvalDirMgr, eval_options: SimpleNamespace = None):
        super().__init__(model_config, lab, dataset, model_manager, eval_dir_mgr, eval_options)

        # TODO: This should be in base
        self.dataset_config = dataset
        self.eval_loader = None
        self.eval_labels = None

        # TODO: Should these be in the training config?
        self.shape_hwc = self.model_config.model_architecture.get_shape_hwc()

        self.eval_results = None
        self.predictions = None

    def check_gpu_availability(self, required: int):
        return 0

    def setup(self) -> None:
        logger.info(f"Setting random seed: {self.model_config.seed}")
        random.seed(self.model_config.seed)
        np.random.seed(self.model_config.seed)
        tf.random.set_seed(self.model_config.seed)

    def obtain_dataset(self) -> None:
        logger.info(f"Splitting the dataset according to the model's validation split instructions.")
        self.eval_loader, self.eval_labels = tf_data.load_eval_dataset(
            self.lab, self.eval_dataset_config, self.model_config, self.eval_dir_mgr,
            self.use_train_split, self.use_val_split)

    def obtain_model(self) -> None:
        hdf5_file = self.model_manager.get_tensorflow_model_path()
        logger.info(f"Loading model {hdf5_file}...")
        self.model = tf.keras.models.load_model(hdf5_file)
        logger.info("...complete")

    def evaluate_data(self) -> None:
        # loss, acc
        logger.info("Evaluating...")
        self.eval_results = self.model.evaluate(self.eval_loader)
        logger.info(f"  loss={self.eval_results[0]}, accuracy={self.eval_results[1]}")
        logger.info(f"...generating predictions...")
        self.predictions = self.model.predict(self.eval_loader)
        logger.info(f"...evaluation complete.")

    def format_evaluation(self) -> None:
        logger.info(f"Formatting raw evaluation data")
        # Add the dataset mapping and the number of classes the model is aware of to the output.
        self.output.options.dataset.classes = self.eval_dataset_config.label_names
        self.output.options.model.num_classes = self.eval_dataset_config.num_model_classes

        self.output.results.labels = self.eval_labels
        self.output.results.metrics.loss = self.eval_results[0]
        self.output.results.metrics.accuracy = self.eval_results[1]
        self.output.results.predictions = self.predictions.tolist()
        self.output_builder.save_predictions(self.eval_dir_mgr.get_predictions_path())
        self.output_builder.save_metrics(self.eval_dir_mgr.get_metrics_path())


# HACK - This needs to go into the workspace
class GloroEvaluator(TFEvaluator):
    def __init__(self, model_config: ModelConfig, lab, dataset: DatasetConfig, model_manager: ModelManager,
                 eval_dir_mgr: EvalDirMgr, eval_options: SimpleNamespace = None):
        super().__init__(model_config, lab, dataset, model_manager, eval_dir_mgr, eval_options)

    def obtain_model(self) -> None:
        from gloro import GloroNet
        model_file = self.model_manager.get_train_root_dir() / "model.gloronet"
        logger.info(f"Loading model {model_file}...")
        self.model = GloroNet.load_model(model_file)
        logger.info("...complete")
