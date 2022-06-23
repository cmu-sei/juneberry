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
from dataclasses import dataclass
import logging
import os

from juneberry.config.model import LRStepFrequency
from juneberry.evaluation.evaluator import EvaluatorBase
import juneberry.loader as jb_loader
from juneberry.scripting.sprout import Sprout, TrainingSprout
from juneberry.training.trainer import Trainer
import juneberry.training.utils as jb_training_utils
from juneberry.tuning.tuner import Tuner

logger = logging.getLogger(__name__)


@dataclass
class Pollinator:
    evaluator: EvaluatorBase = None
    sprout: Sprout = None
    trainer: Trainer = None
    tuner: Tuner = None

    def assign_sprout(self, sprout: Sprout):
        self.sprout = sprout

    def get_trainer(self):
        if type(self.sprout) is not TrainingSprout:
            logger.warning(f"The sprout associated with this pollinator is not a TrainingSprout. "
                           f"Attempting to establish a Trainer anyway.")

        if self.sprout.model_config is None:
            logger.warning(f"There is no model config associated with the sprout. Unable to "
                           f"determine which type of trainer to build.")

        else:
            self._assemble_trainer()

        return self.trainer

    def _assemble_trainer(self):
        if self.sprout.model_config.trainer is None:
            self.trainer = jb_training_utils.assemble_stanza_and_construct_trainer(self.sprout.model_config)
        else:
            fqcn = self.sprout.model_config.trainer.fqcn
            kwargs = self.sprout.model_config.trainer.kwargs
            self.trainer = jb_loader.construct_instance(fqcn, kwargs)
            self._pollinate_trainer(fqcn)

    def _pollinate_trainer(self, fqcn: str):
        if fqcn == "juneberry.pytorch.classifier_trainer.ClassifierTrainer":
            self._pytorch_classifier_pollination()

    def _base_trainer_pollination(self):
        self.trainer.model_manager = self.sprout.model_manager
        self.trainer.model_config = self.sprout.model_config
        self.trainer.log_level = self.sprout.log_level
        self.trainer.num_gpus = self.sprout.num_gpus
        self.trainer.dryrun = self.sprout.dryrun
        self.trainer.native_output_format = self.sprout.native_output_format
        self.trainer.onnx_output_format = self.sprout.onnx_output_format
        self.trainer.lab = self.sprout.lab
        self.trainer.dataset_config = self.trainer.lab.load_dataset_config(
            self.trainer.model_config.training_dataset_config_path)

    def _epoch_trainer_pollination(self):
        self._base_trainer_pollination()

        self.trainer.max_epochs = self.trainer.model_config.epochs
        self.trainer.done = False if self.trainer.max_epochs > self.trainer.epoch else True

    def _pytorch_classifier_pollination(self):
        self._epoch_trainer_pollination()

        self.trainer.data_version = self.trainer.model_manager.model_version
        self.trainer.binary = self.trainer.dataset_config.is_binary
        self.trainer.pytorch_options = self.trainer.model_config.pytorch

        self.trainer.no_paging = False
        if "JB_NO_PAGING" in os.environ and os.environ['JB_NO_PAGING'] == "1":
            logger.info("Setting to no paging mode.")
            self.no_paging = True

        self.trainer.memory_summary_freq = int(os.environ.get("JUNEBERRY_CUDA_MEMORY_SUMMARY_PERIOD", 0))

        self.trainer.lr_step_frequency = LRStepFrequency.EPOCH
