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
from typing import Union

from juneberry.config.dataset import DatasetConfig
from juneberry.config.model import ModelConfig
import juneberry.filesystem as jb_fs
from juneberry.lab import Lab
import juneberry.loader as jb_loader
from juneberry.trainer import Trainer
import juneberry.training.utils as jb_training_utils

logger = logging.getLogger(__name__)


@dataclass
class TrainerFactory:
    # Only contains attributes that are required to build Trainers
    dataset_config: DatasetConfig = None
    lab: Lab = None
    log_level: int = None
    model_config: ModelConfig = None
    model_manager: jb_fs.ModelManager = None

    def set_model_config(self, model_config: ModelConfig = None):
        """
        This method is responsible for setting the 'model_config" attribute of a TrainerFactory. When
        the desired ModelConfig is not provided to this method, it will attempt to use the ModelManager
        associated with the TrainerFactor to read the model config file from the model directory.
        :param model_config: A Juneberry ModelConfig object to associate with the TrainerFactory.
        :return: Nothing.
        """
        # If a ModelConfig was provided to the method, set the attribute directly.
        if model_config:
            self.model_config = model_config

        # When a ModelConfig isn't provided, attempt to use the ModelManager to read the model config
        # file located in the model directory.
        else:
            logger.info(f"Attempting to load the ModelConfig from the model directory.")
            if self.model_manager is not None:
                self.model_config = ModelConfig.load(self.model_manager.get_model_config())
            else:
                logger.warning(f"There is no ModelManager associated with the TrainerFactory, so "
                               f"the model directory could not be determined.")

    def get_trainer(self, resume: bool = False, native: bool = True, onnx: bool = False) -> Union[None, Trainer]:
        """
        This method is responsible for using the attributes in the TrainerFactory to build and return a
        Juneberry Trainer object.
        :param resume: A boolean indicating if the 'resume' optional arg should be passed to the Trainer
        object when it is created.
        :param native: A boolean indicating if the Trainer should output the resulting trained model file
        in the platform's native format.
        :param onnx: A boolean indicating if the Trainer should output the resulting trained model file
        in the ONNX format.
        :return: A Juneberry Trainer, or None when a Trainer could not be built.
        """
        # If a ModelConfig is not associated with the TrainerFactory, the Factory can't determine
        # the type of Trainer to build.
        if self.model_config is None:
            logger.warning(f"There is no model config associated with the TrainerFactory. Unable to "
                           f"determine which type of trainer to build.")
            return None

        # When a ModelConfig does exist, ModelConfig properties can be used to determine which type
        # of Trainer to build. Return the constructed Trainer.
        else:
            # Assemble the Trainer, allocate GPUs, and set the Trainer's output formats.
            trainer = self._assemble_trainer(resume=resume)
            trainer = self._allocate_trainer_gpus(trainer)
            trainer.set_output_format(native=native, onnx=onnx)

            return trainer

    def _assemble_trainer(self, resume: bool = False) -> Trainer:
        """
        This method is responsible for constructing a Juneberry Trainer object using the TrainerFactory's
        attributes.
        :param resume: A boolean indicating if the 'resume' optional arg should be passed to the Trainer
        object when it is created.
        :return: A Juneberry Trainer object.
        """
        # If the model config doesn't have a "trainer" stanza, then it's likely an older version. The
        # correct trainer fqcn will need to be retrieved via a task/platform mapping. There will be no kwargs.
        if self.model_config.trainer is None:
            trainer_fqcn = jb_training_utils.assemble_stanza_and_construct_trainer(self.model_config)
            trainer_kwargs = {}

        # Otherwise, retrieve the Trainer fqcn and kwargs from the ModelConfig's Trainer stanza.
        else:
            trainer_fqcn = self.model_config.trainer.fqcn
            trainer_kwargs = self.model_config.trainer.kwargs if self.model_config.trainer.kwargs is not None else {}

        # Set some additional Trainer kwargs to reflect TrainerFactory attributes. These kwargs are needed
        # to construct the Trainer instance.
        trainer_kwargs['lab'] = self.lab
        trainer_kwargs['model_manager'] = self.model_manager
        trainer_kwargs['model_config'] = self.model_config
        trainer_kwargs['log_level'] = self.log_level

        # The "dataset_config" kwarg must also be set, but it requires an extra step to determine the correct
        # location of the dataset config file within the workspace.
        dataset_config_path = self.lab.workspace() / self.model_config.training_dataset_config_path
        trainer_kwargs['dataset_config'] = self.lab.load_dataset_config(dataset_config_path)

        # Set the 'optional' resume kwarg.
        opt_args = {'resume': resume}

        # Construct the Trainer object.
        logger.info(f"Instantiating trainer: {trainer_fqcn}")
        return jb_loader.construct_instance(trainer_fqcn, trainer_kwargs, opt_args)

    def _allocate_trainer_gpus(self, trainer: Trainer) -> Trainer:
        # Allocate the GPUs the user asked for, getting the current number if None are required.
        trainer.num_gpus = trainer.check_gpu_availability(self.lab.profile.num_gpus)

        if self.lab.profile.max_gpus is not None:
            if trainer.num_gpus > self.lab.profile.max_gpus:
                logger.info(f"Maximum numbers of GPUs {trainer.num_gpus} being capped to "
                            f"{self.lab.profile.max_gpus} because of lab profile.")
                trainer.num_gpus = self.lab.profile.max_gpus

        return trainer
