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
import sys
from types import SimpleNamespace

import torch
from torch import Tensor

import juneberry.config.dataset as jb_dataset
from juneberry.config.model import ModelConfig
import juneberry.data as jbdata
from juneberry.evaluation.evaluator import Evaluator
from juneberry.filesystem import EvalDirMgr, ModelManager
from juneberry.lab import Lab
import juneberry.pytorch.processing as processing
import juneberry.pytorch.utils as pyt_utils
from juneberry.transform_manager import TransformManager
import juneberry.utils

logger = logging.getLogger(__name__)


class PytorchEvaluator(Evaluator):
    """
    This subclass is the Pytorch-specific version of the Evaluator.
    """

    def __init__(self, model_config: ModelConfig, lab: Lab, dataset: jb_dataset.DatasetConfig,
                 model_manager: ModelManager, eval_dir_mgr: EvalDirMgr, eval_options: SimpleNamespace = None):
        """
        Creates a PytorchEvaluator object based on command line arguments and a Juneberry
        ModelManager object.
        :param model_config: The model config to be used during evaluation.
        :param lab: The Juneberry lab describing the current execution.
        :param dataset: A Juneberry DatasetConfig object representing the dataset to be evaluated.
        :param model_manager: A Juneberry ModelManager object responsible for managing operations involving the
        model to be evaluated.
        :param eval_dir_mgr: A Juneberry EvalDirMgr object responsible for managing file path operations
        within the model's eval directory.
        :param eval_options: A SimpleNamespace containing various options for the evaluation. Expected options
        include the following: topK, dryrun, extract_validation.
        """
        super().__init__(model_config, lab, dataset, model_manager, eval_dir_mgr, eval_options)

        # These attributes are used by Pytorch to send information to the correct device (CPU | GPU)
        self.use_cuda = False
        self.device = None
        self.device_ids = []

        # These attributes relate to Pytorch's way of loading data from a dataloader.
        self.eval_loader = None

    def check_gpu_availability(self, required: int):
        count = processing.determine_gpus(required)
        # TODO: Test to see if we can use more than 1 GPU with DP.
        if count > 1:
            logger.warning(f"The evaluator is only configured to support 1 GPU. Reducing {count} to 1.")
            count = 1
        return count

    def setup(self) -> None:
        """
        This is the Pytorch version of the extension point that's responsible for setting up the Evaluator.
        :return: Nothing.
        """
        logger.info(f"Performing Pytorch setup steps...")

        # Check if cuda is available; set the appropriate "default" device.
        if self.num_gpus == 0:
            self.use_cuda = False
            self.device = torch.device("cpu")
            self.device_ids = []
        elif self.num_gpus == 1:
            self.use_cuda = True
            self.device = torch.device("cuda:0")
            self.device_ids = [0]
        else:
            # We should NEVER get here because the GPU availability should never return more than one.
            logger.error("PyTorch Evaluator does NOT support more than one GPU device. EXITING.")
            sys.exit(-1)

        # Read the evaluation methods from the ModelConfig.
        self.eval_method = self.model_config.evaluation_procedure
        self.eval_output_method = self.model_config.evaluation_output

        # TODO: Shouldn't this be done in the lab??

        if self.model_config.hints is not None and 'num_workers' in self.model_config.hints.keys():
            num_workers = self.model_config.hints.num_workers
            logger.warning(f"Overriding number of workers. Found {num_workers} in ModelConfig")
            self.lab.num_workers = num_workers

        # Set the seeds using the value from the ModelConfig.
        pyt_utils.set_pytorch_seeds(self.model_config.seed)

        logger.info(f"PyTorch setup steps are complete.")

    def obtain_dataset(self) -> None:
        """
        This is the PyTorch version of the extension point that's responsible for obtaining the dataset
        being used in the evaluation.
        :return: Nothing.
        """
        # Create the dataloader and data list for the evaluation data.

        if self.eval_dataset_config.data_type == jb_dataset.DataType.TORCHVISION:
            logger.info(f"Creating EVALUATION dataloader and list of EVALUATION files.")

            tv_data = self.eval_dataset_config.torchvision_data
            data_transforms = None
            target_transforms = None
            if self.model_config.evaluation_transforms:
                data_transforms = TransformManager(self.model_config.evaluation_transforms)
            if self.model_config.evaluation_target_transforms:
                target_transforms = TransformManager(self.model_config.evaluation_target_transforms)
            val_dataset = pyt_utils.construct_torchvision_dataset(
                self.lab, tv_data.fqcn, tv_data.root, tv_data.eval_kwargs,
                data_transforms=data_transforms,
                target_transforms=target_transforms)

            # We don't really have names for these so we just us the number.
            self.eval_name_targets = []
            for i, v in enumerate(val_dataset.targets):
                if isinstance(v, Tensor):
                    self.eval_name_targets.append([i, v.item()])
                else:
                    self.eval_name_targets.append([i, int(v)])

            # NOTE: We do NOT shuffle the data here because it HAS to match the order from above
            self.eval_loader = pyt_utils.wrap_dataset_in_dataloader(self.lab, val_dataset, self.model_config.batch_size)

        else:
            logger.info(f"Creating EVALUATION dataloader and list of EVALUATION files")

            splitting_config = None
            if self.use_train_split or self.use_val_split:
                logger.info(f"Splitting the dataset according to the model's validation split instructions.")
                splitting_config = self.model_config.get_validation_split_config()

            eval_list, split = jbdata.dataspec_to_manifests(self.lab,
                                                            dataset_config=self.eval_dataset_config,
                                                            splitting_config=splitting_config,
                                                            preprocessors=TransformManager(
                                                                self.model_config.preprocessors))

            if self.use_train_split:
                logger.info("Evaluating using ONLY the training portion of the split data.")

            elif self.use_val_split:
                logger.info("Evaluating using ONLY the validation portion of the split data.")
                eval_list = split

            # The eval list is the already a list of name and targets
            self.eval_name_targets = eval_list

            self.eval_loader = pyt_utils.make_data_loader(self.lab,
                                                          self.eval_dataset_config,
                                                          eval_list,
                                                          self.model_config.evaluation_transforms,
                                                          self.model_config.batch_size)

        logger.info(f"EVALUATION dataloader created.")
        logger.info(f"There are {len(self.eval_name_targets)} pieces of data in the evaluation list.")

    def obtain_model(self) -> None:
        """
        This is the Pytorch version of the extension point that's responsible for obtaining the model
        to be evaluated.
        :return: Nothing.
        """
        logger.info(f"Building the model for EVALUATION...")

        # Construct the model using the architecture and the number of classes.
        # TODO: If model_mapping becomes a thing, should this change to be the number of classes that the
        #  model is aware of?
        logger.info(f"Constructing the model...")
        self.model = pyt_utils.construct_model(self.model_config.model_architecture,
                                               self.eval_dataset_config.num_model_classes)

        # If the ModelConfig contains model transforms, apply them to the model.
        logger.info(f"Checking for model transforms...")
        if self.model_config.model_transforms is not None:
            transforms = TransformManager(self.model_config.model_transforms)
            transforms.transform(self.model)
            logger.info(f"Successfully applied transforms to the model.")
        else:
            logger.info(f"Model config does not contain model transforms. Skipping model transform application.")

        # Load the weights into the model.
        logger.info(f"Loading model weights.")
        pyt_utils.load_model(self.model_manager.get_pytorch_model_path(), self.model)

        # If a GPU is present, wrap the model in DataParallel.
        if self.use_cuda:
            logger.info(f"Detected CUDA. Wrapping the model in DataParallel.")
            self.model = torch.nn.DataParallel(self.model, device_ids=self.device_ids)

        # Send the model to the device.
        logger.info(f"Sending the model to device - {self.device}")
        self.model.to(self.device)

        logger.info(f"Model has been constructed for EVALUATION.")

    def evaluate_data(self) -> None:
        """
        This is the Pytorch version of the extension point that's responsible for feeding the evaluation
        dataset into the model and obtaining the raw evaluation data. That process is usually defined in some
        external method, usually found in juneberry.pytorch.evaluation.
        :return: Nothing.
        """

        if self.dryrun:
            logger.info(f"Dry run complete.")
            sys.exit(0)

        logger.info(f"Generating EVALUATION data according to {self.eval_method}")
        logger.info(f"Will evaluate model {self.model_manager.model_name} using {self.eval_dataset_config_path}")

        juneberry.utils.invoke_evaluator_method(self, self.eval_method)

        logger.info(f"EVALUATION COMPLETE.")

    def format_evaluation(self) -> None:
        """
        This is the Pytorch version of the extension point that's responsible for converting the raw
        evaluation data into the format the user wants. Much like evaluate_data, the actual process is
        usually defined in some external method, typically found in juneberry.pytorch.evaluation.
        :return:
        """
        logger.info(f"Formatting raw EVALUATION data according to {self.eval_output_method}")

        juneberry.utils.invoke_evaluator_method(self, self.eval_output_method)
