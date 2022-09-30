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

from pathlib import Path
import torch

import juneberry.config.dataset as jb_dataset
from juneberry.config.dataset import DatasetConfig
from juneberry.config.eval_output import EvaluationOutput
from juneberry.config.model import ModelConfig, Plugin
import juneberry.data as jb_data
from juneberry.evaluation.evaluator import EvaluatorBase
from juneberry.filesystem import EvalDirMgr, ModelManager
from juneberry.lab import Lab
import juneberry.pytorch.data as pyt_data
import juneberry.pytorch.processing as processing
import juneberry.pytorch.utils as pyt_utils
from juneberry.pytorch.utils import PyTorchPlatformDefinitions
from juneberry.transforms.transform_manager import TransformManager
import juneberry.zoo as jb_zoo

logger = logging.getLogger(__name__)


class Evaluator(EvaluatorBase):
    """
    This subclass is the Pytorch-specific version of the Evaluator.
    """

    def __init__(self, model_config: ModelConfig, lab: Lab, model_manager: ModelManager, eval_dir_mgr: EvalDirMgr,
                 dataset: DatasetConfig, eval_options: SimpleNamespace = None, log_file: str = None):
        """
        Creates an Evaluator object based on command line arguments and a Juneberry
        ModelManager object.
        :param model_config: The model config to be used during evaluation.
        :param lab: The Juneberry lab describing the current execution.
        :param model_manager: A Juneberry ModelManager object responsible for managing operations involving the
        model to be evaluated.
        :param eval_dir_mgr: A Juneberry EvalDirMgr object responsible for managing file path operations
        within the model's eval directory.
        :param dataset: A Juneberry DatasetConfig object representing the dataset to be evaluated.
        :param eval_options: A SimpleNamespace containing various options for the evaluation. Expected options
        include the following: topK, use_train_split, use_val_split.
        :param log_file: A string indicating the location of the current log file.
        """
        super().__init__(model_config, lab, model_manager, eval_dir_mgr, dataset, eval_options, log_file)

        # These attributes are used by Pytorch to send information to the correct device (CPU | GPU)
        self.use_cuda = False
        self.device = None
        self.device_ids = []

        # These attributes relate to Pytorch's way of loading data from a dataloader.
        self.eval_loader = None

        if not self.metrics_plugins:
            self.metrics_plugins = _get_default_metrics_plugins()

    # ==========================================================================

    @classmethod
    def get_platform_defs(cls):
        return PyTorchPlatformDefinitions()

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
                    eval_dir_mgr.get_dryrun_imgs_dir(),
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
        return eval_data.results.metrics.classification["balanced_accuracy"], "balanced_accuracy"

    # ==========================================================================
    def dry_run(self) -> None:
        self.dryrun = True
        self.setup()
        self.obtain_dataset()

        # Write out a few dry run images
        if self.eval_dataset_config.is_image_type():
            _ = pyt_utils.generate_sample_images(self.eval_loader, 5, Path(self.eval_dir_mgr.get_dryrun_imgs_dir()))

        self.obtain_model()

        logger.info(f"Dryrun complete.")

    # ==========================================================================

    def check_gpu_availability(self, required: int):
        count = processing.determine_gpus(required)
        # TODO: Test to see if we can use more than 1 GPU with DP.
        if count > 1:
            logger.warning(f"The evaluator is only configured to support 1 GPU. Reducing {count} to 1.")
            count = 1
        return count

    def setup(self) -> None:
        """
        This is the PyTorch version of the extension point that's responsible for setting up the Evaluator.
        :return: Nothing.
        """
        logger.info(f"Performing PyTorch setup steps...")

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
            logger.error("PyTorch Evaluator does NOT support more than one GPU device. Exiting.")
            sys.exit(-1)

        # Set the seeds using the value from the ModelConfig.
        pyt_utils.set_pytorch_seeds(self.model_config.seed)

        # Use default values if they were not provided in the model config.
        if self.eval_method is None:
            self.eval_method = "juneberry.pytorch.evaluation.default.PyTorchEvaluationProcedure"
        if self.eval_output_method is None:
            self.eval_output_method = "juneberry.pytorch.evaluation.default.PyTorchEvaluationOutput"

        logger.info(f"PyTorch Evaluator setup steps are complete.")

    def reset(self) -> None:
        # Set the seeds using the value from the ModelConfig.
        pyt_utils.set_pytorch_seeds(self.model_config.seed)

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
            val_dataset = pyt_data.construct_torchvision_dataset(
                self.lab, tv_data.fqcn, tv_data.root, tv_data.eval_kwargs,
                data_transforms=data_transforms,
                target_transforms=target_transforms)

            # We don't really have names for these so we just us the number.
            self.eval_name_targets = []
            for i, v in enumerate(val_dataset.targets):
                if isinstance(v, torch.Tensor):
                    self.eval_name_targets.append([i, v.item()])
                else:
                    self.eval_name_targets.append([i, int(v)])

            # NOTE: We do NOT shuffle the data here because it HAS to match the order from above
            self.eval_loader = pyt_data.wrap_dataset_in_dataloader(self.lab, val_dataset, self.model_config.batch_size)

        else:
            logger.info(f"Creating EVALUATION dataloader and list of EVALUATION files")

            splitting_config = None
            if self.use_train_split or self.use_val_split:
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

            # The eval list is the already a list of name and targets
            self.eval_name_targets = eval_list

            logger.info(f"...shuffling manifest with seed {self.model_config.seed}...")
            jb_data.shuffle_manifests(self.model_config.seed, eval_list)

            # Save the manifest
            if self.eval_dataset_config.is_image_type():
                eval_manifest_path = self.eval_dir_mgr.get_manifest_path()
                logger.info(f"...saving eval manifest to {eval_manifest_path}")
                jb_data.save_path_label_manifest(eval_list, eval_manifest_path, self.lab.data_root())

            logger.info(f"...making data loaders...")
            self.eval_loader = pyt_data.make_eval_data_loader(self.lab, self.eval_dataset_config, self.model_config,
                                                              eval_list)

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
            self.model = transforms.transform(self.model)
            logger.info(f"Successfully applied transforms to the model.")
        else:
            logger.info(f"Model config does not contain model transforms. Skipping model transform application.")

        # Identify the model file.
        model_path = self.model_manager.get_model_path(PyTorchPlatformDefinitions())

        # If the model file exists, load the weights.
        if model_path.exists():
            # Check to see if we can load it.
            image_shape = pyt_utils.get_image_shape(self.eval_loader)
            if not jb_zoo.check_allow_load_model(self.model_manager,
                                                 lambda: pyt_utils.hash_summary(self.model, image_shape)):
                logger.error("Cannot load model because of ARCHITECTURE hash mismatch. "
                             "Either delete the hash, retrain or get the correct model. Exiting.")
                raise RuntimeError("Model architecture does not match hash. See log for details.")

            logger.info(f"Loading model weights...")
            self.model = pyt_utils.load_model(model_path, self.model, self.model_config.pytorch.strict)

        # If the model file doesn't exist...
        else:
            # A missing model file is not a big deal for a dryrun, just inform that the weights
            # could not be loaded.
            if self.dryrun:
                logger.info(f"Did not load model weights. {model_path} does not exist.")
                return

            # If there's no model file and it's not a dryrun, then there's still a chance to get model
            # weights if the model architecture specifies a pretrained model. So log a warning and attempt
            # to proceed.
            else:
                logger.warning(f"No 'model.pt' found, running with default model produced from model architecture. "
                               f"Expected to find: {model_path}")

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
        logger.info(f"Generating EVALUATION data according to {self.eval_method}")
        logger.info(f"Will evaluate model {self.model_manager.model_name} using {self.eval_dataset_config_path}")

        pyt_utils.invoke_evaluator_method(self, self.eval_method)

        logger.info(f"EVALUATION COMPLETE")

    def format_evaluation(self) -> None:
        """
        This is the Pytorch version of the extension point that's responsible for converting the raw
        evaluation data into the format the user wants. Much like evaluate_data, the actual process is
        usually defined in some external method, typically found in juneberry.pytorch.evaluation.
        :return:
        """
        logger.info(f"Formatting raw EVALUATION data according to {self.eval_output_method}")

        pyt_utils.invoke_evaluator_method(self, self.eval_output_method)


def _get_default_metrics_plugins():
    result = []
    evaluation_metrics = [
        {
            "fqcn": "juneberry.metrics.classification.sklearn.metrics.Metrics",
            "kwargs": {
                "fqn": "sklearn.metrics.accuracy_score",
                "name": "accuracy",
                "kwargs": {
                    "sample_weight": None,
                    "normalize": True
                }
            }
        },
        {
            "fqcn": "juneberry.metrics.classification.sklearn.metrics.Metrics",
            "kwargs": {
                "fqn": "sklearn.metrics.balanced_accuracy_score",
                "name": "balanced_accuracy",
                "kwargs": {
                    "sample_weight": None,
                    "adjusted": True
                }
            }
        }
    ]
    for metric in evaluation_metrics:
        result.append(Plugin.from_dict(metric))
    return result
