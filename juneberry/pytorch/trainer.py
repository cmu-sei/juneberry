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
import math
import os
import sys
from typing import Union

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

import juneberry.config.dataset as jb_dataset
from juneberry.config.model import LRStepFrequency, PytorchOptions, StoppingCriteria
import juneberry.data as jb_data
import juneberry.filesystem as jb_fs
from juneberry.logging import setup_logger
import juneberry.metrics.classification.metrics_manager as mm
from juneberry.onnx.utils import ONNXPlatformDefinitions
import juneberry.plotting
from juneberry.pytorch.acceptance_checker import AcceptanceChecker
import juneberry.pytorch.data as pyt_data
import juneberry.pytorch.processing as processing
import juneberry.pytorch.utils as pyt_utils
from juneberry.pytorch.utils import PyTorchPlatformDefinitions
import juneberry.tensorboard as jb_tb
from juneberry.training.trainer import EpochTrainer
from juneberry.transforms.transform_manager import TransformManager
import juneberry.zoo as jb_zoo

logger = logging.getLogger(__name__)


class ClassifierTrainer(EpochTrainer):
    def __init__(self, lab, model_manager, model_config, dataset_config, log_level):
        super().__init__(lab, model_manager, model_config, dataset_config, log_level)

        # Assigned during setup
        self.loss_function = None
        self.lr_scheduler = None
        self.optimizer = None
        self.evaluator = None
        self.acceptance_checker = None

        # We should probably be given a data manager
        self.binary = dataset_config.is_binary
        self.pytorch_options: PytorchOptions = model_config.pytorch

        # A single sample from the input data. The dimensions of the sample matter during
        # construction of the ONNX model.
        self.input_sample = None

        # Tensorboard manager
        self.tb_mgr = None

        # This model is for saving
        self.unwrapped_model = None

        # This model is for training
        self.model = None

        # Where we store all the in-flight results
        self.history = {}

        # This is the pytorch device we are associated with
        self.device = None

        self.num_batches = -1

        self.memory_summary_freq = int(os.environ.get("JUNEBERRY_CUDA_MEMORY_SUMMARY_PERIOD", 0))

        # These properties are used for DistributedDataParallel (if necessary)
        self.training_loss_list = None
        self.training_metrics_lists = {}

        self.lr_step_frequency = LRStepFrequency.EPOCH

        # Added for the acceptance checker
        self.history_key = None
        self.direction = None
        self.abs_tol = None

    # ==========================================================================

    @classmethod
    def get_platform_defs(cls):
        return PyTorchPlatformDefinitions()

    # ==========================================================================

    @classmethod
    def get_training_output_files(cls, model_mgr: jb_fs.ModelManager, dryrun: bool = False):
        """
        Returns a list of files to clean from the training directory. This list should contain ONLY
        files or directories that were produced by the training command. Directories in this list
        will be deleted even if they are not empty.
        :param model_mgr: A ModelManager to help locate files.
        :param dryrun: When True, returns a list of files created during a dryrun of the Trainer.
        :return: The files to clean from the training directory.
        """
        if dryrun:
            return [model_mgr.get_model_summary_path(),
                    model_mgr.get_dryrun_imgs_dir(),
                    model_mgr.get_training_data_manifest_path(),
                    model_mgr.get_validation_data_manifest_path()]
        else:
            return [model_mgr.get_model_path(cls.get_platform_defs()),
                    model_mgr.get_model_path(ONNXPlatformDefinitions()),
                    model_mgr.get_training_out_file(),
                    model_mgr.get_training_summary_plot(),
                    model_mgr.get_training_data_manifest_path(),
                    model_mgr.get_validation_data_manifest_path()]

    @classmethod
    def get_training_clean_extras(cls, model_mgr: jb_fs.ModelManager, dryrun: bool = False):
        """
        Returns a list of extra "training" files/directories to clean. Directories in this list will NOT
        be deleted if they are not empty.
        :param model_mgr: A ModelManager to help locate files.
        :param dryrun: When True, returns a list of files created during a dryrun of the Trainer.
        :return: The extra files to clean from the training directory.
        """
        if dryrun:
            return [model_mgr.get_train_root_dir()]
        else:
            return [model_mgr.get_train_root_dir()]

    # ==========================================================================

    def dry_run(self) -> None:
        # Setup is the same for dry run
        self.setup()

        summary_path = self.model_manager.get_model_summary_path()
        if self.dataset_config.is_image_type():
            # Save some sample images to verify augmentations
            image_shape = pyt_utils.generate_sample_images(self.training_iterable, 5,
                                                           self.model_manager.get_dryrun_imgs_dir())
            pyt_utils.output_summary_file(self.model, image_shape, summary_path)

        elif self.dataset_config.is_tabular_type():
            # TODO Emit sample row modified data
            data, labels = next(iter(self.training_iterable))
            pyt_utils.output_summary_file(self.model, data[0].shape, summary_path)

        else:
            logger.error("Dry run doesn't support anything beyond IMAGE or TABULAR type. Exiting.")

    # ==========================================================================

    def establish_loggers(self) -> None:

        # In a distributed training situation, the root Juneberry logger must be set up again for each
        # rank process. In non-distributed training, the Trainer can continue to use the root Juneberry
        # logger that was set up earlier, thus no additional actions are required.
        if self.distributed:
            setup_logger(self.model_manager.get_training_log(), "", dist_rank=self.gpu, level=self.log_level)

    def setup(self):

        # Construct helper objects
        if self.lab.tensorboard:
            self.tb_mgr = jb_tb.TensorBoardManager(self.lab.tensorboard, self.model_manager)

        pyt_utils.set_pytorch_seeds(self.model_config.seed)

        self.setup_hardware()
        self.setup_data_loaders()
        self.setup_model()

        self.loss_function = pyt_utils.make_loss(self.pytorch_options, self.model, self.binary)
        self.optimizer = pyt_utils.make_optimizer(self.pytorch_options, self.model)
        self.lr_scheduler = pyt_utils.make_lr_scheduler(self.pytorch_options, self.optimizer, self.model_config.epochs)
        self.setup_acceptance_checker()
        if self.pytorch_options.lr_step_frequency == LRStepFrequency.BATCH:
            self.lr_step_frequency = LRStepFrequency.BATCH

        self.num_batches = len(self.training_iterable)

        self.history = {'loss': [], 'val_loss': []}

        # Create an entry in the history for each metrics plugin in the config.
        # TODO what if one of our metrics is called "loss?" That will step on the already
        #  existing "loss" entry in self.history.
        #  For now, log an error if "loss" is specified in the training_metrics section.
        for plugin in self.metrics_plugins:
            self.history[plugin["kwargs"]["name"]] = []
            self.history["val_" + plugin["kwargs"]["name"]] = []

        self.history['epoch_duration'] = []
        self.history['lr'] = []

    def finish(self):
        super().finish()
        if self.tb_mgr is not None:
            self.tb_mgr.close()
            self.tb_mgr = None

    # ==========================================================================

    def start_epoch_phase(self, train: bool):
        result = {}

        if train:
            self.model.train()
            torch.set_grad_enabled(True)
            # If our datasets understand epochs, then tell them.
            if isinstance(self.training_iterable.dataset, pyt_utils.EpochDataset):
                self.training_iterable.dataset.set_epoch(self.epoch)
        else:
            self.model.eval()
            torch.set_grad_enabled(False)
            # If our datasets understand epochs, then tell them.
            if isinstance(self.evaluation_iterable.dataset, pyt_utils.EpochDataset):
                self.evaluation_iterable.dataset.set_epoch(self.epoch)

        if self.distributed:
            self.training_loss_list = [torch.Tensor(1).cuda() for i in range(self.num_gpus)]
            # TODO: Unlike when we were only storing accuracy, I don't know that we can automatically
            #  initialize to float64 tensors for every kind of metric
            for plugin in self.metrics_plugins:
                self.training_metrics_lists[plugin["kwargs"]["name"] + "_list"] = \
                    [torch.zeros(1, dtype=torch.float64).cuda() for i in range(self.num_gpus)]

        result["loss_list"] = []

        # Start off with empty metrics
        for plugin in self.metrics_plugins:
            result[plugin["kwargs"]["name"] + "_list"] = []

        return result

    def process_batch(self, train: bool, data, targets):

        # Move the data to the device
        local_batch, local_labels = data.to(self.device), targets.to(self.device)

        # Forward pass: Pass in the batch of images for it to do its thing
        output = self.model(local_batch)

        # Compute and store loss based on the provided function
        loss = self.loss_function(output, local_labels)

        # Compute and store metrics based on metrics plugin functions
        metrics_mgr = mm.MetricsManager(self.metrics_plugins)
        preds_np, target_np = _tensors_to_numpy(output, local_labels)
        metrics = metrics_mgr(target_np, preds_np, self.dataset_config.is_binary)

        return loss, metrics

    def update_metrics(self, train: bool, metrics, results) -> None:
        # Unpack the loss and metrics we returned on process batch
        current_loss, current_metrics = results

        # In distributed training, each process will have a different loss/accuracy value. These lists are used to
        # collect the values from each process, so we need one tensor in the list for every process in the "world".
        if self.distributed:
            tensor_metrics_results = {}
            # TODO Assuming data type (float) for an unknown metric
            for k, v in current_metrics.items():
                tensor_metrics_results[k] = torch.from_numpy(np.asarray(current_metrics[k], dtype=float)).to(self.device)

            loss_on_device = current_loss.to(self.device)

            # Create a barrier to wait for all processes to reach this point. Once they do, gather up
            # the loss and accuracy from each process and place it in the appropriate tensor list.
            dist.barrier()
            dist.all_gather(self.training_loss_list, loss_on_device)
            for k, v in tensor_metrics_results.items():
                dist.all_gather(self.training_metrics_lists[k], tensor_metrics_results[k])

            # Take the value from each tensor in the tensor list and place it in the corresponding metric.
            for tensor in self.training_loss_list:
                metrics['loss_list'].append(tensor.item())
            for k, v in self.training_metrics_lists.items():
                for val in self.training_metrics_lists[k]:
                    metrics[f"{k}_list"].append(val.item())
            return

        # Record the values in the metrics dictionary (non-distributed case).
        metrics["loss_list"].append(current_loss.item())
        for k, v in current_metrics.items():
            metrics[f"{k}_list"].append(current_metrics[k].item())

    def update_model(self, results) -> None:
        # Unpack the results we returned on process batch
        # We don't need the metrics here
        loss, _ = results

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.lr_scheduler is not None and self.lr_step_frequency == LRStepFrequency.BATCH:
            self.lr_scheduler.step()

    def summarize_metrics(self, train, metrics) -> None:
        for k, v in metrics.items():
            history_key = k[:-5] # strip off the ending "_list" in key
            if not train:
                history_key = f"val_{history_key}"
            self.history[history_key].append(float(np.mean(metrics[k])))

    def end_epoch(self, tuning_mode: bool = False) -> dict:
        if self.lr_scheduler is not None:
            if self.lr_step_frequency == LRStepFrequency.EPOCH:
                self.lr_scheduler.step()
            # TODO: Should we try to average learning rate for reporting here?
            self.history['lr'].append(self.lr_scheduler.get_last_lr()[0])
        else:
            for param_group in self.optimizer.param_groups:
                self.history['lr'].append(param_group['lr'])

        # Pass the model and value we want to check to the acceptance checker
        allow_save = (self.gpu is None or self.gpu == 0) and not tuning_mode
        self.done = self.acceptance_checker.add_checkpoint(self.unwrapped_model, self.input_sample,
                                                           self.history[self.history_key][-1],
                                                           allow_save=allow_save)

        # TODO: Check if loss when nan.

        # Capture the data for TensorBoard (if necessary)
        if self.tb_mgr is not None and not tuning_mode:
            self.tb_mgr.update(self.history, self.epoch - 1)

        self.show_memory_summary(False)

        # Construct the dictionary of metrics and return the result.
        return_val = {}
        for x in self.history:
            if len(self.history[x]) > 0:
                return_val[x] = self.history[x][-1]

        return return_val

    def summarize_epoch(self) -> str:
        metric_str = ""
        metrics_history, non_metrics_history = _separate_metrics_history(self.history)
        for x in metrics_history:
            if metrics_history[x] and len(metrics_history[x]) > 0:
                metric_str += f"{x}: {metrics_history[x][-1]:.4f}, "
        for x in non_metrics_history:
            if non_metrics_history[x] and len(non_metrics_history[x]) > 0:
                metric_str += f"{x}: {non_metrics_history[x][-1]:.2E}, "

        return metric_str[:-2]  # remove trailing ", "

    def finalize_results(self) -> None:
        # If we're in distributed mode, only one process needs to perform these actions (since all processes should
        # have the same model).
        if self.distributed and not self.gpu == 0:
            return

        logger.info(f"Training stopped because: >> {self.acceptance_checker.stop_message} <<")

        # Add a hash of the model.
        if self.native:
            model_path = self.model_manager.get_model_path(self.get_platform_defs())
            self.history['model_hash'] = jb_fs.generate_file_hash(model_path)

        if self.onnx:
            model_path = self.model_manager.get_model_path(ONNXPlatformDefinitions())
            self.history['onnx_model_hash'] = jb_fs.generate_file_hash(model_path)

        logger.info("Generating and saving output...")
        history_to_results(self.history, self.results, self.native, self.onnx)

        logger.info("Generating summary plot...")
        juneberry.plotting.plot_training_summary_chart(self.results, self.model_manager)

        logger.info("Updating model_architecture hash...")
        self._updated_hashes()

    # ==========================================================================

    def check_gpu_availability(self, required: int = None):
        return processing.determine_gpus(required)

    def train_distributed(self, num_gpus) -> None:
        # This call initializes this base object before we spawn multiple processes
        # which get copies.  For the most part, everything can come through via this
        # object except we use the environment variables for the address and port
        # as is traditional.
        self.distributed = True
        self.num_gpus = num_gpus

        # Setup the hardware (cuda/multiprocessing) for distributed
        processing.prepare_for_distributed()

        # Use the number of GPUs detected and adjust the batch size so that the batch size
        # specified in the config is evenly distributed among all processes in the "world".
        # TODO: Inject learning rate scaling code
        new_batch_size = int(self.model_config.batch_size / self.num_gpus)
        if self.model_config.batch_size != new_batch_size:
            logger.info(f"Adjusting batch size from {self.model_config.batch_size} "
                        f"to {new_batch_size} for distributed training...")
            self.model_config.batch_size = new_batch_size
            logger.warning("!!! NOT ADJUSTING LEARNING RATE")

        # Start up the processes
        processing.start_distributed(self.train_model, self.num_gpus)

    # ==========================================================================

    #  _____      _
    # /  ___|    | |
    # \ `--.  ___| |_ _   _ _ __
    #  `--. \/ _ \ __| | | | '_ \
    # /\__/ /  __/ |_| |_| | |_) |
    # \____/ \___|\__|\__,_| .__/
    #                      | |
    #                      |_|

    def setup_hardware(self):
        self.device = processing.setup_cuda_device(self.num_gpus, self.gpu)
        if self.distributed:
            processing.setup_distributed(self.num_gpus, self.gpu)
        processing.log_cuda_configuration(self.num_gpus, self.gpu, logger)

        # These two options must be set in order to achieve reproducibility.
        if self.model_config.pytorch.get("deterministic", False):
            cudnn.deterministic = True
            cudnn.benchmark = False

    def setup_data_loaders(self):
        logger.info(f"Preparing data loaders...")

        # If we're doing distributed training, then we need to set up a sampler for the data loader to make
        # sure individual processes don't use the same images as input.
        sampler_args = (self.num_gpus, self.gpu) if self.distributed else None

        if self.dataset_config.data_type == jb_dataset.DataType.TORCHVISION:
            if self.model_config.validation is not None:
                logger.warning("Using a Torchvision Dataset. Ignoring validation split.")

            self.training_iterable, self.evaluation_iterable = pyt_data.construct_torchvision_dataloaders(
                self.lab, self.dataset_config.torchvision_data, self.model_config,
                self.dataset_config.get_sampling_config(),
                sampler_args=sampler_args)

        else:
            logger.info(f"...converting dataspec to manifests...")
            train_list, val_list = jb_data.dataspec_to_manifests(
                self.lab,
                dataset_config=self.dataset_config,
                splitting_config=self.model_config.get_validation_split_config(),
                preprocessors=TransformManager(self.model_config.preprocessors))

            # Shuffle the data sets
            logger.info(f"...shuffling manifests with seed {self.model_config.seed} ...")
            jb_data.shuffle_manifests(self.model_config.seed, train_list, val_list)

            if self.dataset_config.is_image_type():
                # Save the manifest files for traceability
                logger.info(f"...saving manifests to disk...")
                train_manifest_path = self.lab.workspace() / self.model_manager.get_training_data_manifest_path()
                val_manifest_path = self.lab.workspace() / self.model_manager.get_validation_data_manifest_path()
                jb_data.save_path_label_manifest(train_list, train_manifest_path, self.lab.data_root())
                jb_data.save_path_label_manifest(val_list, val_manifest_path, self.lab.data_root())

            logger.info(f"...making data loaders...")
            self.training_iterable, self.evaluation_iterable = \
                pyt_data.make_training_data_loaders(self.lab,
                                                    self.dataset_config,
                                                    self.model_config,
                                                    train_list,
                                                    val_list,
                                                    sampler_args=sampler_args)

            # Sample a single item from the input dataset.
            dataset = self.training_iterable.dataset
            if len(dataset) > 0:
                self.input_sample, label = dataset[0]

                # If the input sample is a numpy array, convert to a tensor.
                if type(self.input_sample) == np.ndarray:
                    self.input_sample = torch.from_numpy(self.input_sample)

                # Set the input sample as the data from the tensor and send it to a training device.
                self.input_sample = self.input_sample.unsqueeze(0).to(self.device)

    def setup_model(self):
        logger.info(f"Constructing the model {self.model_config.model_architecture.fqcn} "
                    f"with args: {self.model_config.model_architecture.kwargs} ...")
        self.model = pyt_utils.construct_model(self.model_config.model_architecture,
                                               self.dataset_config.num_model_classes)

        # If this model is based off another model, then load its weights.
        previous_model = self.model_config.get_previous_model()
        if previous_model is not None:
            logger.info(f"Loading weights from previous model: {previous_model}")

            prev_model_manager = jb_fs.ModelManager(previous_model)

            pyt_utils.load_weights_from_model(prev_model_manager, self.model, self.model_config.pytorch.strict)

        # Apply model transforms.
        if self.model_config.model_transforms is not None:
            transforms = TransformManager(self.model_config.model_transforms)
            self.model = transforms.transform(self.model)

        # Save off a reference to the unwrapped model for saving.
        self.unwrapped_model = self.model

        # Prepare the model for cuda and/or distributed use.
        self.model = processing.prepare_model(self.distributed, self.num_gpus, self.gpu, self.model, self.device)
        self.show_memory_summary(True)

    def setup_acceptance_checker(self) -> None:
        """
        Creates an acceptance checker based on the parameters in the training config
        """
        stopping_options = self.model_config.stopping_criteria
        if stopping_options is None:
            stopping_options = StoppingCriteria()
        self.history_key = stopping_options.history_key
        self.direction = stopping_options.direction
        self.abs_tol = stopping_options.abs_tol
        logger.info(
            f"Adding '{self.history_key}' '{self.direction}' with tolerance '{self.abs_tol}' "
            f"as the acceptance checking condition.")

        self.acceptance_checker = AcceptanceChecker(self.model_manager,
                                                    comparator=lambda x, y: self.acceptance_comparator(x, y),
                                                    max_epochs=self.max_epochs,
                                                    threshold=stopping_options.get('threshold', None),
                                                    plateau_count=stopping_options.get('plateau_count', None))

        self.acceptance_checker.native = self.native
        self.acceptance_checker.onnx = self.onnx

    def show_memory_summary(self, model_loading):
        """Used to show a memory summary at appropriate times."""
        if not self.gpu or self.memory_summary_freq == 0:
            return

        if model_loading or self.epoch == 1 or (self.epoch - 1) % self.memory_summary_freq == 0:
            if model_loading:
                logger.info(f"CUDA memory summary after model load")
            else:
                logger.info(f"CUDA memory summary for epoch {self.epoch}")
            logger.info(torch.cuda.memory_summary(self.device))

    def acceptance_comparator(self, x, y):
        if math.isclose(x, y, abs_tol=self.abs_tol):
            return 0
        elif self.direction == 'ge':
            return x - y
        elif self.direction == 'le':
            return y - x
        else:
            logger.error(f"acceptance_comparator requires a direction of 'ge' or 'le'.")
            sys.exit(-1)

    # ==========================================================================

    def _updated_hashes(self):
        # If we have an existing hash file, update it.
        # Always update 'latest' as it is what gets used when packaging a model for the zoo.
        image_shape = pyt_utils.get_image_shape(self.training_iterable)
        model_arch_hash = pyt_utils.hash_summary(self.model, image_shape)
        jb_zoo.update_hashes_after_training(self.model_manager, model_arch_hash)


# ==================================================================================================


def compute_preliminary_eta(num_batches, batch_mean, epoch_start, max_epochs, validation_scale):
    # We are going to make a preliminary estimate based on some set of training batches.
    # We need to scale this based on the size of the validation set. NOTE, this will be
    # WRONG because the validation size doesn't include back propagation.
    epoch_duration = batch_mean * num_batches
    total_duration = epoch_duration * max_epochs * validation_scale
    eta = epoch_start + datetime.timedelta(seconds=total_duration)

    logger.info(f"PRELIMINARY ROUGH Estimate of epoch duration {epoch_duration:.3f} seconds, "
                f"total ETA {eta.strftime('%H:%M:%S')} ")


def history_to_results(history, results, native, onnx):
    """
    Places our history into the results for final output. (Uses JSON style.)
    :param history: A history of the training
    :param results: Where to store the information so it can be retrieved when constructing the final output.
    :param native: A Boolean controlling whether or not to include the hash of the native PyTorch model.
    :param onnx: A Boolean controlling whether or not to include the hash of the ONNX model.
    """
    # The learning rate can change over time...
    results['options']['learning_rate'] = history['lr']

    if native:
        results['results']['model_hash'] = history['model_hash']
    if onnx:
        results['results']['onnx_model_hash'] = history['onnx_model_hash']

    metrics_history, _ = _separate_metrics_history(history)
    for k in metrics_history.keys():
        results['results'][k] = metrics_history[k]

def _separate_metrics_history(history):
    non_metrics_keys = {"lr", "model_hash", "onnx_model_hash", "epoch_duration"}
    metrics_keys = history.keys() - non_metrics_keys
    metrics_history = {k: history.get(k) for k in metrics_keys}
    non_metrics_history = {k: history.get(k) for k in non_metrics_keys}
    return metrics_history, non_metrics_history

def _tensors_to_numpy(preds, target):
    with torch.set_grad_enabled(False):
        preds_np = preds.cpu().numpy()
        target_np = target.cpu().detach().numpy()
    return preds_np, target_np

def main():
    print("Nothing to see here.")


if __name__ == "__main__":
    main()
