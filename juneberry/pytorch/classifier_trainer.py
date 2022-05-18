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

import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

import juneberry
import juneberry.config.dataset as jb_dataset
from juneberry.config.model import LRStepFrequency, PytorchOptions, StoppingCriteria
import juneberry.data as jbdata
import juneberry.filesystem as jbfs
from juneberry.jb_logging import setup_logger
import juneberry.plotting
from juneberry.pytorch.acceptance_checker import AcceptanceChecker
import juneberry.pytorch.data as pyt_data
import juneberry.pytorch.processing as processing
import juneberry.pytorch.utils as pyt_utils
import juneberry.tensorboard as jbtb
from juneberry.trainer import EpochTrainer
from juneberry.transform_manager import TransformManager

logger = logging.getLogger(__name__)


class ClassifierTrainer(EpochTrainer):
    def __init__(self, lab, model_manager, model_config, dataset_config, log_level):
        super().__init__(lab, model_manager, model_config, dataset_config, log_level)

        # Assigned during setup
        self.loss_function = None
        self.accuracy_function = None
        self.lr_scheduler = None
        self.optimizer = None
        self.evaluator = None
        self.acceptance_checker = None

        # We should probably be given a data manager
        self.data_version = model_manager.model_version
        self.binary = dataset_config.is_binary
        self.pytorch_options: PytorchOptions = model_config.pytorch

        # A single sample from the input data. The dimensions of the sample matter during
        # construction of the ONNX model.
        self.input_sample = None

        # Should we load all the data at one time.  Edge case optimization.
        self.no_paging = False
        if "JB_NO_PAGING" in os.environ and os.environ['JB_NO_PAGING'] == "1":
            logger.info("Setting to no paging mode.")
            self.no_paging = True

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
        self.training_accuracy_list = None

        self.lr_step_frequency = LRStepFrequency.EPOCH

        # Added for the acceptance checker
        self.history_key = None
        self.direction = None
        self.abs_tol = None

    # ==========================================================================
    def dry_run(self) -> None:
        # Setup is the same for dry run
        self.setup()

        summary_path = self.model_manager.get_pytorch_model_summary_path()
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
            logger.error("Dry run doesn't support anything beyond IMAGE or TABULAR type. EXITING")

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
            self.tb_mgr = jbtb.TensorBoardManager(self.lab.tensorboard, self.model_manager)

        pyt_utils.set_pytorch_seeds(self.model_config.seed)

        self.setup_hardware()
        self.setup_data_loaders()
        self.setup_model()

        self.loss_function = pyt_utils.make_loss(self.pytorch_options, self.model, self.binary)
        self.optimizer = pyt_utils.make_optimizer(self.pytorch_options, self.model)
        self.lr_scheduler = pyt_utils.make_lr_scheduler(self.pytorch_options, self.optimizer, self.model_config.epochs)
        self.accuracy_function = pyt_utils.make_accuracy(self.pytorch_options, self.binary)
        self.setup_acceptance_checker()
        if self.pytorch_options.lr_step_frequency == LRStepFrequency.BATCH:
            self.lr_step_frequency = LRStepFrequency.BATCH

        self.num_batches = len(self.training_iterable)

        self.history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': [], 'epoch_duration': [], 'lr': []}

    def finish(self):
        super().finish()
        if self.tb_mgr is not None:
            self.tb_mgr.close()
            self.tb_mgr = None

    # ==========================================================================

    def start_epoch_phase(self, train: bool):
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

        # In distributed training, each process will have a different loss/accuracy value. These lists are used to
        # collect the values from each process, so we need one tensor in the list for every process in the "world".
        if self.distributed:
            self.training_loss_list = [torch.Tensor(1).cuda() for i in range(self.num_gpus)]
            self.training_accuracy_list = [torch.zeros(1, dtype=torch.float64).cuda() for i in range(self.num_gpus)]

        # Start off with empty metrics
        return {'losses': [], 'accuracies': []}

    def process_batch(self, train: bool, data, targets):

        # Move the data to the device
        local_batch, local_labels = data.to(self.device), targets.to(self.device)

        # Forward pass: Pass in the batch of images for it to do its thing
        output = self.model(local_batch)

        # Compute and store loss and accuracy based on the provided functions
        loss = self.loss_function(output, local_labels)
        accuracy = self.accuracy_function(output, local_labels)

        return loss, accuracy

    def update_metrics(self, train: bool, metrics, results) -> None:
        # Unpack the results we returned on process batch
        loss, accuracy = results

        # If we're doing distributed training, the process is a little different.
        if self.distributed:
            # Convert the accuracy to a tensor, so it can be gathered.
            acc_tensor = torch.from_numpy(np.asarray(accuracy, dtype=float)).to(self.device)

            # Make sure the loss can be gathered
            loss_on_device = loss.to(self.device)

            # Create a barrier to wait for all processes to reach this point. Once they do, gather up
            # the loss and accuracy from each process and place it in the appropriate tensor list.
            dist.barrier()
            dist.all_gather(self.training_loss_list, loss_on_device)
            dist.all_gather(self.training_accuracy_list, acc_tensor)

            # Take the value from each tensor in the tensor list and place it in the corresponding metric.
            for tensor in self.training_loss_list:
                metrics['losses'].append(tensor.item())
            for tensor in self.training_accuracy_list:
                metrics['accuracies'].append(tensor.item())
            return

        # Record the loss/accuracy values in the metrics dictionary.
        metrics['losses'].append(loss.item())
        metrics['accuracies'].append(accuracy)

    def update_model(self, results) -> None:
        # Unpack the results we returned on process batch
        loss, _ = results

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.lr_scheduler is not None and self.lr_step_frequency == LRStepFrequency.BATCH:
            self.lr_scheduler.step()

    def summarize_metrics(self, train, metrics) -> None:
        if train:
            self.history['loss'].append(float(np.mean(metrics['losses'])))
            self.history['accuracy'].append(float(np.mean(metrics['accuracies'])))
        else:
            self.history['val_loss'].append(float(np.mean(metrics['losses'])))
            self.history['val_accuracy'].append(float(np.mean(metrics['accuracies'])))

    def end_epoch(self) -> str:
        if self.lr_scheduler is not None:
            if self.lr_step_frequency == LRStepFrequency.EPOCH:
                self.lr_scheduler.step()
            # TODO: Should we try to average learning rate for reporting here?
            self.history['lr'].append(self.lr_scheduler.get_last_lr()[0])
        else:
            for param_group in self.optimizer.param_groups:
                self.history['lr'].append(param_group['lr'])

        # Pass the model and value we want to check to the acceptance checker
        self.done = self.acceptance_checker.add_checkpoint(self.unwrapped_model, self.input_sample,
                                                           self.history[self.history_key][-1],
                                                           allow_save=(self.gpu is None or self.gpu == 0))

        # TODO: Check if loss when nan.

        # Capture the data for TensorBoard (if necessary)
        if self.tb_mgr is not None:
            self.tb_mgr.update(self.history, self.epoch - 1)

        self.show_memory_summary(False)

        # Make a nice metric message for the epoch output
        metric_str = ""
        for x in self.history:
            if len(self.history[x]) > 0:
                if 'accuracy' in x or 'loss' in x:
                    metric_str += f"{x}: {self.history[x][-1]:.4f}, "
                else:
                    metric_str += f"{x}: {self.history[x][-1]:.2E}, "

        return metric_str

    def finalize_results(self) -> None:
        # If we're in distributed mode, only one process needs to perform these actions (since all processes should
        # have the same model).
        if self.distributed and not self.gpu == 0:
            return

        logger.info(f"Training stopped because: >> {self.acceptance_checker.stop_message} <<")

        # Add a hash of the model.
        if self.native:
            self.history['model_hash'] = jbfs.generate_file_hash(self.model_manager.get_pytorch_model_path())

        if self.onnx:
            self.history['onnx_model_hash'] = jbfs.generate_file_hash(self.model_manager.get_onnx_model_path())

        logger.info("Generating and saving output...")
        history_to_results(self.history, self.results, self.native, self.onnx)

        logger.info("Generating summary plot...")
        juneberry.plotting.plot_training_summary_chart(self.results, self.model_manager)

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
            train_list, val_list = jbdata.dataspec_to_manifests(
                self.lab,
                dataset_config=self.dataset_config,
                splitting_config=self.model_config.get_validation_split_config(),
                preprocessors=TransformManager(self.model_config.preprocessors))

            self.training_iterable, self.evaluation_iterable = \
                pyt_data.make_training_data_loaders(self.lab,
                                                    self.dataset_config,
                                                    self.model_config,
                                                    train_list,
                                                    val_list,
                                                    no_paging=self.no_paging,
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
        logger.info(f"Constructing the model {self.model_config.model_architecture['module']} "
                    f"with args: {self.model_config.model_architecture['args']} ...")
        self.model = pyt_utils.construct_model(self.model_config.model_architecture,
                                               self.dataset_config.num_model_classes)

        # If this model is based off another model, then load its weights.
        previous_model, prev_model_version = self.model_config.get_previous_model()
        if previous_model is not None:
            logger.info(f"Loading weights from previous model: {previous_model}, version: {prev_model_version}")

            prev_model_manager = jbfs.ModelManager(previous_model, prev_model_version)

            pyt_utils.load_weights_from_model(prev_model_manager, self.model)

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

    results['results']['loss'] = history['loss']
    results['results']['accuracy'] = history['accuracy']

    results['results']['val_loss'] = history['val_loss']
    results['results']['val_accuracy'] = history['val_accuracy']


def main():
    print("Nothing to see here.")


if __name__ == "__main__":
    main()
