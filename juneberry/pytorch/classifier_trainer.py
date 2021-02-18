#! /usr/bin/env python3

# ==========================================================================================================================================================
#  Copyright 2021 Carnegie Mellon University.
#
#  NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS"
#  BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER
#  INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED
#  FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM
#  FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT. Released under a BSD (SEI)-style license, please see license.txt
#  or contact permission@sei.cmu.edu for full terms.
#
#  [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see
#  Copyright notice for non-US Government use and distribution.
#
#  This Software includes and/or makes use of the following Third-Party Software subject to its own license:
#  1. Pytorch (https://github.com/pytorch/pytorch/blob/master/LICENSE) Copyright 2016 facebook, inc..
#  2. NumPY (https://github.com/numpy/numpy/blob/master/LICENSE.txt) Copyright 2020 Numpy developers.
#  3. Matplotlib (https://matplotlib.org/3.1.1/users/license.html) Copyright 2013 Matplotlib Development Team.
#  4. pillow (https://github.com/python-pillow/Pillow/blob/master/LICENSE) Copyright 2020 Alex Clark and contributors.
#  5. SKlearn (https://github.com/scikit-learn/sklearn-docbuilder/blob/master/LICENSE) Copyright 2013 scikit-learn
#      developers.
#  6. torchsummary (https://github.com/TylerYep/torch-summary/blob/master/LICENSE) Copyright 2020 Tyler Yep.
#  7. adversarial robust toolbox (https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/LICENSE)
#      Copyright 2018 the adversarial robustness toolbox authors.
#  8. pytest (https://docs.pytest.org/en/stable/license.html) Copyright 2020 Holger Krekel and others.
#  9. pylint (https://github.com/PyCQA/pylint/blob/master/COPYING) Copyright 1991 Free Software Foundation, Inc..
#  10. python (https://docs.python.org/3/license.html#psf-license) Copyright 2001 python software foundation.
#
#  DM20-1149
#
# ==========================================================================================================================================================

import datetime
import logging
import json
import math
import os

import numpy as np

import torch
import torch.backends.cudnn as cudnn

import juneberry
from juneberry.config.dataset import DatasetConfig
from juneberry.config.training import TrainingConfig
from juneberry.trainer import EpochTrainer
import juneberry.data as jbdata
import juneberry.filesystem as jbfs
import juneberry.plotting
from juneberry.pytorch.acceptance_checker import AcceptanceChecker
import juneberry.pytorch.util as pyt_utils
import juneberry.tensorboard as jbtb


class ClassifierTrainer(EpochTrainer):
    def __init__(self, model_manager, training_config, data_set_config, *,
                 no_paging=False,
                 **kwargs):
        super().__init__(training_config, data_set_config, **kwargs)

        # Assigned during setup
        self.loss_function = None
        self.accuracy_function = None
        self.lr_scheduler = None
        self.optimizer = None
        self.evaluator = None
        self.acceptance_checker = None

        self.model_manager = model_manager

        # We should probably be given a data manager
        self.data_version = model_manager.model_version
        self.data_manager = None
        self.binary = data_set_config.is_binary
        self.pytorch_options = training_config.pytorch

        # Should we load all the data at one time.  Edge case optimization.
        self.no_paging = no_paging

        self.tb_mgr = None

        # This is the model we use
        self.model = None

        # Where we store all the results
        self.history = {}

        # Tracking whether we are using cuda or not
        self.use_cuda = False
        self.device = None

        self.num_batches = -1

        self.memory_summary_freq = int(os.environ.get("JUNEBERRY_CUDA_MEMORY_SUMMARY_PERIOD", 0))

    # ==========================================================================
    # Overrides of EpochTrainer
    def setup(self):
        # Construct helper objects
        self.data_manager = jbfs.DataManager(self.data_set_config.config, self.data_version)

        if juneberry.TENSORBOARD_ROOT:
            self.tb_mgr = jbtb.TensorBoardManager(juneberry.TENSORBOARD_ROOT, self.model_manager)

        pyt_utils.set_seeds(self.training_config.seed)

        self.setup_hardware()
        self.setup_data_loaders()
        self.setup_model()

        self.loss_function = pyt_utils.make_criterion(self.pytorch_options, self.binary)
        self.optimizer = pyt_utils.make_optimizer(self.pytorch_options, self.model)
        self.lr_scheduler = pyt_utils.make_lr_scheduler(self.pytorch_options, self.optimizer)
        self.accuracy_function = pyt_utils.make_accuracy(self.pytorch_options, self.binary)
        self.setup_acceptance_checker()

        self.num_batches = len(self.training_iterable)

        self.history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': [], 'epoch_duration': [], 'lr': []}

    def dry_run(self) -> None:
        summary_path = self.model_manager.get_pytorch_model_summary_file()
        if self.data_set_config.is_image_type():
            # Save some sample images to verify augmentations
            image_shape = pyt_utils.generate_sample_images(self.training_iterable, 5,
                                                           self.model_manager.get_dryrun_imgs_dir())
            pyt_utils.output_summary_file(self.model, image_shape, summary_path)

        elif self.data_set_config.is_tabular_type():
            # TODO Emit sample row modified data
            data, labels = next(iter(self.training_iterable))
            pyt_utils.output_summary_file(self.model, data[0].shape, summary_path)

        else:
            self.logger.error("Dry run doesn't support anything beyond IMAGE or TABULAR type. EXITING")

    def start_epoch_phase(self, train: bool):
        if train:
            self.model.train()
            torch.set_grad_enabled(True)
        else:
            self.model.eval()
            torch.set_grad_enabled(False)

        # Start of with empty metrics
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
        # Losses is a tensorflow thing
        metrics['losses'].append(loss.item())
        metrics['accuracies'].append(accuracy)
        pass

    def update_model(self, results) -> None:
        # Unpack the results we returned on process batch
        loss, _ = results

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def summarize_metrics(self, train, metrics) -> None:
        if train:
            self.history['loss'].append(float(np.mean(metrics['losses'])))
            self.history['accuracy'].append(float(np.mean(metrics['accuracies'])))
        else:
            self.history['val_loss'].append(float(np.mean(metrics['losses'])))
            self.history['val_accuracy'].append(float(np.mean(metrics['accuracies'])))

    def end_epoch(self, elapsed_secs: float) -> str:
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            self.history['lr'].append(self.lr_scheduler.get_last_lr()[0])
        else:
            for param_group in self.optimizer.param_groups:
                self.history['lr'].append(param_group['lr'])

        # Pass the model and value we want to check to the acceptance checker
        # For now, we do validation loss
        self.done = self.acceptance_checker.add_checkpoint(self.model, self.history['val_loss'][-1])

        # Capture the data for TensorBoard (if necessary)
        if self.tb_mgr is not None:
            self.tb_mgr.update(self.history, self.epoch - 1)

        self.history['epoch_duration'].append(elapsed_secs)

        self.show_memory_summary(False)

        # Make a nice metric message for the epoch output
        return f"lr: {self.history['lr'][-1]:.2E}, " \
               f"loss: {self.history['loss'][-1]:.4f}, accuracy: {self.history['accuracy'][-1]:.4f}, " \
               f"val_loss: {self.history['val_loss'][-1]:.4f}, val_accuracy: {self.history['val_accuracy'][-1]:.4f}"

    def finalize_results(self) -> None:
        logging.info(f"Training stopped because: >> {self.acceptance_checker.stop_message} <<")

        # Add a hash of the model
        self.history['modelHash'] = jbfs.generate_file_hash(self.model_manager.get_pytorch_model_file())

        logging.info("Generating and saving output...")
        output = generate_output(self.train_start_time, self.training_config, self.data_set_config, self.history)
        with open(self.model_manager.get_training_out_file(), 'w') as output_file:
            json.dump(output, output_file, indent=4)

        logging.info("Generating summary plot...")
        juneberry.plotting.plot_training_summary_chart(self.model_manager)

    def close(self):
        if self.tb_mgr is not None:
            self.tb_mgr.close()
            self.tb_mgr = None

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
        # Setup for cuda
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.device = torch.device("cuda:0")
            self.logger.info(f"** Using {torch.cuda.device_count()} GPUs.")
        else:
            self.device = torch.device("cpu")

        # These two options must be set in order to achieve
        # reproducibility on a GPU.
        if self.training_config.pytorch.get("deterministic", False):
            cudnn.deterministic = True
            cudnn.benchmark = False

    def setup_data_loaders(self):
        logging.info(f"Preparing data loaders...")
        self.training_iterable, self.evaluation_iterable = \
            jbdata.setup_data_loaders(self.training_config, self.data_set_config,
                                      self.data_manager, self.no_paging)

    def setup_model(self):
        logging.info(f"Constructing the model {self.training_config.model_architecture['module']} "
                     f"with args: {self.training_config.model_architecture['args']} ...")
        self.model = pyt_utils.construct_model(self.training_config.model_architecture,
                                               self.data_set_config.num_model_classes)

        previous_model, prev_model_version = self.training_config.get_previous_model()
        if previous_model is not None:
            self.logger.info(f"Loading weights from previous model: {previous_model}, version: {prev_model_version}")

            prev_model_manager = jbfs.ModelManager(previous_model, prev_model_version)

            pyt_utils.load_weights_from_model(prev_model_manager, self.model)

        if self.use_cuda:
            self.model = torch.nn.DataParallel(self.model)

        # Move the model to the device
        # Q: Do we need to do this for cpu?
        logging.info(f"Moving the model to device={self.device}")
        self.model.to(self.device)
        self.show_memory_summary(True)

    def setup_acceptance_checker(self) -> None:
        """
        Creates an acceptance checker based on the parameters in the training config
        """
        stopping_options = self.training_config.get('stopping', {})
        tol = stopping_options['plateau_abs_tol']
        self.acceptance_checker = AcceptanceChecker(self.model_manager,
                                                    max_epochs=self.max_epochs,
                                                    threshold=stopping_options.get('threshold'),
                                                    plateau_count=stopping_options.get('plateau_count'),
                                                    comparator=lambda x, y: acceptance_loss_comparator(x, y, tol))

    def show_memory_summary(self, model_loading):
        """Used to show a memory summary at appropriate times."""
        if not self.use_cuda or self.memory_summary_freq == 0:
            return

        if model_loading or self.epoch == 1 or (self.epoch - 1) % self.memory_summary_freq == 0:
            if model_loading:
                logging.info(f"CUDA memory summary after model load")
            else:
                logging.info(f"CUDA memory summary for epoch {self.epoch}")
            logging.info(torch.cuda.memory_summary(self.device))


# ==================================================================================================

def acceptance_loss_comparator(x, y, abs_tol):
    """
    Simple function for comparing loss with tolerance for plateaus where is X is "better" than Y.
    :param x: The x value to compare.
    :param y: Y value to compare.
    :param abs_tol: Absolute different.
    :return: Number indicating difference.  Less than, 0 or greater than.
    """
    if math.isclose(x, y, abs_tol=abs_tol):
        return 0
    return -x - -y


def compute_preliminary_eta(num_batches, batch_mean, epoch_start, max_epochs, validation_scale):
    # We are going to make a preliminary estimate based on some set of training batches.
    # We need to scale this based on the size of the validation set. NOTE, this will be
    # WRONG because the validation size doesn't include back propagation.
    epoch_duration = batch_mean * num_batches
    total_duration = epoch_duration * max_epochs * validation_scale
    eta = epoch_start + datetime.timedelta(seconds=total_duration)

    logging.info(f"PRELIMINARY ROUGH Estimate of epoch duration {epoch_duration:.3f} seconds, "
                 f"total ETA {eta.strftime('%H:%M:%S')} ")


def generate_output(start_time,
                    training_config: TrainingConfig,
                    data_set_config: DatasetConfig,
                    history):
    """
    Generates a combined output suitable for JSON output. (Uses JSON style.)
    :param start_time: When the run was started
    :param training_config: The training configuration
    :param data_set_config: The data set configuration
    :param history: A history of the training
    :return: A combined data structure of our output to be written to JSON.
    """
    end_time = datetime.datetime.now().replace(microsecond=0)
    output = {'trainingTimes': {}, 'trainingOptions': {}, 'trainingResults': {}}

    duration = end_time - start_time

    output['trainingTimes']['startTime'] = start_time.isoformat()
    output['trainingTimes']['endTime'] = end_time.isoformat()
    output['trainingTimes']['duration'] = duration.total_seconds()
    output['trainingTimes']['epochDurationSec'] = history['epoch_duration']

    output['trainingOptions']['dataConfig'] = str(training_config.data_set_path)

    output['trainingOptions']['nnArchitecture'] = training_config.model_architecture
    output['trainingOptions']['epochs'] = training_config.epochs
    output['trainingOptions']['batchSize'] = training_config.batch_size
    output['trainingOptions']['seed'] = training_config.seed
    output['trainingOptions']['learningRate'] = history['lr']

    output['trainingOptions']['dataType'] = str(data_set_config.data_type.name)

    # Populate parts from the history
    output['trainingResults']['modelName'] = training_config.model_name
    output['trainingResults']['modelHash'] = history['modelHash']
    output['trainingResults']['loss'] = history['loss']
    output['trainingResults']['accuracy'] = history['accuracy']

    output['trainingResults']['valLoss'] = history['val_loss']
    output['trainingResults']['valAccuracy'] = history['val_accuracy']

    output['formatVersion'] = "1.2.0"

    return output


def main():
    print("Nothing to see here.")


if __name__ == "__main__":
    main()
