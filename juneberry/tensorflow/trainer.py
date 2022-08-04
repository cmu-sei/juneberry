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
import numpy as np
import os
from pathlib import Path
import random
import sys

import tensorflow as tf

from juneberry.config.dataset import DatasetConfig
from juneberry.config.model import ModelConfig
from juneberry.config.training_output import TrainingOutput
import juneberry.data as jb_data
import juneberry.filesystem as jbfs
from juneberry.filesystem import ModelManager
from juneberry.lab import Lab
import juneberry.loader as jb_loader
import juneberry.plotting
import juneberry.tensorflow.callbacks as tf_callbacks
import juneberry.tensorflow.data as tf_data
import juneberry.tensorflow.utils as tf_utils
import juneberry.trainer

logger = logging.getLogger(__name__)


class ClassifierTrainer(juneberry.trainer.Trainer):
    def __init__(self, lab: Lab, model_manager: ModelManager, model_config: ModelConfig,
                 dataset_config: DatasetConfig, log_level):
        super().__init__(lab, model_manager, model_config, dataset_config, log_level)

        # Grab these out of the model architecture for convenience.
        self.width = self.model_config.model_architecture.args['img_width']
        self.height = self.model_config.model_architecture.args['img_height']
        self.channels = self.model_config.model_architecture.args['channels']

        self.train_ds = None
        self.val_ds = None

        # TensorFlow uses callbacks to do things like change learning rate, etc.
        self.callbacks = []

        # The model we are going to fit.
        self.model = None

        # The learning rate is the current one, the schedule changes it.
        self.lr_schedule = None

        # The loss function.
        self.loss_fn = None
        self.show_batch_loss = False
        self.bl_callback = None

        # The metrics to use.
        self.metrics = None

        # The optimizer to use.
        self.optimizer = None

        # Should we set TensorFlow to be verbose?
        # 0 no progress bars, 1 progress bars (in live console)
        self.verbose = 1

        # The values generated during train.
        self.history = None

    # ==========================

    def dry_run(self) -> None:
        # Set the seeds
        logger.info(f"Setting random seed: {self.model_config.seed}")
        random.seed(self.model_config.seed)
        np.random.seed(self.model_config.seed)
        tf.random.set_seed(self.model_config.seed)

        # Setup the data loaders like normal.
        self.setup_datasets()

        # Dump some images.
        path = Path(self.model_manager.get_dryrun_imgs_dir())
        path.mkdir(exist_ok=True)
        label_names = jb_data.get_label_mapping(model_manager=self.model_manager, model_config=self.model_config,
                                                train_config=self.dataset_config)
        tf_data.save_sample_images(self.train_ds, self.model_manager.get_dryrun_imgs_dir(), label_names)

        # Setup the model and dump the summary.
        self.setup_model()

        # TODO: Change "get_model_summary_path"
        tf_utils.save_summary(self.model, self.model_manager.get_model_summary_path())

    # ==========================

    def node_setup(self) -> None:
        # Not sure need this, but might need it for multiple concurrent models per cpu
        # https://stackoverflow.com/questions/60048292/how-to-set-dynamic-memory-growth-on-tf-2-1/60064123#60064123
        logger.info("Setting TensorFlow's dynamic memory growth.")
        physical_devices = tf.config.list_physical_devices('GPU')
        for gpu_instance in physical_devices:
            tf.config.experimental.set_memory_growth(gpu_instance, True)

    def establish_loggers(self) -> None:
        logger.warning("establish_loggers() not implemented in base Trainer.")

    def setup(self) -> None:
        # Set the seeds
        random.seed(self.model_config.seed)
        np.random.seed(self.model_config.seed)
        tf.random.set_seed(self.model_config.seed)

        # Setup the data loaders.
        self.setup_datasets()

        # Build the model architecture, optimizers, learning rate, etc.
        self.setup_model()

        # Set up tensorboard
        if self.lab.tensorboard:
            log_dir = self.model_manager.create_tensorboard_directory_name(self.lab.tensorboard)
            logger.info(f"Setting up TensorBoard for directory {log_dir}")
            self.callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1))
        else:
            logger.info(f"TensorBoard NOT configured.")

        # Setup other instrumentation and callbacks - MUST happen after the model is constructed
        self.setup_callbacks()

    def train(self) -> None:
        self.history = self.model.fit(
            self.train_ds,
            epochs=self.model_config.epochs,
            validation_split=None,
            validation_data=self.val_ds,
            validation_freq=1,  # Validate each epoch. We could be explicit e.g. [1,2,10]
            shuffle=False,
            callbacks=self.callbacks,
            # max_queue_size                    # TODO: Should we set this?
            workers=self.lab.profile.num_workers,
            use_multiprocessing=True,  # True since we use tf.keras.Sequence
            verbose=self.verbose,
        )

    def finish(self) -> None:

        self._finalize_results_prep()
        history = self.history.history

        # TODO: Find a better way to do this. TF Callbacks don't have an output option.
        for item in self.callbacks:
            if isinstance(item, juneberry.tensorflow.callbacks.TrainingMetricsCallback):
                history['train_error'] = item.train_error
                history['val_error'] = item.val_error
        history['batch_loss'] = self.bl_callback.batch_loss if self.bl_callback is not None else []

        history_to_results(history, self.results)

        out_model_filename = self.model_manager.get_tensorflow_model_path()

        # Both the native and onnx output formats require this version of the model file to exist.
        self.model.save(str(out_model_filename))

        if self.native:
            logger.info(f"Saving model to '{out_model_filename}'")
            self.results['results']['model_hash'] = jbfs.generate_file_hash(out_model_filename)

        if self.onnx:
            model = tf.keras.models.load_model(out_model_filename)
            tf.saved_model.save(model, "tmp_model")
            onnx_outfile = self.model_manager.get_onnx_model_path()
            logger.info(f"Saving ONNX version of the model to {onnx_outfile}")
            os.system(f"python -m tf2onnx.convert --saved-model tmp_model --output {onnx_outfile}")
            self.results['results']['onnx_model_hash'] = jbfs.generate_file_hash(onnx_outfile)

            # If the native TF output format was not requested, discard the native version of the model.
            if not self.native:
                out_model_filename.unlink()

        if self.model_config.epochs > 0:
            logger.info("Generating summary plot...")
            juneberry.plotting.plot_training_summary_chart(self.results, self.model_manager)

        self._serialize_results()

    # ==========================

    def check_gpu_availability(self, required: int = None):
        """
        This allows the particular backend to use its own method of determining resource
        availability.
        :param required: The number of required GPUs. 'None' will use the maximum available.
        :return: The number of GPUs the trainer can use.
        """
        return 0

    def train_distributed(self, num_gpus) -> None:
        """
        Trains the model in a distributed fashion.
        :param num_gpus: The number of GPUs to use for training.
        :return: None
        """
        logger.warning("train_distributed() not implemented in the base Trainer.")

    # ==========================================================================

    def setup_datasets(self) -> None:
        logger.info(f"Preparing data loaders...")

        self.train_ds, self.val_ds = tf_data.load_split_datasets(self.lab, self.dataset_config, self.model_config,
                                                                 self.model_manager)

        # output['options']['num_training_images'] = reporting['num_train_images']
        self.results['num_training_images'] = len(self.train_ds) * self.model_config.batch_size
        self.results['num_validation_images'] = len(self.val_ds) * self.model_config.batch_size

    def setup_model(self) -> None:
        # Construct the basic model from the model architecture.
        self.construct_model()

        # First, construct learning rate.
        self.make_learning_rate()

        # Now make the loss function.
        self.make_loss_function()

        # We can now make the optimizer with the loss and initial learning rate.
        self.make_optimizer()

        # Make the metrics to feed into the compiler.
        self.make_metrics()

        # Now that we have all the parts, compile it all together.
        self.compile_model()

    def construct_model(self):
        # Construct the basic model from the model architecture.
        args = self.model_config.model_architecture.args
        if not args:
            args = {}
        label_names = jb_data.get_label_mapping(model_manager=self.model_manager, model_config=self.model_config,
                                                train_config=self.dataset_config)
        optional_kwargs = {'labels': label_names}
        jb_data.check_num_classes(args, self.dataset_config.num_model_classes)
        self.model = jb_loader.invoke_call_function_on_class(self.model_config.model_architecture.module, args,
                                                             optional_kwargs)

    def make_learning_rate(self):
        tf_options = self.model_config.tensorflow
        if tf_options.lr_schedule_fn:
            # Construct the scheduler, add it to callbacks, and set the initial learning rate.
            fn = tf_options.lr_schedule_fn
            args = tf_options.lr_schedule_args
            logger.info(f"Instantiating lr_schedule '{fn}' with args: {args}")
            self.lr_schedule = jb_loader.construct_instance(fn, args)

    def make_loss_function(self):
        # Construct a loss function.
        tf_options = self.model_config.tensorflow
        fn = tf_options.loss_fn
        args = tf_options.loss_args
        logger.info(f"Instantiating loss_fn '{fn}' with args: {args}")
        self.loss_fn = jb_loader.construct_instance(fn, args)

        # Establish Batch Loss Callback.
        if self.show_batch_loss:
            logger.info("Adding batch loss callback")
            self.bl_callback = tf_callbacks.BatchLossCallback()
            self.callbacks.append(self.bl_callback)

    def make_optimizer(self):
        tf_options = self.model_config.tensorflow

        args = tf_options.optimizer_args
        if not args:
            args = {}

        # If we have constructed a learning rate schedule, pass it in.
        if self.lr_schedule:
            args['learning_rate'] = self.lr_schedule
        logger.info(f"Instantiating optimizer '{tf_options.optimizer_fn}' with args: {tf_options.optimizer_args}")
        self.optimizer = jb_loader.construct_instance(tf_options.optimizer_fn, tf_options.optimizer_args)

    def make_metrics(self):
        if self.model_config.tensorflow.metrics is None:
            self.metrics = ['accuracy']
            return

        # Let's walk through the metrics in the list and build them.
        self.metrics = []
        for item in self.model_config.tensorflow.metrics:
            if isinstance(item, str):
                self.metrics.append(item)
            elif isinstance(item, dict) and len(item) == 2:
                self.metrics.append(jb_loader.construct_instance(item['fqcn'], item['kwargs']))
            else:
                logger.error(f"Unknown metric {item}. Should be string or pair of FQCN and args. EXITING.")
                sys.exit(-1)

    def compile_model(self):
        logger.info("Compiling the model")
        if self.loss_fn is None or self.optimizer is None:
            logger.error("Cannot compile a model without an optimizer and loss function. EXITING.")
            sys.exit(-1)

        logger.info(f"Compiling with:")
        logger.info(f"...optimizer: {self.optimizer}")
        logger.info(f"...loss:      {self.loss_fn}")
        logger.info(f"...metrics:   {self.metrics}")
        self.model.compile(optimizer=self.optimizer, loss=self.loss_fn, metrics=self.metrics)

    def setup_callbacks(self):
        # Create the training metrics callback.
        user_callbacks = self.model_config.tensorflow.callbacks
        opt_args = {"model": self.model}
        if user_callbacks is not None:
            for item in user_callbacks:
                logger.info(f"Adding user callback: fqcn='{item.fqcn}' args='{item.kwargs}'")
                self.callbacks.append(jb_loader.construct_instance(item.fqcn, item.kwargs, opt_args))
        logger.info(f"Set up callbacks: {self.callbacks}")


def history_to_results(history, output: TrainingOutput):
    """
    Places our history into the results for final output. (Uses JSON style.)
    :param history: A history of the training.
    :param output: Where to store the information so it can be retrieved when constructing the final output.
    """
    # TODO: Move in time from timing callback - epoch_duration_sec
    # TODO: Why handle type conversion here? Wouldn't it be better to assume that a metric must return a correctly typed value?
    from typing import Iterable
    for k, v in history.items():
        if isinstance(v, Iterable):
            v = [float(x) for x in v]
        output.results[k] = v

    # TODO: Are these extra metrics always necessary?
    output.results.train_error = history.get('train_error')
    output.results.val_error = history.get('val_error')
    return

    # output.results.loss = history['loss']
    # output.results.val_loss = history['val_loss']

    # # TODO: What are the list versions from?
    # if isinstance(history['accuracy'], list):
    #     output.results.accuracy = [float(i) for i in history['accuracy']]
    # else:
    #     output.results.accuracy = history['accuracy']

    # if isinstance(history['val_accuracy'], list):
    #     output.results.val_accuracy = [float(i) for i in history['val_accuracy']]
    # else:
    #     output.results.val_accuracy = history['val_accuracy']

    # # These will only be there if the metrics callback was added.
    # output.results.train_error = history.get('train_error', None)
    # output.results.val_error = history.get('val_error', None)

    # output.results.batch_loss = history['batch_loss']
