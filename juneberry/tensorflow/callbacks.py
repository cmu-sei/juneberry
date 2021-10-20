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
import statistics
import sys

import numpy as np

import tensorflow as tf

logger = logging.getLogger(__name__)


# =======================


class BatchLossCallback(tf.keras.callbacks.Callback):
    """
    This class logs the loss at each batch during training to the logger and to train_out.json.
    Primarily we use this for documenting and collecting epoch data and for generating other
    diagnostics and statistics.
    """

    def __init__(self, verbose=0):
        super().__init__()
        self.verbose = verbose
        self.epoch = 0
        self.batch_loss = []

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        self.batch_loss.append([])

    def on_batch_end(self, batch, logs=None):
        self.batch_loss[-1].append(np.float64(logs['loss']))

        if self.verbose > 0:
            logger.info(f"Epoch: {self.epoch}   "
                        f"Batch: {batch + 1}   "
                        f"Loss: {logs['loss']:.4f}"
                        )


class TimingCallback(tf.keras.callbacks.Callback):
    """
    NOTE: This MUST be the first callback so we track the training time properly without post-processing.

    We show / track two different times so we can show just training vs training and other processing.
    - Epoch training (just the trainer) - This is epoch start to now.
    - Epoch processing time - last reported time to now.
    """

    def __init__(self, total_epochs, num_images):
        super().__init__()
        self.total_epochs = total_epochs
        self.num_images = num_images
        self.time_per_epoch = []
        self.epoch_start_time = None
        self.last_report_time = None

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = datetime.datetime.now()

    def on_epoch_end(self, epoch, logs=None):
        # Get the time right after training so we can track training time.
        report_time = datetime.datetime.now()

        train_elapsed = round((report_time - self.epoch_start_time).total_seconds(), 3)
        if self.last_report_time is not None:
            elapsed = round((report_time - self.last_report_time).total_seconds(), 3)
            self.time_per_epoch.append(elapsed)
            mean_time = statistics.mean(self.time_per_epoch)
        else:
            mean_time = train_elapsed

        # NOTE: We use a DIFFERENT time for computing how long the ENTIRE train plus post
        # process cycle takes so we can give an eta.
        eta = report_time + datetime.timedelta(seconds=(mean_time * (self.total_epochs - epoch - 1)))

        self.last_report_time = report_time

        logging.info(f"{epoch + 1}/{self.total_epochs} "
                     f"time: {train_elapsed:.2f}s "
                     f"eta: {eta.strftime('%H:%M:%S')} "
                     f"images/s: {self.num_images / train_elapsed:.2f} "
                     f"-- "
                     f"loss: {logs['loss']:.4f} "
                     f"acc: {logs['accuracy']:.4f} "
                     f"val_loss: {logs['val_loss']:.4f} "
                     f"val_acc: {logs['val_accuracy']:.4f} "
                     )


class TrainingMetricsCallback(tf.keras.callbacks.Callback):
    """
    This class holds callbacks to track common training metrics.
    """

    def __init__(self, train_key: str = 'accuracy', val_key: str = 'val_accuracy'):
        super().__init__()
        self.train_error = []
        self.val_error = []
        self.norm_before = None
        self.weights_before = None
        self.ratio = None
        self.train_key = train_key
        self.val_key = val_key

    def on_epoch_begin(self, epoch, logs=None):
        # Save weights and norm to calculate ratio of weight updates at end of epoch.
        weights = np.concatenate([layer_weights.ravel() for layer_weights in self.model.get_weights()]).ravel()
        self.norm_before = np.linalg.norm(weights)
        self.weights_before = weights

    def on_epoch_end(self, epoch, logs=None):
        if self.train_key not in logs:
            logger.error(f"TrainingMetricsCallback: Logs do not contain TRAIN key '{self.train_key}'. "
                         f"Options: {logs.keys()}. EXITING")
            sys.exit(-1)
        if self.train_key not in logs:
            logger.error(f"TrainingMetricsCallback: Logs do not contain VAL key '{self.train_key}'. "
                         f"Options: {logs.keys()}. EXITING")
            sys.exit(-1)

        # Use saved weights and updated weights to calculate ratio of weight updates.
        weights = np.concatenate([layer_weights.ravel() for layer_weights in self.model.get_weights()]).ravel()
        diff = weights - self.weights_before
        self.ratio = np.linalg.norm(diff) / self.norm_before  # should be close to 1e-3 for good LR
        logger.info(f"Weight Update Ratio: {self.ratio}")

        # Calculate the train and val errors and append them to the lists.
        self.train_error.append(1 - logs[self.train_key])
        self.val_error.append(1 - logs[self.val_key])


class TestErrorCallback(tf.keras.callbacks.Callback):
    """
    This class holds a callback for computing test error from a specific test set using an image generator.
    """

    def __init__(self, batch_size, test_images, test_labels, image_generator):
        super().__init__()
        self.batch_size = batch_size
        self.test_images = test_images
        self.test_labels = test_labels
        self.test_error = []
        self.image_generator = image_generator

    def on_epoch_end(self, epoch, logs=None):
        # This bit of code was used to display the first 100 test images.
        # it = self.image_generator.flow(self.test_images)
        # for i in range(100):
        #     batch = it.next()
        #     image = batch[0].astype('uint8')
        #     plt.imshow(image)
        #     plt.show()

        test_start = datetime.datetime.now()
        if self.image_generator is not None:
            results = self.model.evaluate(x=self.image_generator.flow(self.test_images, self.test_labels,
                                                                      batch_size=self.batch_size))
        else:
            results = self.model.evaluate(x=self.test_images, y=self.test_labels, batch_size=self.batch_size,
                                          verbose=0)
        logger.info(f"Test evaluate (sec): {round((datetime.datetime.now() - test_start).total_seconds(), 3)}")

        self.test_error.append(1 - results[1])
