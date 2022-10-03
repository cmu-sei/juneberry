#! /usr/bin/env python3

# ======================================================================================================================
# Juneberry - Release 0.5
#
# Copyright 2022 Carnegie Mellon University.
#
# NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS"
# BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER
# INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED
# FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM
# FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.
#
# Released under a BSD (SEI)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.
#
# [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution. Please see
# Copyright notice for non-US Government use and distribution.
#
# This Software includes and/or makes use of Third-Party Software each subject to its own license.
# 
# DM22-0856
#
# ======================================================================================================================

import hashlib
import io
import logging
import sys

import tensorflow as tf

from juneberry.platform import PlatformDefinitions
import juneberry.utils as jb_utils

logger = logging.getLogger(__name__)


class TensorFlowPlatformDefinitions(PlatformDefinitions):
    def get_model_filename(self) -> str:
        """ :return: The name of the model file that the trainer saves and what evaluators should load"""
        return "model.h5"


def save_summary(model, summary_file_path):
    orig = sys.stdout
    sys.stdout = open(summary_file_path, 'w+', encoding="utf-8")
    model.summary()
    sys.stdout = orig


def hash_summary(model):
    # Swap out a string buffer and capture the summary in the buffer.
    output = io.StringIO()
    orig = sys.stdout
    sys.stdout = output
    model.summary()
    sys.stdout = orig

    # Hash the model summary and stash off the digest before destroying the buffer.
    hasher = hashlib.sha256()
    hasher.update(output.getvalue().encode('utf-8'))
    digest = hasher.hexdigest()

    # Close the object and discard the memory buffer.
    output.close()

    return digest


def set_tensorflow_seeds(seed: int):
    """
    Sets all the random seeds used by all the various pieces.
    :param seed: A random seed to use. Can not be None.
    """
    jb_utils.set_seeds(seed)
    logger.info(f"Setting TensorFlow seed to: {str(seed)}")
    tf.random.set_seed(seed)
