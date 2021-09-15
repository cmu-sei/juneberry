#! /usr/bin/env python3

# ======================================================================================================================
#  Copyright 2021 Carnegie Mellon University.
#
#  NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS"
#  BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER
#  INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED
#  FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM
#  FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.
#
#  Released under a BSD (SEI)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.
#
#  [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.
#  Please see Copyright notice for non-US Government use and distribution.
#
#  This Software includes and/or makes use of the following Third-Party Software subject to its own license:
#
#  1. PyTorch (https://github.com/pytorch/pytorch/blob/master/LICENSE) Copyright 2016 facebook, inc..
#  2. NumPY (https://github.com/numpy/numpy/blob/master/LICENSE.txt) Copyright 2020 Numpy developers.
#  3. Matplotlib (https://matplotlib.org/3.1.1/users/license.html) Copyright 2013 Matplotlib Development Team.
#  4. pillow (https://github.com/python-pillow/Pillow/blob/master/LICENSE) Copyright 2020 Alex Clark and contributors.
#  5. SKlearn (https://github.com/scikit-learn/sklearn-docbuilder/blob/master/LICENSE) Copyright 2013 scikit-learn 
#      developers.
#  6. torchsummary (https://github.com/TylerYep/torch-summary/blob/master/LICENSE) Copyright 2020 Tyler Yep.
#  7. pytest (https://docs.pytest.org/en/stable/license.html) Copyright 2020 Holger Krekel and others.
#  8. pylint (https://github.com/PyCQA/pylint/blob/main/LICENSE) Copyright 1991 Free Software Foundation, Inc..
#  9. Python (https://docs.python.org/3/license.html#psf-license) Copyright 2001 python software foundation.
#  10. doit (https://github.com/pydoit/doit/blob/master/LICENSE) Copyright 2014 Eduardo Naufel Schettino.
#  11. tensorboard (https://github.com/tensorflow/tensorboard/blob/master/LICENSE) Copyright 2017 The TensorFlow 
#                  Authors.
#  12. pandas (https://github.com/pandas-dev/pandas/blob/master/LICENSE) Copyright 2011 AQR Capital Management, LLC,
#             Lambda Foundry, Inc. and PyData Development Team.
#  13. pycocotools (https://github.com/cocodataset/cocoapi/blob/master/license.txt) Copyright 2014 Piotr Dollar and
#                  Tsung-Yi Lin.
#  14. brambox (https://gitlab.com/EAVISE/brambox/-/blob/master/LICENSE) Copyright 2017 EAVISE.
#  15. pyyaml  (https://github.com/yaml/pyyaml/blob/master/LICENSE) Copyright 2017 Ingy döt Net ; Kirill Simonov.
#  16. natsort (https://github.com/SethMMorton/natsort/blob/master/LICENSE) Copyright 2020 Seth M. Morton.
#  17. prodict  (https://github.com/ramazanpolat/prodict/blob/master/LICENSE.txt) Copyright 2018 Ramazan Polat
#               (ramazanpolat@gmail.com).
#  18. jsonschema (https://github.com/Julian/jsonschema/blob/main/COPYING) Copyright 2013 Julian Berman.
#
#  DM21-0689
#
# ======================================================================================================================

"""
Component for checking the model for acceptances to stop training.
"""

import logging
import sys

import juneberry.pytorch.util as pyutil

logger = logging.getLogger(__name__)


class AcceptanceChecker:
    def __init__(self, model_manager, comparator, max_epochs=None, threshold=None, plateau_count=None):
        """
        Initializes an acceptance checker to tell when a model is done training. The acceptance checker
        should be provided updated models with the model's acceptance value via the add_checkpoint call.
        Acceptance is defined by number of epochs or meets a threshold or hasn't gotten "better"
        (i.e. plateaus) for some number of epochs. It is up to the caller to choose what value to provided
        and the definition of "better" for the value is provided via the comparator at initialization.
        Once the the acceptance criteria has been met, the 'done' property will be set to true and the model
        will be saved to the models directory.
        :param model_manager: The model manager to save model files.
        :param comparator: Comparator that should return <, =, or > to for comparing values to previous.
        :param max_epochs: OPTIONAL maximum number of epochs.
        :param threshold: OPTIONAL Value threshold for when to stop.
        :param plateau_count: OPTIONAL number of values of a plateau when wee see no improvement
        This comparator defines what "better" is for the model.
        values.
        """
        self.model_manager = model_manager
        self.max_epochs = max_epochs
        self.threshold = threshold
        self.plateau_count = plateau_count
        self.comparator = comparator
        self.stop_message = 'Running'

        # We need to keep track of epoch for epoch stopping and for tracking checkpoints
        self.current_epoch = 0

        # For Plateaus we need to keep track of the "best" value and what epoch it was for
        self.best_value = None
        self.best_epoch = 0

        # We keep going until we are done
        self.done = False

        # Look through the options for the stopping mode.
        if threshold is None and plateau_count is None:
            if max_epochs is None:
                logger.error("AcceptanceChecker requires max_epoch, threshold or plateau_max_count. EXITING")
                sys.exit(-1)
        if comparator is None:
            logger.error("AcceptanceChecker: A comparator must be provided. EXITING.")
            sys.exit(-1)

        if self.plateau_count is not None:
            logger.info(f"AcceptanceChecker: Will stop when value doesn't improve for {self.plateau_count} epochs.")
        if self.threshold is not None:
            logger.info(f'AcceptanceChecker: Will stop when value reaches threshold: {self.threshold}')
        if self.max_epochs is not None:
            logger.info(f"AcceptanceChecker: Will stop when epoch reaches {self.max_epochs}")

    def add_checkpoint(self, model, input_sample, value, allow_save=True) -> bool:
        """
        Adds a single checkpoint for the model. If the value is "accepted" based on the configuration of the
        checker, then the checker's 'done' state will be set to true.
        :param model: The model to save.
        :param input_sample:
        :param value: The value to be provided to the comparator to check for acceptance.
        :param allow_save: Boolean that enables the caller to decide if the model should be saved.
        :return self.done: Boolean status indicating if the training process should end.
        """
        if self.done:
            logger.error(f"AcceptanceChecker called with another checkpoint when already completed! EXITING.")
            sys.exit(-1)

        # We always increase epoch count
        self.current_epoch += 1

        # If they are looking for a plateau, then check that
        if self.plateau_count is not None:

            # If we have a better value we need to save it else check plateau length
            if self.best_value is None or self.comparator(value, self.best_value) > 0:
                self.best_value = value
                self.best_epoch = self.current_epoch
                if allow_save:
                    self._save_model(model, input_sample)

            elif self.current_epoch - self.best_epoch >= self.plateau_count - 1:
                self.stop_message = f"Training reached plateau: {self.plateau_count}. Best epoch: {self.best_epoch}"
                self.done = True

        if not self.done and self.threshold is not None and self.comparator(value, self.threshold) >= 0:
            self.stop_message = f"Training value {value} reached or exceeded {self.threshold}."
            self.done = True
            if allow_save:
                self._save_model(model, input_sample)

        if not self.done and self.max_epochs is not None:
            # Even for epoch based training we always save the best
            if self.best_value is None or self.comparator(value, self.best_value) > 0:
                self.best_value = value
                self.best_epoch = self.current_epoch
                if allow_save:
                    self._save_model(model, input_sample)

            if self.current_epoch >= self.max_epochs:
                self.stop_message = f"Training reached MAX EPOCH: {self.max_epochs}. Best epoch: {self.best_epoch}"
                self.done = True

        return self.done

    # Internal method for mocking out during unit test
    def _save_model(self, model, input_sample) -> None:
        """
        Saves the model to the models directory overwriting any previous model file.
        :param model: The model to save.
        :param input_sample:
        """
        # We need to remove any old one if it exists
        model_path = self.model_manager.get_pytorch_model_path()
        if model_path.exists():
            model_path.unlink()

        pyutil.save_model(self.model_manager, model, input_sample)
