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

import tensorflow as tf

class EvalEpsilonCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, eval_epsilon, start_at_epoch=0):
        self._model = model
        self._eval_epsilon = eval_epsilon
        self._start_epoch = start_at_epoch
        self._last_epsilon = None
        self._do_change = False

    def on_epoch_begin(self, epoch, logs=None):
        if self._start_epoch <= epoch:
            self._do_change = True

    def on_test_begin(self, logs=None):
        if self._do_change:
            self._last_epsilon = self._model.layers[-1].epsilon
            self._model.layers[-1].epsilon = self._eval_epsilon

    def on_test_end(self, logs=None):
        if self._do_change:
            self._model.layers[-1].epsilon = self._last_epsilon

