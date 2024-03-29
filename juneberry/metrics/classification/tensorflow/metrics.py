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

"""
This module provides a tensorflow classification metric plugin to be used with Juneberry.
"""
import inspect
import logging
from typing import Dict

from juneberry.evaluation import utils as jb_eval_utils
from juneberry.loader import construct_instance, load_verify_fqn_function
from juneberry.metrics.classification.metrics import MetricsBase

logger = logging.getLogger(__name__)


class Metrics(MetricsBase):

    def __init__(self,
                 fqn: str,
                 name: str,
                 kwargs: Dict = None) -> None:
        super().__init__(fqn, name, kwargs)

    def __call__(self, target, preds, binary=False):
        result = None

        # Tensorflow metrics may be called in "standalone" or "compile" mode. If called in standalone mode,
        # the target and preds are sent to the metrics function, the answer is computed and sent back to
        # the caller immediately. If called in "compile" mode, the metrics are not computed immediately; rather,
        # the fqcn and kwargs are passed to the tensorflow model, which computes the metrics during training or
        # evaluation.
        if self.kwargs.standalone:
            # Convert the continuous predictions to single class predictions
            singular_preds = jb_eval_utils.continuous_predictions_to_class(preds, binary)

            logger.info(f"Standalone mode: computing {self.fqn}.")
            del self.kwargs["standalone"]

            # Tensorflow has class-based and functional versions of its metrics.
            # If we fail to instantiate self.fqn as a function, try to construct a class instance instead.
            metrics_function = load_verify_fqn_function(self.fqn, {**{"y_true": [], "y_pred": []}, **self.kwargs})
            if not metrics_function:
                # Keras metrics classes take "name" as a kwarg parameter.
                self.kwargs["name"] = self.name
                metrics_function = construct_instance(self.fqn, self.kwargs)

            # If metrics_function doesn't exist now, we were unable to instantiate either
            # a class instance or a functional version of the metric.
            if not metrics_function:
                log_msg = f"Unable to create metrics function: fqn={self.fqn}, name={self.name}, kwargs={self.kwargs}."
                logger.error(log_msg)
                raise ValueError(log_msg)
            else:
                # If metrics_function is a function, call it directly. If it's a class instance,
                # call update_state on it.
                if inspect.isfunction(metrics_function):
                    result = metrics_function(target, singular_preds, **self.kwargs)
                else:
                    result = metrics_function.update_state(target, singular_preds)
                result = result.numpy()
        else:
            # Since we're not in standalone mode, do not compute the metrics now, it will
            # be done later.
            logger.info(f"Compile mode: deferring computation of metrics with fqn={self.fqn}, name={self.name}, kwargs={self.kwargs}.")
            del self.kwargs["standalone"]

        return result
