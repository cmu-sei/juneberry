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

"""
This module provides a sklearn classification metric plugin to be used with Juneberry.
"""
import logging
from typing import Dict

from juneberry.evaluation import utils as jb_eval_utils
from juneberry.loader import load_verify_fqn_function
from juneberry.metrics.classification.metrics import MetricsBase

logger = logging.getLogger(__name__)


class Metrics(MetricsBase):

    def __init__(self,
                 fqn: str,
                 name: str,
                 kwargs: Dict = None) -> None:
        super().__init__(fqn, name, kwargs)

    def __call__(self, target, preds, binary=False):
        singular_preds = jb_eval_utils.continuous_predictions_to_class(preds, binary)
        metrics_function = load_verify_fqn_function(self.fqn, {**{"y_true": [], "y_pred": []}, **self.kwargs})

        if not metrics_function:
            log_msg = f"Unable to create metrics function: fqn={self.fqn}, name={self.name}, kwargs={self.kwargs}."
            logger.error(log_msg)
            raise ValueError(log_msg)

        return metrics_function(target, singular_preds, **self.kwargs)
