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
This module provides a torchnn classification metric plugin to be used with Juneberry.
"""
import logging
from typing import Dict

import torch

from juneberry.loader import construct_instance
from juneberry.metrics.classification.metrics import MetricsBase
import juneberry.pytorch.utils as pyt_utils

logger = logging.getLogger(__name__)


class Metrics(MetricsBase):

    def __init__(self,
                 fqn: str,
                 name: str,
                 kwargs: Dict = None) -> None:
        super().__init__(fqn, name, kwargs)

    def __call__(self, target, preds, binary=False):
        target, preds = torch.LongTensor(target), torch.FloatTensor(preds)
        metrics_function = construct_instance(self.fqn, self.kwargs)

        if not metrics_function:
            log_msg = f"Unable to create metrics function: fqn={self.fqn}, name={self.name}, kwargs={self.kwargs}."
            logger.error(log_msg)
            raise ValueError(log_msg)

        if binary:
            metrics_function = pyt_utils.function_wrapper_unsqueeze_1(metrics_function)
        result = metrics_function(preds, target, **self.kwargs)
        return result.numpy()
