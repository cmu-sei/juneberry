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
This module provides object detection plugin classes to be used with Juneberry.
Add one or more of these plugins to the "evaluation_metrics" section of your
model config. When the MetricsManager is called with annotations and detections,
the metrics will be computed.
"""
import logging
from typing import Dict

import torch

from juneberry.loader import construct_instance, load_verify_fqn_function
import juneberry.pytorch.utils as pyt_utils

logger = logging.getLogger(__name__)


class Metrics:

    def __init__(self,
                 fqn: str,
                 name: str,
                 kwargs: Dict = None) -> None:
        self.fqn = fqn
        self.name = name
        self.kwargs = kwargs


    def __call__(self, target, preds, binary=False):
        target, preds = torch.LongTensor(target), torch.FloatTensor(preds)
        metrics_function = construct_instance(self.fqn, self.kwargs)
        if binary:
            metrics_function = pyt_utils.function_wrapper_unsqueeze_1(metrics_function)
        result = metrics_function(preds, target, **self.kwargs)
        return result.numpy()
