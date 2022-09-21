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
from typing import Dict, List

from juneberry.config.model import Plugin
import juneberry.loader as loader

logger = logging.getLogger(__name__)


class MetricsManager:

    class Entry:
        """
        A convenience class used to hold Metrics plugin information from
        the model config, as well as an instance of the Metrics class
        instantiated using that information.
        """
        def __init__(self,
                     fqcn: str,
                     kwargs: dict = None) -> None:
            """
            Initialize an Entry with the fully-qualified class name and kwargs
            for an "evaluation_metrics" entry in the config.
            :param fqcn: The fully-qualified class name for this entry.
            :param kwargs: The keyword args given with this entry.
            :return: None
            """
            self.fqcn = fqcn
            self.kwargs = kwargs

    def __init__(self, metrics_config: List[Plugin], opt_args: Dict = None) -> None:
        """
        Populate an Entry for each metrics plugin listed in the model config.
        :param metrics_config: list of Metrics plugin entries in the model config
        :param opt_args: optional arguments passed to this metrics manager
        :return: None
        """
        self.metrics_entries = []

        # For each metrics plugin entry in the config,
        # instantiate a metrics plugin object and add to the list of metrics
        if not metrics_config:
            logger.warning("No metrics stanza found in model config, no metrics will be generated!")
        else:
            for i in metrics_config:
                # Create a metrics plugin for each entry in the config
                entry = MetricsManager.Entry(i.fqcn, i.kwargs)
                logger.info(f"Constructing metrics: {entry.fqcn} with args: {entry.kwargs}")
                entry.metrics = loader.construct_instance(entry.fqcn, entry.kwargs, opt_args)
                self.metrics_entries.append(entry)

    def __call__(self, target, preds, binary=False):
        """
        Compute metrics given target and preds in numpy arrays.
        :param anno: target array
        :param det: preds array
        :return: the metrics calculations in a numpy array
        """
        results = {}

        if not target.any():
            logger.info("There are no annotations; cannot populate metrics output!")
        else:
            # For each metrics plugin we've created, use the target and
            # preds to compute the metrics and add to our results.
            for entry in self.metrics_entries:
                results[entry.kwargs["name"]] = entry.metrics(target, preds, binary)

        return results
