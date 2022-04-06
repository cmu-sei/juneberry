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

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from juneberry.config.model import Plugin
import juneberry.loader as loader
from juneberry.filesystem import EvalDirMgr

logger = logging.getLogger(__name__)


class MetricsManager:

    class Entry:
        """
        A convenience class used to hold Metrics plugin information from
        the model config, as well as an instance of the Metrics class
        instantiated using that information.
        """
        def __init__(self, fqcn: str, kwargs: dict = None) -> None:
            """
            Initialize an Entry with the fully-qualified class name and kwargs
            for an "evaluation_metrics" entry in the config.
            :param fqcn: The fully-qualified class name for this entry.
            :param kwargs: The keyword args given with this entry.
            :return: None
            """
            self.fqcn = fqcn
            self.kwargs = kwargs
            self.metrics = None

    def __init__(self, metrics_config: List[Plugin], formatter_config: Plugin = None, opt_args: Dict = None) -> None:
        """
        Populate an Entry for each metrics plugin listed in the "evaluation_metrics" section of
        the model config. Also create a metrics formatter if an "evaluation_metrics_formatter" entry
        exists in the config.
        :param metrics_config: list of Metrics plugin entries in the model config's "evaluation_metrics" section
        :param formatter_config: a Metrics formatter plugin entry in the model config's
        "evaluation_metrics_formatter" section
        :param opt_args: optional arguments passed to this metrics manager
        :return: None
        """
        self.metrics_entries = []
        self.formatter = None

        # For each metrics plugin entry in the config,
        # instantiate a metrics plugin object and add to the list of metrics
        for i in metrics_config:
            # Create a metrics plugin for each entry in the config
            entry = MetricsManager.Entry(i.fqcn, i.kwargs)
            logger.info(f"Constructing metrics: {entry.fqcn} with args: {entry.kwargs}")
            entry.metrics = loader.construct_instance(entry.fqcn, entry.kwargs, opt_args)
            self.metrics_entries.append(entry)

        if formatter_config:
            self.formatter = loader.construct_instance(formatter_config.fqcn, formatter_config.kwargs, opt_args)

    def __call__(self, anno: Dict, det: Dict) -> Dict[str, Any]:
        """
        Compute metrics given annotations and detections in dicts.
        :param anno: Annotations dict in COCO format
        :param det: Detections dict in COCO format
        :return: the metrics calculations in a dict
        """
        if not anno:
            logger.info("There are no annotations; cannot populate metrics output!")
            raise ValueError

        results = {}

        # For each metrics plugin we've created, use the annotations and
        # detections to compute the metrics and add to our results.
        for entry in self.metrics_entries:
            results[entry.fqcn] = entry.metrics(anno, det)

        if self.formatter:
            results = self.formatter(results)

        return results

    def call_with_files(self, anno_file: str, det_file: str) -> Dict[str, Any]:
        """
        Convenience method to compute metrics given annotations and detections file in COCO format as input.
        :param anno_file: Annotations file location in COCO format.
        :param det_file: Detections file location in COCO format.
        :return: the metrics calculations in a dict
        """
        # Load annotations and detections JSON files into dicts
        with open(anno_file, "r") as f:
            anno = json.load(f)
        with open(det_file, "r") as f:
            det = json.load(f)
        return self.__call__(anno, det)

    def call_with_eval_dir_manager(self, eval_dir_mgr: EvalDirMgr) -> Dict[str, Any]:
        """
        Convenience method to compute metrics given an EvalDirMgr as input.
        :param eval_dir_mgr: The EvalDirMgr that points to our annotations and detections.
        :return: the metrics calculations in a dict
        """
        anno_file = Path(eval_dir_mgr.get_manifest_path())
        det_file = Path(eval_dir_mgr.get_detections_path())
        return self.call_with_files(str(anno_file), str(det_file))
