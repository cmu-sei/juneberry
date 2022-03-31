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

import brambox as bb

from juneberry.config import coco_utils
from juneberry.config.model import Metrics
import juneberry.loader as loader
from juneberry.filesystem import EvalDirMgr

logger = logging.getLogger(__name__)


class MetricsManager:
    class Entry:
        """
        Contains the data for each metrics entry in the config.
        :param fqcn: The fully-qualified class name for this entry.
        :param kwargs: The keyword args given with this entry.
        :return: None
        """

        def __init__(self, fqcn: str, kwargs: dict = None) -> None:
            self.fqcn = fqcn
            self.kwargs = kwargs
            self.metrics = None
            self.formatter = None

    def __init__(self, config: List[Metrics], opt_args: dict = None) -> None:
        """
        Create a metrics plugin instance for each entry in the config.
        :param config: list of Metrics plugin entries usually from a config file
        :param opt_args: optional arguments passed to this metrics manager
        :return: None
        """
        self.config = []

        # For each metrics plugin entry in the config, instantiate a metrics plugin object
        # (and associated formatter, if given) and add to the list of metrics
        for i in config:
            # Create a metrics plugin for each entry in the config
            entry = MetricsManager.Entry(i.fqcn, i.kwargs)
            logger.info(f"Constructing metrics: {entry.fqcn} with args: {entry.kwargs}")
            entry.metrics = loader.construct_instance(entry.fqcn, entry.kwargs, opt_args)

            if i.formatter:
                entry.formatter = loader.construct_instance(i.formatter.fqcn, i.formatter.kwargs)

            self.config.append(entry)

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

        # TODO instead of calling get_df() and passing a dataframe to the metrics plugin,
        # send the dict instead and let the plugin convert it to a dataframe for the brambox call

        anno_parser = bb.io.parser.annotation.CocoParser(parse_image_names=False)
        anno_parser.deserialize(json.dumps(anno))
        anno_df = anno_parser.get_df()

        det_parser = bb.io.parser.detection.CocoParser(class_label_map=coco_utils.get_class_label_map(anno))
        det_parser.deserialize(json.dumps(det))
        det_df = det_parser.get_df()

        # For each metrics plugin listed in the config, use the annotations and
        # detections to compute the metrics and add to our results.
        for entry in self.config:
            results[entry.fqcn] = entry.metrics(anno_df, det_df)
            # If a formatter was specified for this metrics plugin, pass the
            # results through the formatter.
            if entry.formatter:
                results[entry.fqcn] = entry.formatter(results[entry.fqcn])
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
