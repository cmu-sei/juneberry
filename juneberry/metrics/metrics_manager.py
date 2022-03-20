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
import sys
import tempfile
from typing import Dict, List

import brambox as bb

import juneberry.config.coco_utils as coco_utils
import juneberry.loader as loader
from juneberry.filesystem import EvalDirMgr


logger = logging.getLogger(__name__)


class MetricsManager:

    class Entry:
        def __init__(self, fqcn: str, kwargs: dict = None):
            self.fqcn = fqcn
            self.kwargs = kwargs
            self.metrics = None

    def __init__(self,
                 config: list,
                 opt_args: dict = None):
        self.config = []

        for i in config:
            if 'fqcn' not in i:
                logger.error(f"Metrics entry does not have required key 'fqcn' {i}")
                sys.exit(-1)

            entry = MetricsManager.Entry(i['fqcn'], i.get('kwargs', None))

            logger.info(f"Constructing metrics: {entry.fqcn} with args: {entry.kwargs}")
            entry.metrics = loader.construct_instance(entry.fqcn, entry.kwargs, opt_args)

            self.config.append(entry)

    def __call__(self, anno, det):
        if not anno:
            logger.info("There are no annotations; cannot populate metrics output!")
            raise ValueError

        results = {}
        anno_df, det_df = MetricsManager._load_annotations_and_detections(anno, det)
        for entry in self.config:
            results[entry.fqcn] = entry.metrics(anno_df, det_df)
        return results

    def call_with_eval_dir_manager(self, eval_dir_mgr: EvalDirMgr):
        anno_file = Path(eval_dir_mgr.get_manifest_path())
        det_file = Path(eval_dir_mgr.get_detections_path())
        with open(anno_file, 'r') as f:
            anno = json.load(f)
        with open(det_file, 'r') as f:
            det = json.load(f)
        return self.__call__(anno, det)

    @staticmethod
    def _load_annotations_and_detections(anno_data: Dict, det_data: Dict):
        with tempfile.NamedTemporaryFile(mode="w+") as anno_file, tempfile.NamedTemporaryFile(mode="w+") as det_file:
            # Write the anno_data and det_data dicts to temporary files
            json.dump(anno_data, anno_file)
            json.dump(det_data, det_file)
            anno_file.flush()
            det_file.flush()
            # Load anno_file and det_file via brambox into DataFrames
            anno_df = bb.io.load("anno_coco", anno_file.name, parse_image_names=False)
            # NOTE: Loading the detections requires access to the annotations file. That's why we load
            # detections and annotations in the same method; so that the TemporaryFiles anno_file
            # and det_file both exist for this call.
            det_df = bb.io.load("det_coco",
                                det_file.name,
                                class_label_map=MetricsManager._get_class_label_map(anno_file.name))
            return anno_df, det_df

    @staticmethod
    def _get_class_label_map(anno_file: str) -> List[str]:
        """
        This function is responsible for retrieving the class label map from the annotations file.
        The class label map is used to convert the values in the class_label column of the
        detections Dataframe from integers into strings.
        :param anno_file: The annotations file containing the class label information.
        :return: A List of str containing the classes for each integer label.
        """

        # Open the annotation file and retrieve the information in the
        # categories field.
        with open(anno_file) as json_file:
            categories = json.load(json_file)["categories"]

        # Create an ID list, which contains every integer value that appears
        # as a category in the annotations file.
        id_list = []
        for category in categories:
            id_list.append(category["id"])

        # Set up the class label map such that there is one entry for every
        # possible integer, even if the integer does not appear as a category
        # in the annotations file.
        class_label_map = [None] * (max(id_list) + 1)

        # For the categories that appear in the annotations file, fill in the
        # appropriate entry of the class label map using the string for that
        # integer.
        for category in categories:
            class_label_map[category["id"]] = category["name"]

        # Brambox expects the first item in the class label map to be for
        # label 1, so take the first item (label 0) and move it to the end of
        # the class label map.
        class_label_map.append(class_label_map.pop(0))

        return class_label_map
