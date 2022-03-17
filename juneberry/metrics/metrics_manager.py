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

import juneberry.loader as loader
from juneberry.metrics.metrics_utils import MetricsUtils
from juneberry.filesystem import EvalDirMgr

logger = logging.getLogger(__name__)


class MetricsManager:

    class Entry:
        def __init__(self, fqcn: str, kwargs: dict = None):
            self.fqcn = fqcn
            self.kwargs = kwargs
            self.metrics = None

    def __init__(self,
                 config: dict,
                 opt_args: dict = None):
        self.config = []

        for i in config["metrics"]:
            if 'fqcn' not in i:
                logger.error(f"Metrics entry does not have required key 'fqcn' {i}")
                sys.exit(-1)

            entry = MetricsManager.Entry(i['fqcn'], i.get('kwargs', None))

            logger.info(f"Constructing metrics: {entry.fqcn} with args: {entry.kwargs}")
            entry.metrics = loader.construct_instance(entry.fqcn, entry.kwargs, opt_args)

            self.config.append(entry)

    def __call__(self, anno, det):
        results = {}

        det_df, anno_df = MetricsUtils.load_dets_and_annos(det, anno)

        for entry in self.config:
            if anno is not None:
                results[entry.fqcn] = entry.metrics(anno_df, det_df)

        return results

    def call_with_eval_dir_manager(self, eval_dir_mgr: EvalDirMgr):
        anno_file = Path(eval_dir_mgr.get_manifest_path())
        det_file = Path(eval_dir_mgr.get_detections_path())
        with open(anno_file, 'r') as f:
            anno = json.load(f)
        with open(det_file, 'r') as f:
            det = json.load(f)
        self.__call__(anno, det)
