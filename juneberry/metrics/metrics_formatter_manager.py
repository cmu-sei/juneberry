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
import sys
from typing import Dict

import juneberry.loader as loader


logger = logging.getLogger(__name__)


class MetricsFormatterManager:

    class Entry:
        def __init__(self, fqcn: str, kwargs: dict = None):
            self.fqcn = fqcn
            self.kwargs = kwargs
            self.metrics_formatter = None

    def __init__(self,
                 config: list,
                 opt_args: dict = None):
        self.config = []

        for i in config:
            if 'fqcn' not in i:
                logger.error(f"MetricsFormatterManager entry does not have required key 'fqcn' {i}")
                sys.exit(-1)

            entry = MetricsFormatterManager.Entry(i['fqcn'], i.get('kwargs', None))

            logger.info(f"Constructing metrics formatter: {entry.fqcn} with args: {entry.kwargs}")
            entry.metrics_formatter = loader.construct_instance(entry.fqcn, entry.kwargs, opt_args)

            self.config.append(entry)

    def __call__(self, metrics: Dict) -> Dict:
        results = {}
        for entry in self.config:
            results = entry.metrics_formatter(metrics)
        return results
