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

import logging
from typing import Dict, List

from ray.tune import CLIReporter

logger = logging.getLogger(__name__)


class CustomReporter(CLIReporter):
    """
    This class is responsible for altering the logging behavior in Ray Tune. By default, Ray Tune just
    prints its log messages. This CustomReporter overrides that print statement and changes it to a
    logger message, so a record of the message will be maintained inside Juneberry log files.
    """

    def report(self, trials: List, done: bool, *sys_info: Dict):
        logger.info(f"Status Message from Ray Tune:\n{self._progress_str(trials, done, *sys_info)}")
