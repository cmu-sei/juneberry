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
from pathlib import Path
import tempfile

import juneberry.logging as jb_logging


def test_filter_repeated_messages():

    with tempfile.TemporaryDirectory() as tmpdir:
        tmplog = Path(tmpdir, "test_jb_logging.out")
        jb_logging.setup_logger(tmplog, "", name="test_jb_logging", level=logging.DEBUG,
                                log_filter_class=jb_logging.RemoveDuplicatesFilter)

        logger = logging.getLogger("test_jb_logging")
        logger.info("Repeated message.")
        logger.info("Repeated message.")
        logger.info("Repeated message.")

        # We logged three messages, but because the messages were duplicates,
        # only one message should have been logged.
        with open(tmplog, 'r') as f:
            num_lines = len(f.readlines())
            assert num_lines == 1
