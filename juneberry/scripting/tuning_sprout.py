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
from argparse import Namespace
from dataclasses import dataclass
import logging

from juneberry.scripting.sprout import Sprout

logger = logging.getLogger(__name__)


@dataclass
class TuningSprout(Sprout):
    """
    The TuningSprout class extends the Sprout class to include args related to model tuning.
    """
    # ========== SCRIPT ARGS ==========
    # ===== EXECUTION MODE ARGS =====
    dryrun: bool = None

    # ===== MODEL ARGS =====
    model_name: str = None

    # ===== TUNING ARGS =====
    tuning_config: str = None

    def grow_from_args(self, args: Namespace) -> None:
        """
        This method reads a Namespace of arguments and sets the corresponding attributes in the Sprout.
        :param args: A Namespace of arguments, typically created by passing arguments to a Juneberry script.
        :return: Nothing.
        """
        # Start by setting the attributes in the base Sprout.
        super().grow_from_args(args)

        # Now set the attributes listed in the TuningSprout.
        self.dryrun = getattr(args, "dryrun", False)
        self.model_name = getattr(args, "modelName", None)
        self.tuning_config = getattr(args, "tuningConfig", None)
