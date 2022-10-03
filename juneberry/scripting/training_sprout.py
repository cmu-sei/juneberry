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


@dataclass()
class TrainingSprout(Sprout):
    """
    The TrainingSprout class extends the base Sprout class to include attributes related to training
    models in Juneberry.
    """
    # ========== SCRIPT ARGS ==========
    # ===== EXECUTION MODE ARGS =====
    dryrun: bool = None
    num_gpus: int = None
    resume: bool = None

    # ===== OUTPUT FORMAT ARGS =====
    onnx: bool = None
    skip_native: bool = None

    # ===== MODEL ARGS =====
    model_name: str = None

    def grow_from_args(self, args: Namespace) -> None:
        """
        This method reads a Namespace of arguments and sets the corresponding attributes in the Sprout.
        :param args: A Namespace of arguments, typically created by passing arguments to a Juneberry script.
        :return: Nothing.
        """
        # Start by setting the attributes in the base Sprout.
        super().grow_from_args(args)

        # Now set the attributes stored in the TrainingSprout.
        self.model_name = getattr(args, "modelName", None)
        self.num_gpus = getattr(args, "num_gpus", None)
        self.dryrun = getattr(args, "dryrun", False)
        self.resume = getattr(args, "resume", False)
        self.skip_native = getattr(args, "skipNative", False)
        self.onnx = getattr(args, "onnx", False)
