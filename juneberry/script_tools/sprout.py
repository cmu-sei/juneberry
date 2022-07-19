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
from argparse import Namespace
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Sprout:
    """
    The purpose of the Sprout class is to build around the arguments that are passed into Juneberry
    scripts. The base Sprout class reflects the args that are common to all scripts.
    """
    # ========== SCRIPT ARGS ==========
    # ===== DIRECTORY ARGS =====
    workspace_dir: str = None
    dataroot_dir: str = None
    tensorboard_dir: str = None
    log_dir: str = None

    # ===== LOGGING ARGS =====
    silent: bool = None
    log_level: int = None

    # ===== LAB ARGS =====
    profile_name: str = None

    def grow_from_args(self, args: Namespace) -> None:
        """
        This method reads a Namespace of arguments and sets the corresponding attributes in the Sprout.
        :param args: A Namespace of arguments, typically created by passing arguments to a Juneberry script.
        :return: Nothing.
        """
        self.workspace_dir = getattr(args, "workspace", None)
        self.dataroot_dir = getattr(args, "dataRoot", None)
        self.tensorboard_dir = getattr(args, "tensorboard", None)
        self.log_dir = getattr(args, "logDir", None)
        self.silent = getattr(args, "silent", False)
        self.log_level = logging.DEBUG if getattr(args, "verbose", None) else logging.INFO
        self.profile_name = getattr(args, "profileName", None)
