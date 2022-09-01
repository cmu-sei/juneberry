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

"""
This module contains support for the various workflow options.
"""

import logging
from pathlib import Path
import shutil
import subprocess
import sys

logger = logging.getLogger(__name__)


def clean_files(files) -> None:
    """
    Given a set of file, directories, or glob patterns, deletes all the files, directories and patterns.
    :param files: List of files to clean
    :return:
    """
    for file_path in files:
        if "*" in str(file_path):
            clean_files(Path(".").glob(file_path))
        else:
            path = Path(file_path)
            if path.exists():
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()


def add_inputs(args, flag, override):
    """
    If override was specified, appends the user inputs to the list of command arguments.
    :param args: List of arguments to be used with subprocess.
    :param flag: String indicating which option was specified.
    :param override: Value specified by override.
    :return:
    """
    if override:
        args.append(flag)
        args.append(str(override))


def run_command(
        dependencies,
        args,
        workspace=None,
        dataroot=None,
        tensorboard=None,
        silent=None,
        show_command: bool = False,
):
    """
    If the dependencies exist, then runs the command specified by args. If the command returns
    a non-zero result, then the command exits python.
    :param dependencies: A set of dependencies that must exist.
    :param args: An array the command and arguments to used with subprocess.
    :param workspace: Root of workspace directory. Overrides values pulled from config files.
    :param dataroot: Root of data directory. Overrides values pulled from config files.
    :param tensorboard: Tensorboard log directory. Overrides values pulled from config files.
    :param silent: Silences output to console.
    :param show_command: Flag set to true to log command that is being run.
    :return:
    """
    for dependency in dependencies:
        if not Path(dependency).exists():
            logger.error(f"Missing dependency: '{dependency}'. EXITING!!")

    # Convert all the arguments to strings.  We might have things like Paths
    args = [str(x) for x in args]

    # Add overrides
    add_inputs(args, "-w", workspace)
    add_inputs(args, "-d", dataroot)
    add_inputs(args, "-t", tensorboard)
    if silent:
        args.append('-s')

    # Log the command
    if show_command:
        quoted_args = []
        for arg in args:
            if arg.find(" ") != -1:
                quoted_args.append(f'"{arg}"')
            else:
                quoted_args.append(arg)
        logger.info(f"CMD: {' '.join(quoted_args)}")

    # Run Command
    result = subprocess.run(args)
    if result.returncode != 0:
        logger.error(
            f"Command returned error code {result.returncode}. Command: '{args}'. EXITING!!"
        )
        sys.exit(-1)
