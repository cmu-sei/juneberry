#! /usr/bin/env python3

# ======================================================================================================================
#  Copyright 2021 Carnegie Mellon University.
#
#  NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS"
#  BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER
#  INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED
#  FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM
#  FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.
#
#  Released under a BSD (SEI)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.
#
#  [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.
#  Please see Copyright notice for non-US Government use and distribution.
#
#  This Software includes and/or makes use of the following Third-Party Software subject to its own license:
#
#  1. PyTorch (https://github.com/pytorch/pytorch/blob/master/LICENSE) Copyright 2016 facebook, inc..
#  2. NumPY (https://github.com/numpy/numpy/blob/master/LICENSE.txt) Copyright 2020 Numpy developers.
#  3. Matplotlib (https://matplotlib.org/3.1.1/users/license.html) Copyright 2013 Matplotlib Development Team.
#  4. pillow (https://github.com/python-pillow/Pillow/blob/master/LICENSE) Copyright 2020 Alex Clark and contributors.
#  5. SKlearn (https://github.com/scikit-learn/sklearn-docbuilder/blob/master/LICENSE) Copyright 2013 scikit-learn 
#      developers.
#  6. torchsummary (https://github.com/TylerYep/torch-summary/blob/master/LICENSE) Copyright 2020 Tyler Yep.
#  7. pytest (https://docs.pytest.org/en/stable/license.html) Copyright 2020 Holger Krekel and others.
#  8. pylint (https://github.com/PyCQA/pylint/blob/main/LICENSE) Copyright 1991 Free Software Foundation, Inc..
#  9. Python (https://docs.python.org/3/license.html#psf-license) Copyright 2001 python software foundation.
#  10. doit (https://github.com/pydoit/doit/blob/master/LICENSE) Copyright 2014 Eduardo Naufel Schettino.
#  11. tensorboard (https://github.com/tensorflow/tensorboard/blob/master/LICENSE) Copyright 2017 The TensorFlow 
#                  Authors.
#  12. pandas (https://github.com/pandas-dev/pandas/blob/master/LICENSE) Copyright 2011 AQR Capital Management, LLC,
#             Lambda Foundry, Inc. and PyData Development Team.
#  13. pycocotools (https://github.com/cocodataset/cocoapi/blob/master/license.txt) Copyright 2014 Piotr Dollar and
#                  Tsung-Yi Lin.
#  14. brambox (https://gitlab.com/EAVISE/brambox/-/blob/master/LICENSE) Copyright 2017 EAVISE.
#  15. pyyaml  (https://github.com/yaml/pyyaml/blob/master/LICENSE) Copyright 2017 Ingy döt Net ; Kirill Simonov.
#  16. natsort (https://github.com/SethMMorton/natsort/blob/master/LICENSE) Copyright 2020 Seth M. Morton.
#  17. prodict  (https://github.com/ramazanpolat/prodict/blob/master/LICENSE.txt) Copyright 2018 Ramazan Polat
#               (ramazanpolat@gmail.com).
#  18. jsonschema (https://github.com/Julian/jsonschema/blob/main/COPYING) Copyright 2013 Julian Berman.
#
#  DM21-0689
#
# ======================================================================================================================

"""
This module contains support for the various workflow options.
"""

import logging
from pathlib import Path
import subprocess
import sys
import shutil

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
