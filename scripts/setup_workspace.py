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

import argparse
import logging
from pathlib import Path
import sys

logging.basicConfig(level=logging.INFO, format="%(filename)s:%(levelname)s - %(message)s")


def create_dir(some_dir: Path) -> None:
    """
    Create a directory if it doesn't already exist.
    :param some_dir: The directory to create.
    :return: None
    """
    # Doing it this way instead of using the flags on mkdir
    # so we can print the status message.
    if not some_dir.exists():
        logging.info(f"Creating {some_dir.name} directory...")
        some_dir.mkdir()


def create_workspace_dirs(workspace_dir: str) -> None:
    """
    Create the Juneberry workspace directory and subdirectories, if necessary.
    :param workspace_dir: The Juneberry workspace directory.
    :return: None
    """
    workspace_dir = Path(workspace_dir)
    create_dir(workspace_dir)
    create_dir(workspace_dir / "data_sets")
    create_dir(workspace_dir / "experiments")
    create_dir(workspace_dir / "models")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace", help="Juneberry workspace to set up.")
    args = parser.parse_args()

    logging.info(f"Setting up workspace in {args.workspace}...")

    # Create the Juneberry workspace directories and files.
    # Return 0 for success, 1 for failure.
    try:
        create_workspace_dirs(args.workspace)
    except Exception as e:
        logging.error(e)
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
