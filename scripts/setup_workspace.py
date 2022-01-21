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
import configparser
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys


def create_juneberry_ini(workspace_dir: Path) -> None:
    """
    Create a juneberry.ini file for this workspace.
    :param workspace_dir: The Juneberry workspace directory.
    :return: None
    """
    config = configparser.ConfigParser()
    config["DEFAULT"] = {
        "workspace": f"/workspace",
        "data_root": f"/dataroot",
        "tensorboard": "/tensorboard",
        "num_workers": "1",
    }
    with open((workspace_dir / "juneberry.ini"), "w") as f:
        config.write(f)


def create_dir(dir: Path) -> None:
    """
    Create a directory if it doesn't already exist.
    :param dir: The directory to create.
    :return: None
    """
    # Doing it this way instead of using the flags on mkdir
    # so we can print the status message.
    if not dir.exists():
        print(f"Creating {dir.name} directory...")
        dir.mkdir()


def create_workspace_dirs(workspace_dir: Path) -> None:
    """
    Create the Juneberry workspace directory and subdirectories, if necessary.
    :param workspace_dir: The Juneberry workspace directory.
    :return: None
    """
    create_dir(workspace_dir)
    create_dir(workspace_dir / "data_sets")
    create_dir(workspace_dir / "experiments")
    create_dir(workspace_dir / "models")
    create_dir(workspace_dir / "src")


def copy_container_start(workspace_dir: Path, project_dir: Path) -> None:
    """
    Copy the template container_start.sh file from the project to the
    workspace.
    :param workspace_dir: The Juneberry workspace directory.
    :param project_dir: The Juneberry project directory.
    :return: None
    """
    container_start = Path(project_dir, "juneberry/docker/container_start.sh")
    shutil.copy(container_start, workspace_dir)


def add_host_os_to_container_start(workspace_dir: Path) -> None:
    """
    Write the HOST_OS to the workspace's copy of container_start.sh.
    :param workspace_dir: The Juneberry workspace directory.
    :return: None
    """
    container_start = Path(workspace_dir / "container_start.sh")
    outlines = []

    with open(container_start, "r") as f:
        inlines = f.readlines()

        for l in inlines:
            outlines.append(re.sub(r"^HOST_OS=\"\"$",
                f"HOST_OS=\"{platform.system()}\"", l))

    with open(container_start, "w") as f:
        f.writelines(outlines)


def create_workspace_files(workspace_dir: Path, project_dir: Path) -> None:
    """
    Create Juneberry workspace files if missing from this workspace.
    :param workspace_dir: The Juneberry workspace directory.
    :param project_dir: The Juneberry project directory.
    :return: None
    """
    if not (workspace_dir / "juneberry.ini").exists():
        print(f"Creating new juneberry.ini...")
        create_juneberry_ini(workspace_dir)
    if not (workspace_dir / "container_start.sh").exists():
        print(f"Creating new container_start.sh...")
        copy_container_start(workspace_dir, project_dir)
        add_host_os_to_container_start(workspace_dir)

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--project_dir", help="Directory containing the Juneberry project.")
    parser.add_argument("-w", "--workspace", help="Juneberry workspace to set up and run in.")
    args = parser.parse_args()

    project_dir = Path(args.project_dir)
    workspace_dir = Path(args.workspace)
    
    # Create the Juneberry workspace directories and files.
    # Return 0 for success, 1 for failure.
    try:
        create_workspace_dirs(workspace_dir)
        create_workspace_files(workspace_dir, project_dir)
    except Exception as e:
        print(e)
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
