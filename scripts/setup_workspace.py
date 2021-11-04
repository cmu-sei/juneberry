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
import shutil
import subprocess
import sys


def create_juneberry_ini(workspace_dir: Path):
    config = configparser.ConfigParser()
    config["DEFAULT"] = {
        "workspace": f"/{workspace_dir.name}",
        "data_root": f"/datasets",
        "tensorboard": "/tensorboard",
        "num_workers": "1",
    }
    with open((workspace_dir / "juneberry.ini"), "w") as f:
        config.write(f)

def create_workspace_dir(workspace_dir):
    print(f"Creating new workspace {workspace_dir}")
    workspace_dir.mkdir()

def create_subdirs(workspace_dir):
    data_sets_dir = workspace_dir / "data_sets"
    data_sets_dir.mkdir()

    experiments_dir = workspace_dir / "experiments"
    experiments_dir.mkdir()

    models_dir = workspace_dir / "models"
    models_dir.mkdir()

    src_dir = workspace_dir / "src"
    src_dir.mkdir()

def copy_container_start(workspace_dir):
    container_start = Path("docker/container_start.sh")
    shutil.copy(container_start, workspace_dir)
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--workspace")
    args = parser.parse_args()

    workspace_dir = Path(args.workspace)
    
    # If the workspace directory exists and is a directory...
    if workspace_dir.exists() and workspace_dir.is_dir():

        # If the workspace directory is not empty...
        if os.listdir(workspace_dir):

            # If the workspace directory is a Juneberry workspace...
            if (workspace_dir / "juneberry.ini").exists():
                # ... nothing to do, just report success.
                print(f"{workspace_dir} exists and is a Juneberry workspace.")
                sys.exit(0)
            else:
                # ... report failure.
                print(f"{workspace_dir} is non-empty, and is not a Juneberry workspace.")
                sys.exit(1)

        else:
            copy_container_start(workspace_dir)
            create_subdirs(workspace_dir)

    else:
        create_workspace_dir(workspace_dir)
        copy_container_start(workspace_dir)
        create_subdirs(workspace_dir)

    create_juneberry_ini(workspace_dir)
    sys.exit(0)


if __name__ == "__main__":
    main()
