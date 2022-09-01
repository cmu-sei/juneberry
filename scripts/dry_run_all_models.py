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
Trivial script to call "dry run" on every model directory that has a config file.
"""

import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Script to call '--dryrun' on every model directory in "
                                                 "the specified workspace.")
    parser.add_argument("workspace", help="Workspace root (above models).")
    args = parser.parse_args()

    workspace = Path(".") / args.workspace
    os.chdir(workspace)

    for config_path in workspace.glob("models/**/config.json"):
        model_name = "/".join(config_path.parts[1:-1])
        print(f"******** DRY RUN on {model_name}")
        result = subprocess.run(['jb_train', '--dryrun', '-w', str(workspace), model_name])

        if result.returncode != 0:
            print(f"Failed to do dry run '{result.returncode}' on {model_name}. EXITING!!")
            sys.exit(-1)


if __name__ == "__main__":
    main()
