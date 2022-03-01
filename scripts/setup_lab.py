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
from pathlib import Path


def create_missing_dir(d: Path) -> None:
    if not d.exists():
        print(f"Creating project dir \"{d}\".")
        d.mkdir(parents=True, exist_ok=True)


def create_missing_project_dirs(project_dir: str) -> None:
    project_subdirs = [
        "cache",
        "cache/hub",
        "cache/torch",
        "data_root",
        "tensorboard",
    ]
    create_missing_dir(Path(project_dir))
    for subdir in project_subdirs:
        create_missing_dir(Path(project_dir, subdir))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("project_dir", help="Directory containing the Juneberry project.")
    args = parser.parse_args()
    create_missing_project_dirs(args.project_dir)


if __name__ == "__main__":
    main()

