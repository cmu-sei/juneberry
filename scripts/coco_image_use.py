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

import argparse
import json
import logging
from pathlib import Path

import juneberry.config.coco_utils as coco_utils
import juneberry.scripting.utils as jb_scripting

logger = logging.getLogger("juneberry.coco_image_use.py")


def setup_args(parser):
    parser.add_argument("model", help="Model to search through.")
    parser.add_argument("file_name", help="Image filename to search for.")
    parser.add_argument("-e", "--evals", default=False, action='store_true', help="Also scan all eval directories.")


def show_uses(coco_path, file_name):
    # We get the annotations as a merged file list to make it easy to find.
    # We just scan the list looking for that filename, and if we find it, show it.
    logger.info(f"Searching {coco_path} for {file_name}...")
    helper = coco_utils.load_from_json_file(coco_path)
    flat_list = helper.to_image_list()

    for entry in flat_list:
        entry_file_path = Path(entry.file_name)
        if file_name == entry.file_name or file_name == entry_file_path.name:
            logger.info(json.dumps(entry, indent=4))
            return

    logger.info(f"  {file_name} was not found in {coco_path}")


def main():
    parser = argparse.ArgumentParser(description="This script searches the specified model for "
                                                 "uses of the specified image.")
    jb_scripting.setup_args(parser)
    setup_args(parser)
    args = parser.parse_args()

    # Get the lab and model manager
    lab = jb_scripting.setup_for_single_model(args, log_file=None, model_name=args.model)
    model_manager = lab.model_manager(args.model)

    # See if we can find them.
    show_uses(model_manager.get_training_data_manifest_path(), args.file_name)
    show_uses(model_manager.get_validation_data_manifest_path(), args.file_name)

    if args.evals:
        logger.info("Scanning eval dirs.")
        for eval_dir in model_manager.iter_eval_dirs():
            show_uses(eval_dir.get_manifest_path(), args.file_name)


if __name__ == "__main__":
    main()
