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
import logging
from pathlib import Path
import sys

import juneberry.config.coco_utils as coco_utils
import juneberry.scripting.utils as jb_scripting

logger = logging.getLogger("juneberry.scripts.draw_boxes_from_anno_file")


def setup_args(parser) -> None:
    """
    Adds arguments to the parser
    :param parser: The parser in which to add arguments.
    """
    parser.add_argument('annotationsFile',
                        help="COCO annotations file describing both the raw images, and the bounding boxes around "
                             "the objects that were detected in each image. ")
    parser.add_argument('-o', '--outputDir',
                        help='An optional output directory where the image results will be saved. When this argument '
                             'is not provided, the images will be saved to the current working directory in a '
                             'directory named "boxed_imgs".')


def main():
    # Setup and parse all arguments.
    parser = argparse.ArgumentParser(description="This script takes a COCO annotations file and produces a directory "
                                                 "of images with bounding boxes drawn around the objects "
                                                 "described in the annotations.")
    setup_args(parser)
    jb_scripting.setup_args(parser)
    args = parser.parse_args()

    # Set up the Lab.
    lab = jb_scripting.setup_workspace(args, log_file=None)

    # Check if the desired annotations file exists. Log an error and exit if it can't be found.
    anno_file = Path(args.annotationsFile)
    if not anno_file.exists():
        logger.error(f"The annotations file {anno_file} was not found. EXITING.")
        sys.exit(-1)

    # Add the bounding boxes to the images and save them to the output directory.
    coco_utils.generate_bbox_images(anno_file, lab, args.outputDir)

    logger.info("Done.")


if __name__ == "__main__":
    main()
