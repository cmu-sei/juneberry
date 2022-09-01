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

import juneberry.config.coco_utils as coco_utils
import juneberry.scripting.utils as jb_scripting


def setup_args(parser) -> None:
    """
    Adds arguments to the parser
    :param parser: The parser in which to add arguments.
    """
    parser.add_argument('dataset', help='Data set used to drive predictions.')
    parser.add_argument('predictions', help='Path to predictions file within workspace to convert.')
    parser.add_argument('output', help='Path to file for coco output.')


def main():
    # Setup and parse all arguments.
    parser = argparse.ArgumentParser(description="Converts predictions output to coco format.")
    setup_args(parser)
    jb_scripting.setup_args(parser)
    args = parser.parse_args()

    # TODO: Updated jb_scripting to be more clear
    lab = jb_scripting.setup_workspace(args, log_file=None)

    coco_utils.save_predictions_as_anno(data_root=lab.data_root(), dataset_config=args.dataset,
                                        predict_file=args.predictions, output_file=Path(args.output))


if __name__ == "__main__":
    main()
