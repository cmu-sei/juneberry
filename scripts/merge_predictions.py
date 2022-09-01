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
import csv

import juneberry.filesystem as jb_fs


def add_predictions(filepath, id, csvwriter):
    data = jb_fs.load_file(filepath)

    labels = data['testResults']['labels']
    for i, preds in enumerate(data['testResults']['predictions']):
        row = [id, labels[i]] + preds
        csvwriter.writerow(row)


def setup_args(parser) -> None:
    """
    Adds arguments to the parser
    :param parser: The parser in which to add arguments.
    """
    parser.add_argument('predictions0', help='First predictions file.')
    parser.add_argument('predictions1', help='Second predictions file.')
    parser.add_argument('output', help='Path to file for coco output.')


def main():
    # Setup and parse all arguments.
    parser = argparse.ArgumentParser(description="Merges and converts two predictions files to a single csv output.")
    setup_args(parser)
    args = parser.parse_args()

    with open(args.output, "w") as csv_file:
        csv_writer = csv.writer(csv_file)
        add_predictions(args.predictions0, 0, csv_writer)
        add_predictions(args.predictions1, 1, csv_writer)

if __name__ == "__main__":
    main()