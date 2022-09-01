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
import json
from pathlib import Path
import sys


def reformat_data(manifest, pred):
    # The manifest and predictions are in the same order, so just numerically
    # walk the number of images and spew out a new structure.
    new_pred = []

    pred_labels = pred['results']['labels']
    pred_preds = pred['results']['predictions']

    for idx, item in enumerate(manifest):
        # Double check the label
        assert item['label'] == pred_labels[idx]

        # Make a new entry
        new_pred.append({
            "path": item['path'],
            "label": item['label'],
            "predictions": pred_preds[idx]
        })

    # Replace it in the predictions structure and return
    new_out = pred.copy()
    del new_out['results']['labels']
    new_out['results']['predictions'] = new_pred

    return new_out


def reformat_file(eval_dir: str):
    manifest_path = Path(eval_dir) / "eval_manifest.json"
    pred_path = Path(eval_dir) / "predictions.json"
    out_path = Path(eval_dir) / "predictions_v2.json"

    if not manifest_path.exists():
        print(f"Missing '{manifest_path}' file. Exiting.")
        sys.exit()

    if not pred_path.exists():
        print(f"Missing '{pred_path}' file. Exiting.")
        sys.exit()

    with open(pred_path) as pred_file:
        pred_data = json.load(pred_file)

    with open(manifest_path) as manifest_file:
        manifest_data = json.load(manifest_file)

    out_data = reformat_data(manifest_data, pred_data)

    with open(out_path, "w") as out_file:
        json.dump(out_data, out_file, indent=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_dir", help="Path to directory with predictions and manifest.")
    args = parser.parse_args()
    reformat_file(args.eval_dir)


if __name__ == "__main__":
    main()
