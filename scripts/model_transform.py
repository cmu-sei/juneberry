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
import logging
import sys

import juneberry.pytorch.utils as pyt_utils
from juneberry.transform_manager import TransformManager
import juneberry.filesystem as jbfs

logger = logging.getLogger("juneberry.jb_model_transform")


def convert_model(model_architecture, model_transforms, num_model_classes):
    model = pyt_utils.construct_model(model_architecture, num_model_classes)

    # Apply model transforms.
    transforms = TransformManager(model_transforms)
    transforms.transform(model)


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    # Setup and parse all arguments.
    parser = argparse.ArgumentParser(description="Constructs a model, applies transforms, and exits."
                                                 "The config must be a subset of the training config and it"
                                                 "must contain 'model_architecture' and 'model_transforms'"
                                                 "stanzas. For loading weights and saving, include appropriate"
                                                 "transforms in the 'model_transforms' stanza, as this has no inherent"
                                                 "output.")

    parser.add_argument("config_path", help="Path to the config file with 'model_architecture' and 'model_transforms'.")
    parser.add_argument("num_model_classes", type=int, help="Number of model classes to use on construction.")

    args = parser.parse_args()

    # NOTE: We do NOT use the ModelConfig loader, because we do not require a full config at this time.
    with open(args.config_path, 'rb') as file:
        config = jbfs.load(file)

    if 'model_architecture' not in config:
        logger.error("Config does not have stanza 'model_architecture'. EXITING.")
        sys.exit(-1)

    if 'model_transforms' not in config:
        logger.error("Config does not have stanza 'model_transforms'. EXITING.")
        sys.exit(-1)

    convert_model(config['model_architecture'], config['model_transforms'], args.num_model_classes)


if __name__ == "__main__":
    main()
