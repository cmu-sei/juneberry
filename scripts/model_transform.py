#! /usr/bin/env python3



# ======================================================================================================================
#  Copyright 2021 Carnegie Mellon University.
#
#  NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS"
#  BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER
#  INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED
#  FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM
#  FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.
#
#  Released under a BSD (SEI)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.
#
#  [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.
#  Please see Copyright notice for non-US Government use and distribution.
#
#  This Software includes and/or makes use of the following Third-Party Software subject to its own license:
#
#  1. PyTorch (https://github.com/pytorch/pytorch/blob/master/LICENSE) Copyright 2016 facebook, inc..
#  2. NumPY (https://github.com/numpy/numpy/blob/master/LICENSE.txt) Copyright 2020 Numpy developers.
#  3. Matplotlib (https://matplotlib.org/3.1.1/users/license.html) Copyright 2013 Matplotlib Development Team.
#  4. pillow (https://github.com/python-pillow/Pillow/blob/master/LICENSE) Copyright 2020 Alex Clark and contributors.
#  5. SKlearn (https://github.com/scikit-learn/sklearn-docbuilder/blob/master/LICENSE) Copyright 2013 scikit-learn
#      developers.
#  6. torchsummary (https://github.com/TylerYep/torch-summary/blob/master/LICENSE) Copyright 2020 Tyler Yep.
#  7. pytest (https://docs.pytest.org/en/stable/license.html) Copyright 2020 Holger Krekel and others.
#  8. pylint (https://github.com/PyCQA/pylint/blob/main/LICENSE) Copyright 1991 Free Software Foundation, Inc..
#  9. Python (https://docs.python.org/3/license.html#psf-license) Copyright 2001 python software foundation.
#  10. doit (https://github.com/pydoit/doit/blob/master/LICENSE) Copyright 2014 Eduardo Naufel Schettino.
#  11. tensorboard (https://github.com/tensorflow/tensorboard/blob/master/LICENSE) Copyright 2017 The TensorFlow
#                  Authors.
#  12. pandas (https://github.com/pandas-dev/pandas/blob/master/LICENSE) Copyright 2011 AQR Capital Management, LLC,
#             Lambda Foundry, Inc. and PyData Development Team.
#  13. pycocotools (https://github.com/cocodataset/cocoapi/blob/master/license.txt) Copyright 2014 Piotr Dollar and
#                  Tsung-Yi Lin.
#  14. brambox (https://gitlab.com/EAVISE/brambox/-/blob/master/LICENSE) Copyright 2017 EAVISE.
#  15. pyyaml  (https://github.com/yaml/pyyaml/blob/master/LICENSE) Copyright 2017 Ingy döt Net ; Kirill Simonov.
#  16. natsort (https://github.com/SethMMorton/natsort/blob/master/LICENSE) Copyright 2020 Seth M. Morton.
#  17. prodict  (https://github.com/ramazanpolat/prodict/blob/master/LICENSE.txt) Copyright 2018 Ramazan Polat
#               (ramazanpolat@gmail.com).
#  18. jsonschema (https://github.com/Julian/jsonschema/blob/main/COPYING) Copyright 2013 Julian Berman.
#
#  DM21-0689
#
# ======================================================================================================================

import argparse
import json
import logging
import sys

import juneberry.pytorch.utils as pyt_utils
from juneberry.transform_manager import TransformManager

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
    with open(args.config_path) as json_file:
        config = json.load(json_file)

    if 'model_architecture' not in config:
        logger.error("Config does not have stanza 'model_architecture'. EXITING.")
        sys.exit(-1)

    if 'model_transforms' not in config:
        logger.error("Config does not have stanza 'model_transforms'. EXITING.")
        sys.exit(-1)

    convert_model(config['model_architecture'], config['model_transforms'], args.num_model_classes)


if __name__ == "__main__":
    main()
