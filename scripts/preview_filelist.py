#! /usr/bin/env python

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
from argparse import RawDescriptionHelpFormatter
import logging

from juneberry.config.model import ModelConfig
import juneberry.data
from juneberry.lab import Lab


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    parser = argparse.ArgumentParser(description="Takes the provided dataset config (or model config with -m), "
                                                 "and outputs the list of files that will be provided to "
                                                 "training.  If -m is specified a model config should be provided "
                                                 "and the validation split will also be done.",
                                     formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("dataRoot", help="Root of data directory")
    parser.add_argument("config", help='Name/path of config file. Data config OR train config')
    parser.add_argument('-o', "--output", default="file_list_preview.csv", help="Name of output file.")
    parser.add_argument('-m', '--model', default=False, action='store_true',
                        help="Set to true to consume model json files that specify validation split.")
    parser.add_argument('--version', default="",
                        help='Optional parameter used to control which version of a data set to use.')

    args = parser.parse_args()

    if args.model:
        # The user specified a model config file so we load it and then the dataset.
        # We use a workspace of the cwd which should allow any relative path OR full path to work.
        model_config = ModelConfig.load(args.config)

        lab = Lab(data_root=args.dataRoot)
        dataset_config = lab.load_dataset_config(model_config.training_dataset_config_path)

        # get the files from the data set config
        splitting_config = model_config.get_validation_split_config()
        file_list, val_list = juneberry.data.generate_image_manifests(lab, dataset_config,
                                                                      splitting_config=splitting_config)

        with open(args.output, 'w') as out_file:
            print("TRAIN")
            out_file.write(f"type,path,label\n")
            for k, v in file_list:
                out_file.write(f"train,{k},{v}\n")
            print("VALIDATION")
            for k, v in val_list:
                out_file.write(f"validation,{k},{v}\n")

    else:
        # We expect DATA SET config
        lab = Lab(data_root=args.dataRoot)
        dataset_config = lab.load_dataset_config(args.config)

        file_list, vf = juneberry.data.generate_image_manifests(lab, dataset_config)
        with open(args.output, 'w') as out_file:
            out_file.write(f"type,path,label\n")
            for k, v in file_list:
                out_file.write(f"train,{k},{v}\n")


if __name__ == "__main__":
    main()
