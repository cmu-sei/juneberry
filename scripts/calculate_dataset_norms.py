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
import logging

from juneberry.config.dataset import DatasetConfig
import juneberry.data as cutils
import juneberry.image as iutils
from juneberry.lab import Lab


def setup_args(parser) -> None:
    """
    Adds arguments to the parser
    :param parser: The parser in which to add arguments.
    """
    parser.add_argument('dataRoot', help='Root of data directory')
    parser.add_argument('width', help="width to resize images to")
    parser.add_argument('height', help="height to resize images to")
    parser.add_argument('colorspace', help="colorspace to put images in")
    parser.add_argument('-d', '--dataConfig', action='append', required=True,
                        help="A data set file describing the data sets to include in the mean calculation. This "
                             "argument can be used multiple times. Each data set config will be combined into one "
                             "larger set, and the means will be computed for the combined set.")


def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    # Setup and parse all arguments.
    parser = argparse.ArgumentParser(description="Calculates the per-channel means for one or more datasets.")
    setup_args(parser)
    args = parser.parse_args()

    logging.info("Calculating the per channel means using every image in these datasets: " + ','.join(args.dataConfig))

    # Start with an empty list of images.
    image_list = []

    lab = Lab(workspace='.', data_root=args.dataRoot)

    # For each data set, add its images to the image_list.
    for config in args.dataConfig:
        logging.info(f"Adding images from {config} to the list of images.")
        dataset_config = DatasetConfig.load(config)
        # Generate the list of images, process them, then add them to the list.
        image_file_list, _ = cutils.generate_image_manifests(lab, dataset_config)
        images, _ = iutils.load_prepare_images(image_file_list, (int(args.width), int(args.height)), args.colorspace)
        image_list += images
        logging.info(f"Added {len(images)} images to the list of images. Current list total - "
                     f"{len(image_list)} images.")

    # Calculate the means using the images in the list.
    logging.info(f"Calculating the norms for {len(image_list)} images.")
    means = iutils.compute_channel_means(image_list).round(decimals=3).astype(str).tolist()
    std = iutils.compute_channel_stds(image_list).round(decimals=3).astype(str).tolist()
    if type(means) == list:
        means = ", ".join(means)
        std = ", ".join(std)
    logging.info(f"DONE!\n\"mean\": [{means}], \"std\": [{std}]")


if __name__ == "__main__":
    main()
