#! /usr/bin/env python3

"""
This script is used to update data sets config files.
"""
# ==========================================================================================================================================================
#  Copyright 2021 Carnegie Mellon University.
#
#  NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS"
#  BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER
#  INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED
#  FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM
#  FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT. Released under a BSD (SEI)-style license, please see license.txt
#  or contact permission@sei.cmu.edu for full terms.
#
#  [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see
#  Copyright notice for non-US Government use and distribution.
#
#  This Software includes and/or makes use of the following Third-Party Software subject to its own license:
#  1. Pytorch (https://github.com/pytorch/pytorch/blob/master/LICENSE) Copyright 2016 facebook, inc..
#  2. NumPY (https://github.com/numpy/numpy/blob/master/LICENSE.txt) Copyright 2020 Numpy developers.
#  3. Matplotlib (https://matplotlib.org/3.1.1/users/license.html) Copyright 2013 Matplotlib Development Team.
#  4. pillow (https://github.com/python-pillow/Pillow/blob/master/LICENSE) Copyright 2020 Alex Clark and contributors.
#  5. SKlearn (https://github.com/scikit-learn/sklearn-docbuilder/blob/master/LICENSE) Copyright 2013 scikit-learn
#      developers.
#  6. torchsummary (https://github.com/TylerYep/torch-summary/blob/master/LICENSE) Copyright 2020 Tyler Yep.
#  7. adversarial robust toolbox (https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/LICENSE)
#      Copyright 2018 the adversarial robustness toolbox authors.
#  8. pytest (https://docs.pytest.org/en/stable/license.html) Copyright 2020 Holger Krekel and others.
#  9. pylint (https://github.com/PyCQA/pylint/blob/master/COPYING) Copyright 1991 Free Software Foundation, Inc..
#  10. python (https://docs.python.org/3/license.html#psf-license) Copyright 2001 python software foundation.
#
#  DM20-1149
#
# ==========================================================================================================================================================

import argparse
import datetime
import json
from pathlib import Path
import shutil

from juneberry.filesystem import VersionNumber


def update_to_3_0_0(config):
    print(f"   ...to 3.0.0")

    # Add dataType
    config['dataType'] = 'image'

    # Move data and properties into imagedata
    config['imageData'] = {'sources': config['data'], 'properties': config['imageProperties']}
    del config['data']
    del config['imageProperties']

    # Update version
    config['formatVersion'] = "3.0.0"


def update_to_3_1_0(config):
    print(f"   ...to 3.1.0")
    # Remove properties in image Data
    if 'imageData' in config:
        image_data = config['imageData']
        if 'properties' in image_data:
            del image_data['properties']

        mapping = {}
        for source in image_data['sources']:
            mapping[source['label']] = source['labelName']
            del source['labelName']

        config['labelNames'] = mapping
    # Update version
    config['formatVersion'] = "3.1.0"


def update_to_latest(filename):
    path = Path(filename)
    backup = path.parent / (path.stem + "-bak" + path.suffix)

    # Make a backup
    if backup.exists():
        print(f"   ...backup '{backup}' exists, skipping backup.")
    else:
        print(f"   ...making backup as: {backup}")
        shutil.copyfile(filename, backup)

    with open(filename) as json_file:
        config = json.load(json_file)

    if 'formatVersion' not in config:
        print(f"   ...no version number! SKIPPING")
        return

    version = VersionNumber(config['formatVersion'], True)

    if version < VersionNumber("2.0"):
        print(f"{filename}: We don't support version numbers before 2.0! SKIPPING")
        return

    # Now that we have done basic versioning, let's update
    print(f"   ...updating...")

    if version < "3.0.0":
        update_to_3_0_0(config)

    version = VersionNumber(config['formatVersion'], True)
    if version <= "3.1.0":
        update_to_3_1_0(config)

    # Update the timestamp
    config['timestamp'] = datetime.datetime.now().replace(microsecond=0).isoformat()

    with open(path, "w") as json_file:
        json.dump(config, json_file, indent=4)


def main():
    parser = argparse.ArgumentParser(description="Script to update data set configs to the latest version")
    parser.add_argument("filenames", nargs="+", help="Names of files to update.")

    args = parser.parse_args()

    for filename in args.filenames:
        print(f"Processing {filename}")
        update_to_latest(filename)

    print("Done!!")


if __name__ == '__main__':
    main()
