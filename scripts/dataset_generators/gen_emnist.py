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

import torchvision.datasets as datasets
import os
import json
import shutil
import datetime

# GOAL: Convert pytorch dataset into juneberry compatible dataset
# Steps:
# 1) Find dataset class names and put them "somewhere"
# 2) Create juneberry filesystem format from labels
# 3) Download data from "somewhere"
# 4) Copy data files from "somewhere" into juneberry file format

DEST_ROOT = "/home/churilla/nfs/datasets/"
DIR_NAME = "emnist-balanced"
DIR_ROOT = os.path.join(DEST_ROOT, DIR_NAME)
TRAIN_ROOT = os.path.join(DIR_ROOT, "train")
TEST_ROOT = os.path.join(DIR_ROOT, "test")
EMNIST_MAPPING = [48, 49, 50, 52, 53, 51, 54, 55, 56, 57,
                  65, 66, 67, 68, 69, 70, 71, 72, 73, 74,
                  75, 76, 77, 78, 79, 80, 81, 82, 83, 84,
                  85, 86, 87, 88, 89, 90, 97, 98, 100, 101,
                  102, 103, 104, 110, 113, 114, 116]

os.makedirs(DEST_ROOT, exist_ok=True)
os.makedirs(os.path.join(DEST_ROOT, TRAIN_ROOT), exist_ok=True)
os.makedirs(os.path.join(DEST_ROOT, TEST_ROOT), exist_ok=True)
for d in [TRAIN_ROOT, TEST_ROOT]:
    for lab in EMNIST_MAPPING:
        os.makedirs(os.path.join(DEST_ROOT, d, chr(lab)), exist_ok=True)

# Items that need to be in a train_spec file
train_spec_filename = "emnist_balanced_28x28_bw_train_van.json"
train_spec = {}
train_spec['numModelClasses'] = 47
train_spec['imageProperties'] = {"dimensions": "28,28", "colorspace": "gray"}
train_spec['sampling'] = {"algorithm": "none", "arguments": {}}
train_spec["description"] = "emnist balanced train images, 28x28, grayscale"
train_spec["timestamp"] = str(datetime.datetime.now())
train_spec["formatVersion"] = "2.0.0"
train_spec["data"] = [{"label": i, "labelName": chr(label), "directory": os.path.join(DIR_NAME, "train", chr(label))}
                      for i, label in enumerate(EMNIST_MAPPING)]

test_spec_filename = "emnist_balanced_28x28_bw_test_van.json"
test_spec = {}
test_spec['numModelClasses'] = 47
test_spec['imageProperties'] = {"dimensions": "28,28", "colorspace": "gray"}
test_spec['sampling'] = {"algorithm": "none", "arguments": {}}
test_spec["description"] = "emnist balanced test images, 28x28, grayscale"
test_spec["timestamp"] = str(datetime.datetime.now())
test_spec["formatVersion"] = "2.0.0"
test_spec["data"] = [{"label": i, "labelName": chr(label), "directory": os.path.join(DIR_NAME, "test", chr(label))} for
                     i, label in enumerate(EMNIST_MAPPING)]

with open(os.path.join(DEST_ROOT, DIR_ROOT, train_spec_filename), "w") as f:
    json.dump(train_spec, f)
with open(os.path.join(DEST_ROOT, DIR_ROOT, test_spec_filename), "w") as f:
    json.dump(test_spec, f)

train_set = datasets.EMNIST(root='./data', split="balanced", train=True, download=True)
test_set = datasets.EMNIST(root='./data', split="balanced", train=False, download=True)

cnt = 0
for img, label in train_set:
    name = f"{cnt:0>6d}.png"
    img.save(os.path.join(DEST_ROOT, TRAIN_ROOT, chr(EMNIST_MAPPING[label]), name))
    cnt += 1

for img, label in test_set:
    name = f"{cnt:0>6d}.png"
    img.save(os.path.join(DEST_ROOT, TEST_ROOT, chr(EMNIST_MAPPING[label]), name))
    cnt += 1

# Cleanup pytorch data directory
shutil.rmtree("./data")

# REMEMBER: Open spec files and format them for humans
