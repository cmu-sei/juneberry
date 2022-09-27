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

import csv
import math

import juneberry.pytorch.tabular_dataset as tabular
import juneberry.data


def make_sample_csv(tmp_path, filename, content):
    with open(tmp_path / filename, 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for row in content:
            writer.writerow(row)


def test_csv_loader(tmp_path):
    header = ["col1", "col2", "col3"]
    make_sample_csv(tmp_path, "file1.csv", [header, [0.1, 0.2, 1], [0.3, 0.4, 0]])
    make_sample_csv(tmp_path, "file2.csv", [header, [0.5, 0.6, 0], [0.7, 0.8, 2]])

    labeled_data = juneberry.data.load_labeled_csvs([tmp_path / "file1.csv", tmp_path / "file2.csv"], 2)

    # Load the data and put it in into the data set
    rows_labels = juneberry.data.flatten_dict_to_pairs(labeled_data)
    ds = tabular.TabularDataset(rows_labels, None)

    assert 4 == len(ds)

    row, label = ds[0]
    assert math.isclose(row[0], 0.1, rel_tol=1e-2)
    assert math.isclose(row[1], 0.2, rel_tol=1e-2)
    assert label == 1

    row, label = ds[1]
    assert math.isclose(row[0], 0.3, rel_tol=1e-2)
    assert math.isclose(row[1], 0.4, rel_tol=1e-2)
    assert label == 0

    row, label = ds[2]
    assert math.isclose(row[0], 0.5, rel_tol=1e-2)
    assert math.isclose(row[1], 0.6, rel_tol=1e-2)
    assert label == 0

    row, label = ds[3]
    assert math.isclose(row[0], 0.7, rel_tol=1e-2)
    assert math.isclose(row[1], 0.8, rel_tol=1e-2)
    assert label == 2
