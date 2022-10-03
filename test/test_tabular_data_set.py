#! /usr/bin/env python3

# ======================================================================================================================
# Juneberry - Release 0.5
#
# Copyright 2022 Carnegie Mellon University.
#
# NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS"
# BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER
# INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED
# FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM
# FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.
#
# Released under a BSD (SEI)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.
#
# [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution. Please see
# Copyright notice for non-US Government use and distribution.
#
# This Software includes and/or makes use of Third-Party Software each subject to its own license.
# 
# DM22-0856
#
# ======================================================================================================================

import csv
import math

import juneberry.data
import juneberry.pytorch.tabular_dataset as tabular


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
