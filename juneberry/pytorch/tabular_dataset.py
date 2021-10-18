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

import numpy as np

from juneberry.pytorch.util import StagedTransformManager, EpochDataset


class TabularDataset(EpochDataset):
    """
    Loads data from a list of CSV files.
    We assume the CSV has a header in each input file, and that the headers are the same.
    We extract the column that has the label number.
    """

    def __init__(self, rows_labels, transforms=None):
        """
        Initialize the tabular data set loader.
        :param rows_labels: A list of pairs of the row data and labels.
        :param transforms: Any transforms to be applied to each row of floats per epoch.
        """
        super().__init__()

        self.transforms = transforms
        for item in rows_labels:
            assert len(item) == 2
        self.rows_labels = []

        # Pre-process the entire thing to big float arrays so it is ready for transformation.
        for row, label in rows_labels:
            row = [float(x) for x in row]
            self.rows_labels.append([row, label])

    def __len__(self):
        """ :return: Total number of samples. """
        return len(self.rows_labels)

    def __getitem__(self, index):
        """
        Return one item.
        :param index: The index within the data set.
        :return: One transformed item with label
        """
        row, label = self.rows_labels[index]

        if self.transforms is not None:
            row = row.copy()
            if isinstance(self.transforms, StagedTransformManager):
                row = self.transforms(row, index, self.epoch)
            else:
                row = self.transforms(row)

        # They want a row as float
        row = np.array(row).astype(np.float32)

        return row, label
