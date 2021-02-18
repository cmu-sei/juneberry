#! /usr/bin/env python3

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

import numpy as np
from torch.utils import data


class TabularDataset(data.Dataset):
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
            row = self.transforms.transform(row)

        # They want a row as float
        row = np.array(row).astype(np.float32)

        return row, label
