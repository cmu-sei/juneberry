#! /use/bin/env python3

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

import juneberry.version_system as jbvs


def check_version_number_3_no_revision(v1):
    assert v1 == "3.1"
    assert v1 == "3.1.1"
    assert v1 == "3.1.4"
    assert v1 == "3.1.5"
    assert (v1 == "3.0") is False
    assert (v1 == "3.2") is False

    assert v1 != "3.0"
    assert v1 != "3.2"
    assert (v1 != "3.1") is False
    assert (v1 != "3.1.1") is False
    assert (v1 != "3.1.4") is False
    assert (v1 != "3.1.5") is False

    assert v1 < "3.2"
    assert v1 < "100.0"
    assert (v1 < "1.0") is False
    assert (v1 < "3.0") is False
    assert (v1 < "3.0.27") is False

    assert v1 <= "3.1"
    assert v1 <= "3.1.3"
    assert v1 <= "3.1.4"
    assert v1 <= "3.1.5"
    assert v1 <= "100.0"
    assert (v1 <= "1.0") is False
    assert (v1 <= "3.0") is False

    assert v1 > "1.0"
    assert v1 > "3.0"
    assert v1 > "3.0.27"
    assert (v1 > "3.2") is False
    assert (v1 > "100.0") is False

    assert v1 >= "1.0"
    assert v1 >= "3.0"
    assert v1 >= "3.1.0"
    assert v1 >= "3.1.4"
    assert v1 >= "3.1.5"
    assert (v1 >= "3.2") is False
    assert (v1 >= "100.0") is False


def test_version_number_3_no_revision():
    check_version_number_3_no_revision(jbvs.VersionNumber("3.1.4"))


def test_version_number_1_no_revision():
    check_version_number_3_no_revision(jbvs.VersionNumber("3.1"))


def test_version_number_3_with_revision():
    v1 = jbvs.VersionNumber("3.1.4", True)

    assert v1 == jbvs.VersionNumber("3.1.4", True)
    assert (v1 == jbvs.VersionNumber("3.1.3", True)) is False

    assert v1 != jbvs.VersionNumber("3.1.3", True)
    assert (v1 != jbvs.VersionNumber("3.1.4", True)) is False

    assert v1 > jbvs.VersionNumber("3.1.3", True)
    assert (v1 > jbvs.VersionNumber("3.1.4", True)) is False

    assert v1 >= jbvs.VersionNumber("3.1.3", True)
    assert v1 >= jbvs.VersionNumber("3.1.3", True)
    assert (v1 >= jbvs.VersionNumber("3.1.5", True)) is False

    assert v1 < jbvs.VersionNumber("3.1.5", True)
    assert (v1 < jbvs.VersionNumber("3.1.4", True)) is False

    assert v1 <= jbvs.VersionNumber("3.1.4", True)
    assert v1 <= jbvs.VersionNumber("3.1.5", True)
    assert (v1 <= jbvs.VersionNumber("3.1.3", True)) is False
