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

import unittest
from juneberry.loader import *


class TestMethodInvoker(unittest.TestCase):
    def setUp(self):
        self.mod_name = 'moddir.simple_mod'

    def assert_error(self, cm, message):
        self.assertEqual(len(cm.output), 1)
        self.assertIn("ERROR:juneberry", cm.output[0])
        self.assertIn(message, cm.output[0])

    def test_invoke_method_bad_module(self):
        with self.assertLogs(level='ERROR') as cm:
            invoke_method('moddir.BadMod', 'MyClass', 'unary', {'a': 'bar'}, True)
        self.assert_error(cm, "moddir.BadMod")

    def test_invoke_method_bad_class(self):
        with self.assertLogs(level='ERROR') as cm:
            invoke_method(self.mod_name, 'BadClass', 'unary', {'a': 'bar'}, True)
        self.assert_error(cm, "BadClass")

    def test_invoke_method_bad_method(self):
        with self.assertLogs(level='ERROR') as cm:
            invoke_method(self.mod_name, 'MyClass', 'BadMethod', {'a': 'bar'}, True)
        self.assert_error(cm, "BadMethod")

    def test_invoke_method_bad_param(self):
        with self.assertLogs(level='ERROR') as cm:
            invoke_method(self.mod_name, 'MyClass', 'unary', {'BadParam': 'bar'}, True)
        self.assertEqual(len(cm.output), 2)
        self.assertIn("ERROR:juneberry", cm.output[0])
        self.assertIn("ERROR:juneberry", cm.output[1])
        self.assertIn('BadParam', cm.output[0])
        self.assertIn('a', cm.output[1])

    def test_invoke_method(self):
        result = invoke_method(self.mod_name, 'MyClass', 'unary', {'a': 'bar'})
        self.assertEqual("a is bar", result)


class TestFunctionInvoker(unittest.TestCase):
    def setUp(self):
        self.mod_name = 'moddir.simple_mod'

    def assert_error(self, cm, message):
        self.assertEqual(len(cm.output), 1)
        self.assertIn("ERROR:juneberry", cm.output[0])
        self.assertIn(message, cm.output[0])

    def test_invoke_function_bad_module(self):
        with self.assertLogs(level='ERROR') as cm:
            invoke_function('moddir.BadMod', 'unary', {'a': 'bar'}, True)
        self.assert_error(cm, "moddir.BadMod")

    def test_invoke_function_bad_function(self):
        with self.assertLogs(level='ERROR') as cm:
            invoke_function(self.mod_name, 'BadFunction', {'a': 'bar'}, True)
        self.assert_error(cm, "BadFunction")

    def test_invoke_function_bad_param(self):
        with self.assertLogs(level='ERROR') as cm:
            invoke_function(self.mod_name, 'binary_function', {'BadParam': 'bar'}, True)
        self.assertEqual(len(cm.output), 3)
        self.assertIn("ERROR:juneberry", cm.output[0])
        self.assertIn("ERROR:juneberry", cm.output[1])
        self.assertIn("ERROR:juneberry", cm.output[2])
        self.assertIn('BadParam', cm.output[0])
        self.assertIn('a', cm.output[1])
        self.assertIn('b', cm.output[2])

    def test_invoke_function(self):
        result = invoke_function(self.mod_name, 'binary_function', {'a': 'foo', 'b': 'bar'})
        self.assertEqual("foo and bar", result)


class TestConstructInstance(unittest.TestCase):

    def test_construct_instance(self):
        my_instance = construct_instance('moddir.simple_mod.MyClass', {})
        assert (my_instance is not None)
        self.assertEqual(my_instance.unary("foo"), "a is foo")

    def test_construct_instance_kwargs(self):
        my_instance = construct_instance('moddir.simple_mod.ClassWithInit', {'name': 'Frodo'})
        assert (my_instance is not None)
        self.assertEqual(my_instance.get_name(), "Frodo")

    def test_construct_instance_kwargs_extra(self):
        my_instance = construct_instance('moddir.simple_mod.ClassWithInit', {}, {'name': 'Pippin', 'missing': 1.0})
        assert (my_instance is not None)
        self.assertEqual(my_instance.get_name(), "Pippin")


if __name__ == "__main__":
    # unittest.main()
    print("Move along")
