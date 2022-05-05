#! /usr/bin/

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


class TestConstructFunction(unittest.TestCase):
    def test_no_arg(self):
        my_instance = construct_instance('moddir.simple_mod.transform_maker', {})
        assert (my_instance is not None)
        self.assertEqual(my_instance(2), 4)

    def test_no_arg(self):
        my_instance = construct_instance('moddir.simple_mod.transform_maker_arg', {'y': 6})
        assert (my_instance is not None)
        self.assertEqual(my_instance(2), 8)


if __name__ == "__main__":
    # unittest.main()
    print("Move along")
