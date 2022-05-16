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


def binary_function(a, b):
    return f"{a} and {b}"


class MyClass:
    @staticmethod
    def unary(a):
        return f"a is {a}"


class ClassWithInit:
    def __init__(self, name):
        self.name = name

    def __call__(self):
        return f"{self.name}"

    def get_name(self):
        return self.name


class ClassWithInitAndUnaryCall:
    def __init__(self, name):
        self.name = name

    def __call__(self, arg):
        return f"{self.name} {arg}"

    def get_name(self):
        return self.name


class ClassWithUnaryCallWithOptArg1:
    def __init__(self):
        self.name = "No name"

    def __call__(self, arg, opt1=None):
        return f"{arg} {opt1}"

    def get_name(self):
        return self.name


class ClassWithUnaryCallWithOptArg2:
    def __init__(self):
        self.name = "No name"

    def __call__(self, arg, opt2=None):
        return f"{arg} {opt2}"

    def get_name(self):
        return self.name


class LabeledTransformExample:
    def __init__(self):
        self.name = "No name"

    def __call__(self, arg, *, label, opt1=None):
        return f"{arg} {opt1}", int(label) + 1

    def get_name(self):
        return self.name


def transform_maker():
    return lambda x: x + x


def transform_maker_arg(y):
    return lambda x: y + x
