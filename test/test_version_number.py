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

import juneberry.version_system as jb_vs


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
    check_version_number_3_no_revision(jb_vs.VersionNumber("3.1.4"))


def test_version_number_1_no_revision():
    check_version_number_3_no_revision(jb_vs.VersionNumber("3.1"))


def test_version_number_3_with_revision():
    v1 = jb_vs.VersionNumber("3.1.4", True)

    assert v1 == jb_vs.VersionNumber("3.1.4", True)
    assert (v1 == jb_vs.VersionNumber("3.1.3", True)) is False

    assert v1 != jb_vs.VersionNumber("3.1.3", True)
    assert (v1 != jb_vs.VersionNumber("3.1.4", True)) is False

    assert v1 > jb_vs.VersionNumber("3.1.3", True)
    assert (v1 > jb_vs.VersionNumber("3.1.4", True)) is False

    assert v1 >= jb_vs.VersionNumber("3.1.3", True)
    assert v1 >= jb_vs.VersionNumber("3.1.3", True)
    assert (v1 >= jb_vs.VersionNumber("3.1.5", True)) is False

    assert v1 < jb_vs.VersionNumber("3.1.5", True)
    assert (v1 < jb_vs.VersionNumber("3.1.4", True)) is False

    assert v1 <= jb_vs.VersionNumber("3.1.4", True)
    assert v1 <= jb_vs.VersionNumber("3.1.5", True)
    assert (v1 <= jb_vs.VersionNumber("3.1.3", True)) is False
