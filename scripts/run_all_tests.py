#! /usr/bin/env python3

"""
This script runs all the various testing scripts and counts up the errors.
"""

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

from pathlib import Path
import subprocess
import sys


def run_test(test_args: list):
    """
    Runs the tests arguments and returns the result code.
    :param test_args: The arguments to "subprocess"
    :return: The return code
    """
    print(f"<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")
    print(f">>>>>> Running: {test_args} <<<<<<<<")
    result = subprocess.run(test_args)
    if result.returncode != 0:
        print(f"------------------ WARNING: Test '{test_args}' returned non-zero code: {result.returncode}")
    return result.returncode


def main():
    # We want to test the stuff we are part of, not something in the path.
    juneberry_dir = Path(__file__).resolve().parent.parent

    tests = [['pytest', str(juneberry_dir / 'test')],
             ['python3', str(juneberry_dir / 'scripts/run_system_test.py'), '--initifneeded']
             ]

    total_failures = 0
    report = []
    for test in tests:
        return_code = run_test(test)
        report.append(f"Test: '{test}' - Returned code: {return_code}")
        # Things return a variety of positive and negative codes so we just can't sum the numbers
        if return_code != 0:
            total_failures += 1

    # Summarize the results
    print(f"<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")
    print(f"<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")
    print("\n".join(report))

    if total_failures != 0:
        print(f"FAILURE!!!!!!!! >>>> {total_failures} test(s) failed! See log for details. <<<<")
        sys.exit(-1)
    else:
        print("Success!")


if __name__ == "__main__":
    main()
