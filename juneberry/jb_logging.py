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

"""
Logging handler.
"""
# https://stackoverflow.com/questions/11232230/logging-to-two-files-with-different-settings

import atexit
import logging
from logging import Logger
import functools

from pathlib import Path


@functools.lru_cache()
def setup_logger(log_file, log_prefix, dist_rank=0, name="juneberry", log_to_console=True, level=logging.INFO,
                 log_filter=None):
    """Sets up the Juneberry logger. Appends necessary handlers."""

    # Fetch the logger by name, set its level, and make sure the changes don't propagate.
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Determine the proper format string and apply it to the logger.
    format_string = f'{log_prefix}%(asctime)s %(levelname)s (%(name)s:%(lineno)d): %(message)s' \
        if level is logging.DEBUG else f'{log_prefix}%(asctime)s %(levelname)s %(message)s'
    formatter = logging.Formatter(format_string)

    # Set up logging to console for the rank 0 process.
    if dist_rank == 0 and log_to_console:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # Set up logging to file.
    if log_file is not None:

        # Processes other than the rank 0 process get their own unique log file.
        if dist_rank:
            log_file = Path(log_file.parent, f"{log_file.stem}_rank{dist_rank}.txt")

            # If the unique log file already exists, clear out the content.
            if log_file.exists():
                with open(log_file, 'w'):
                    pass

        fh = logging.StreamHandler(_cached_log_stream(log_file))
        fh.setLevel(level)
        fh.setFormatter(formatter)

        if log_filter:
            fh.addFilter(log_filter())

        logger.addHandler(fh)


# cache the opened file object, so that different calls to `setup_logger`
# with the same file name can safely write to the same file.
@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    io = Path(filename).open(mode='a', buffering=-1)
    atexit.register(io.close)
    return io


def log_banner(logger: Logger, msg, *, width=100, level=logging.INFO):
    """
    Prints a very noticeable banner for the log to help identify sections.
    :param logger: The logger to use (so we get the right name)
    :param msg: The message
    :param width: Optional - width of the banner
    :param level: Optional - the logging level
    :return:
    """
    # The 2 is for spacing around the message
    edges = width - len(msg) - 2
    left = int(edges / 2)
    right = edges - left
    logger.log(level, f"# {'=' * width}")
    logger.log(level, f"# {'=' * left} {msg} {'=' * right}")
    logger.log(level, f"# {'=' * width}")


class RemoveDuplicatesFilter(logging.Filter):

    def filter(self, record):
        """
        Remove adjacent duplicate records from the log.
        :param record: a log record to examine for duplicates
        :return: 0 if the message is the same as the previous message, 1 if it's different
        """
        if record.msg != getattr(self, "previous_msg", None):
            self.previous_msg = record.msg
            return 1
        return 0
