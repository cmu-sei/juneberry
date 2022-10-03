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

"""
Logging handler.
"""
# https://stackoverflow.com/questions/11232230/logging-to-two-files-with-different-settings

import atexit
import functools
import logging
from pathlib import Path


@functools.lru_cache()
def setup_logger(log_file: Path,
                 log_prefix: str,
                 dist_rank: int = 0,
                 name: str = "juneberry",
                 log_to_console: bool = True,
                 level: int = logging.INFO,
                 log_filter_class: logging.Filter = None):
    """
    Sets up the Juneberry logger. Appends necessary handlers.
    :param log_file: the file to log messages to
    :param log_prefix: prefix appended to log messages
    :param dist_rank: process rank ??
    :param name: logger name
    :param log_to_console: log messages to console as well as file
    :param level: log level for this logger
    :param log_filter_class: subclass of logging.Filter for filtering log messages
    """

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

        # If we were given a log filter class, add an instance of the filter to the console stream handler.
        if log_filter_class:
            ch.addFilter(log_filter_class())

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

        # If we were given a log filter class, add an instance of the filter to the file stream handler.
        if log_filter_class:
            fh.addFilter(log_filter_class())

        logger.addHandler(fh)


# cache the opened file object, so that different calls to `setup_logger`
# with the same file name can safely write to the same file.
@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    io = Path(filename).open(mode='a', buffering=-1)
    atexit.register(io.close)
    return io


def log_banner(logger: logging.Logger, msg, *, width=100, level=logging.INFO):
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
    """A subclass of logging.Filter that filters out adjacent duplicate lines."""

    def filter(self, record):
        """
        Filter an incoming record if it's the same as the previous record.
        :param record: the next incoming log record
        :return: 0 if the message is the same as the previous message, 1 if it's different
        """
        if record.msg != getattr(self, "previous_msg", None):
            self.previous_msg = record.msg
            return 1
        return 0
