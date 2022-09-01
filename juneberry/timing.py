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

"""
A simple time tracking widget.
"""

import datetime as dt
import logging
import statistics

logger = logging.getLogger(__name__)


class Berryometer:
    """
    Class to collect measurements about Juneberry.
    """

    def __init__(self):
        self.widgets = {}

    def __call__(self, task_name):
        """
        :task_name: The name of the task.
        :return: Returns the task widget as a ContextManagee.
        """
        if task_name not in self.widgets:
            widget = TimeTrackingWidget(task_name)
            self.widgets[task_name] = widget
        return self.widgets[task_name]

    def start(self, task_name):
        """
        Starts the timer and returns a function to be called when to stop the time
        :param task_name: The timer to start.
        :return: A convenience token to be called (no args) to stop the timer.
        """
        if task_name not in self.widgets:
            widget = TimeTrackingWidget(task_name)
            self.widgets[task_name] = widget

        widget = self.widgets[task_name]
        widget.start()
        return lambda: widget.stop()

    def log_metrics(self) -> None:
        """ Emit all the metrics to the console. """
        for k, v in self.widgets.items():
            logger.info(f"{k}: {v.mean():0.3f} s")

    def get_all_means(self) -> dict:
        """ :return: Means of all widgets as a dict of task_name:means """
        return {k: v.mean() for k, v in self.widgets.items()}


class TimeTrackingWidget:
    """
    A simple widget for tracking iteration times and counts.
    """

    def __init__(self, name):
        self.elapsed = []
        self.start_time = None
        self.name = name

    def start(self) -> None:
        """
        Starts the time on the tracker.
        """
        if self.start_time is not None:
            logger.warning(f"Trying to start a timer tracker ({self.name}) that is already running!")
            self.stop()
            
        self.start_time = dt.datetime.now()

    def stop(self) -> float:
        """
        Stops the tracker and computes the elapsed time. The elapsed time is stored internally
        for computing mean times and also returned. The start time is cleared.
        :return: The elapsed time since start in seconds.
        """
        if self.start_time is not None:
            elapsed = (dt.datetime.now() - self.start_time).total_seconds()
            self.elapsed.append(elapsed)
            self.start_time = None
            return elapsed
        else:
            logger.error(f"stop() called on TimeTrackingWidget {self.name} that isn't running.")
            return 0.0

    def elapsed(self) -> float:
        """
        Returns the elapsed time since start was called, but doesn't update the internal mean
        time tracker.
        :return: The elapsed time since start in seconds.
        """
        if self.start_time is not None:
            return (dt.datetime.now() - self.start_time).total_seconds()
        else:
            logger.error(f"lap() called on TimeTrackingWidget {self.name} that isn't running.")
            return 0.0

    def last_elapsed(self) -> float:
        """ :return: Last elapsed time """
        return self.elapsed[-1]

    def mean(self):
        """
        :return: The mean of all the elapsed times in seconds.
        """
        return statistics.mean(self.elapsed)

    def weighted_mean(self):
        """
        :return: A weighted mean of the elapsed times. The most recent elapsed times have a higher weight.
        """
        # The ones at the end have a higher weight.
        # Right now it is a simple linear weight
        weights = 0
        time_sum = 0
        for i, n in enumerate(self.elapsed):
            time_sum += n * (i + 1)
            weights += i + 1
        return time_sum / weights

    def __len__(self):
        return len(self.elapsed)

    # For use as a ContextManager with "with" statements.
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, type, value, tb):
        self.stop()
        # We want them to propagate exceptions
        return False
