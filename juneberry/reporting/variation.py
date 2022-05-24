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

import logging
from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np

from juneberry.config.eval_output import EvaluationOutput
from juneberry.filesystem import ModelManager, EvalDirMgr
from juneberry.reporting.report import Report


logger = logging.getLogger(__name__)


# TODO: Move to juneberry utils
def compute_accuracy_np(predictions_np, labels):
    """
    Converts the continuous predictions to classes, compares with truth labels and returns per-class accuracies.
    :param predictions_np: Array of arrays as predictions.
    :param labels: The l
    :return:
    """
    np_labels = np.asarray(labels)

    # Argmax finds the highest entry for each image and returns the index (class / label_index) for that.
    # Then we simple check to see if it got it right and see how many were right.
    outputs = np.argmax(predictions_np, axis=1)
    labels_size = np_labels.shape[0]
    return np.sum(outputs == np_labels) / float(labels_size)


def compute_accuracy_one_class(y_scores_np, target_class):
    """
    Converts the continuous predictions to classes, and finds out how many were correct for that class.
    :param y_scores_np: Array of arrays as predictions.
    :param target_class
    :return:
    """
    # Convert continuous predictions to the highest entry which is the predicted class
    outputs = np.argmax(y_scores_np, axis=1)

    # Find how many we ended up in the target class
    correct = np.sum(outputs == target_class)

    return correct / len(outputs)


def assemble_curves(model_mgr: ModelManager, eval_names: list, target_class: int):
    """
    Walk a series of evaluation metrics and assemble the curves for each
    :param eval_names: A list of lists of eval names.
    :param results_metrics: The name of the results.metric field.
    :return: Curves of values
    """
    curves = []
    for series_list in eval_names:
        curve = []
        for eval_name in series_list:
            eval_dir_mgr: EvalDirMgr = model_mgr.get_eval_dir_mgr(eval_name)
            predictions: EvaluationOutput = EvaluationOutput.load(eval_dir_mgr.get_predictions_path())

            # Convert the raw values to numpy arrays for the utility function
            y_score = np.asarray(predictions.results.predictions)

            # Add the accuracy value for this target class
            curve.append(compute_accuracy_one_class(y_score, target_class))
        curves.append(curve)
    return curves


def format_plot(x_values, curves, curve_names, x_title, y_title, output_dir):
    for idx, curve in enumerate(curves):
        plt.plot(x_values, curve, label = curve_names[idx])
    plt.ylabel(y_title)
    plt.xlabel(x_title)
    plt.legend()
    plt.savefig(str(Path(output_dir) / "variation_plot.png"))


class VariationCurve(Report):
    """
    This creates a chart with a number of curves where each curve shows how a single variable (e.g. watermark size)
    is varied (x-axis) and the y-axis is the appropriate metric such as loss.
    """

    def __init__(self, *, model_name, curve_names, eval_names, target_class, x_title, x_values, y_title,
                 output_dir: str = ""):
        super().__init__(output_dir)

        self.model_name = model_name
        self.curve_names = curve_names
        self.eval_names = eval_names
        self.target_class = target_class
        self.x_title = x_title
        self.x_values = x_values
        self.y_title = y_title

        # Safety checks
        # The number of x_labels should match number of files
        for series in eval_names:
            if len(series) != len(x_values):
                logger.error(f"There are {len(x_values)} and we expected that many entries in series: {series}. "
                             f"Exiting")
                raise RuntimeError("Series length does not match numbeer of x_labels")

    def create_report(self) -> None:
        mmgr = ModelManager(self.model_name)

        # Get the curves
        curves = assemble_curves(mmgr, self.eval_names, self.target_class)

        # Format the plot
        format_plot(self.x_values, curves, self.curve_names, self.x_title, self.y_title, self.output_dir)
