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

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
from prodict import List, Prodict
import re
from sklearn.metrics import auc, roc_curve
import sys

from juneberry.config.eval_output import EvaluationOutput
from juneberry.reporting.report import Report
from juneberry.reporting.utils import determine_report_path

logger = logging.getLogger(__name__)


class ROCPlot(Report):
    """
    The purpose of this Report subclass is to reproduce the functionality that was previously
    contained inside the 'jb_plot_roc' script. This report will create a figure of one or more
    ROC curves using the prediction data generated by a trained model.
    """
    def __init__(self, output_filename: str = "", plot_title: str = "", legend_scaling: float = 1.0,
                 curve_sources: Prodict = None, line_width: int = 2, legend_font_size: int = 10):
        super().__init__(output_str=output_filename)

        # Determine where to save the output for this report.
        default_filename = "ROC_curves.png"
        self.report_path = determine_report_path(self.output_dir, output_filename, default_filename)
        logger.info(f"Saving the ROC plot to {self.report_path}")

        # Store some attributes related to the plot.
        self.plot_title = "ROC Curve(s)" if plot_title == "" else plot_title
        self.legend_scaling = legend_scaling
        self.line_width = line_width
        self.legend_font_size = legend_font_size

        # Tracks how many curves have been added to the plot. This will be a factor
        # when it comes to scaling the figure.
        self.num_curves = 0

        # Store the curve sources, which is a dictionary containing which predictions files
        # to pull plot data from, along with which classes to add to the plot.
        self.curve_sources = curve_sources

    def create_report(self) -> None:
        """
        This method is responsible for creating the Figure and saving it to the desired file.
        :return: Nothing.
        """
        # Establish the figure.
        fig = plt.figure()

        # ROC curves are supposed to be square, so the first thing we do is set their aspect ratio
        # but also allow it to be adjustable. Because of how the layout magic works, the anchor is
        # important to get more of the legend.
        plt.gca().set_aspect('equal', adjustable='box', anchor="N")

        # Added a dashed diagonal line to the figure.
        plt.plot([0, 1], [0, 1], 'k--', lw=self.line_width)

        # Some other figure properties, included axis labels and the title.
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(self.plot_title)

        # Adjust the font size in the legend
        plt.rc('legend', fontsize=self.legend_font_size)

        # Get the current axes for the figure.
        ax = plt.gca()

        # Now loop through all the predictions files and plot the desired classes on the same figure.
        if self.curve_sources:
            for source in self.curve_sources:
                self._add_curves_to_figure(source, ax)
        else:
            logger.error(f"No curve sources were provided. Nothing to add to the ROC plot!")
            return

        # Now, we need to have some space for the legend at the bottom.  However, the figure adds
        # extra padding for some reason so the number of inches we set it to isn't really right.
        # When we add some, it adds some more for some unknown reasons. So the multiplier has been
        # generated empirically to look OK for 10 items. At some point it would be nice to understand
        # exactly the relationship to the size and the height and how matplotlib lays out the legend.
        # IF YOU CHANGE ROOT SIZE, CHANGE THE MAGIC
        # Scaling of 1.36 for 25 items results in magic of 0.34
        # Scaling of 1.12 for 50 items results in magic of 0.28
        magic = 0.25 * self.legend_scaling

        # Size the figure to a reasonable size using the magic buffer.
        fig.set_size_inches(w=7, h=5 + magic * self.num_curves)

        # Sort the legend by class (alphabetically) and add the legend to the figure.
        handles, labels = ax.get_legend_handles_labels()
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.15))

        # Adjust the final layout and save the figure to file.
        plt.tight_layout()
        plt.savefig(self.report_path)

    def _add_curves_to_figure(self, curve_source: str, ax: matplotlib.axes) -> None:
        """
        This method is responsible for adding ROC curves to the figure.
        :param curve_source: String indicating a predictions.json file containing the data for the
        desired ROC curve(s).
        :param ax: matplotlib axes for the figure where the ROC curve(s) are being drawn.
        :return: Nothing.
        """
        logger.info(f"Plotting ROC curves using predictions data in {curve_source}")
        # Load the predictions file and retrieve the mapping of classes in the predictions data.
        prediction_file_content = EvaluationOutput.load(curve_source)
        class_mapping = prediction_file_content.options.dataset.classes

        # Retrieve the string describing which classes from the current predictions file the user
        # wants to generate ROC curves for.
        desired_classes_str = self.curve_sources[curve_source]

        # Indicate the initial request, convert the class string to a list of corresponding
        # class integers, and indicated the results of that conversion.
        logger.info(f"  Desired classes - original form: {desired_classes_str}")
        converted_classes_list = self._convert_classes(desired_classes_str, class_mapping, curve_source)
        logger.info(f"  Desired classes -  integer form: {converted_classes_list}")

        # Convert the labels (truth) from the predictions file into an array.
        labels = np.asarray(prediction_file_content.results.labels)

        # Set up an empty array for the y_test data, then fill out the array.
        y_test = np.zeros((labels.size, prediction_file_content.options.model.num_classes))
        for i in range(labels.size):
            j = labels[i]
            y_test[i][j] = 1

        # Convert the predictions data into the y_score array.
        y_score = np.asarray(prediction_file_content.results.predictions)

        # Loop through each ROC curve the user wants to add to the figure.
        for curve_class in converted_classes_list:
            logger.info(f"Drawing ROC curve for class: {curve_class}")
            # Increment the curve counter.
            self.num_curves += 1

            # A curve may be for one or more classes. If it's more than one class, a "+" sign
            # indicates multiple classes are being combined.
            if "+" in curve_class:
                # Split the combined classes into their individual components.
                components = curve_class.split("+")

                # Create empty arrays to hold test and score day, plus an empty string for the class name.
                test_data = np.zeros(len(y_test[:, 0]))
                score_data = np.zeros(len(y_score[:, 0]))
                class_string = ''

                # Now for each individual class in this combined class ROC curve, add the individual
                # class contribution to the arrays and string for this curve.
                for component in components:
                    # Add its contribution to the test data, score data, and string
                    test_data += y_test[:, int(component)]
                    score_data += y_score[:, int(component)]
                    class_string += class_mapping[component] + " + "

                # Trim the final "+" and spacing from the class string.
                class_string = class_string[:-3]

            # If there's no "+" in this curve_class, that means the ROC curve is only for a single
            # class so there is no need to aggregate data across multiple classes.
            else:
                test_data = y_test[:, int(curve_class)]
                score_data = y_score[:, int(curve_class)]
                class_string = class_mapping[curve_class]

            # Calculate fpr, tpr, and the auc using sklearn.metrics.
            fpr, tpr, _ = roc_curve(test_data, score_data)
            roc_auc = auc(fpr, tpr)

            # Retrieve the name of the dataset used to perform the eval that generated this set
            # of predictions data.
            eval_dataset = prediction_file_content.options.dataset.config.split("/")[-1]
            model_name = prediction_file_content.options.model.name

            # Plot the curve. The label identifies this curve in the legend.
            ax.plot(fpr, tpr, lw=self.line_width,
                    label=f"'{class_string}' AUC: {roc_auc:0.2f} Source: {model_name}:{eval_dataset}")

    @staticmethod
    def _convert_classes(desired_classes: str, mapping: Prodict, current_file: str) -> List:
        """
        This method is responsible for converting the desired classes the user want to plot ROC
        curves for into a list of integers representing each class number. Users have the option to
        provide the desired classes for the plot as integers, human-readable strings, or the string 'all'
        which should translate to every class in the predictions file. This method performs any conversions
        that are required in order to produce a list of ONLY integer classes.
        :param desired_classes: A string indicating which classes the user want to produce ROC curves for.
        :param mapping: A dictionary of the class labels that the predictions file is aware of.
        :param current_file: A string indicating the current predictions file serving as a source of
        data for the ROC curves.
        :return: A List of integers of which classes to plot ROC curves for.
        """
        # If the user wants ROC curves for 'all' of the classes, retrieve every class integer
        # that the prediction file is aware of and put it in a list.
        if desired_classes == 'all':
            new_class_list = []
            for label in mapping.keys():
                new_class_list.append(label)
            return new_class_list

        # Otherwise, split the string representing the desired classes into a list of individual components.
        # The '+' is also one of the characters used to split because the user may want to join two
        # (or more) classes in the same curve, i.e. 1+2+dog. We want to isolate any class strings and
        # replace those with the proper integer.
        desired_classes_list = re.split(r'[,|+]', desired_classes)

        # Loop through every desired class in the list.
        for item in desired_classes_list:
            # Attempt to convert the current class into an integer. If it succeeds, then it's
            # already an integer and no further work is required.
            try:
                int(item)

            # If the conversion throws a ValueError, that means the desired class is a string.
            except ValueError:
                # Retrieve the desired class integer from the mapping and replace it in the
                # desired class string.
                if item in mapping.values():
                    key = next(key for key, value in mapping.items() if value == item)
                    desired_classes = desired_classes.replace(item, key)
                # If the class integer can't be determined, log an error and exit because the
                # class that the user wants to plot can't be determined.
                else:
                    logger.error(f"Unknown class: \"{item}\" not found in mapping of {current_file}")
                    sys.exit(-1)

        # Now that all strings have been replaced with integers, return the list of desired classes
        # split by just a ',' (which preserves any '+' in the string).
        return desired_classes.split(",")
