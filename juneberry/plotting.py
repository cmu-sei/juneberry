#! /usr/bin/env python

"""
A set of plotting utilities.
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

import json

import matplotlib.pyplot as plt


def plot_means_stds_layers(title, means, stds, output_filename) -> None:
    """
    Generates a png plot to the specified file name that contains the means as a line
    and the standard deviations as error bars.
    :param title: The title for the plot.
    :param means: The means to plot.
    :param stds: The standard deviations.
    :param output_filename: The file in which to place the output.
    """
    plot_values_errors(title, means, stds, "Layers", "Means", output_filename)


def plot_values_errors(title, values, errors, x_label, y_label, output_name) -> None:
    """
    Generates a plot to the specified file name that contains the values as a line
    and the error values as error bars.
    :param title: The title for the plot.
    :param values: The means to plot.
    :param errors: The standard deviations.
    :param x_label: Label for the x-axis
    :param y_label: Label for the y-axis
    :param output_name: The file in which to place the output.
    """
    layers = list(range(len(values)))

    plt.plot(layers, values, linestyle='-', marker='o')
    plt.errorbar(layers, values, errors, fmt='ok', lw=3)
    plt.title(f"{y_label} across {x_label} of {title}")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(output_name)
    plt.close()


def plot_training_summary_chart(model_manager) -> None:
    """
    Plots the accuracies and losses from the training output into an image.
    :param model_manager: Model manager object that determines which model to process.
    """
    with open(model_manager.get_training_out_file()) as json_file:
        data = json.load(json_file)

    results = data['trainingResults']

    epochs = range(1, len(results['accuracy']) + 1)
    fig, ax1 = plt.subplots()
    plt.ylim(0.0, 1.0)

    ax1.set_xlabel('Epoch')

    # ================= Accuracy
    color = 'tab:red'
    ax1.set_ylabel('Accuracy', color=color)
    ax1.plot(epochs, results['accuracy'], linestyle='-', marker='', color=color, label="Accuracy")
    ax1.plot(epochs, results['valAccuracy'], linestyle='--', marker='', color=color, label="Validation Accuracy")
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)

    # ================= Loss
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Loss', color=color)
    ax2.plot(epochs, results['loss'], linestyle='-', marker='', color=color, label="Loss", )
    ax2.plot(epochs, results['valLoss'], linestyle='--', marker='', color=color, label="Validation Loss")
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2)

    # ================= General
    plt.title(f'Training results: {model_manager.model_name}')

    # otherwise the right y-label is slightly clipped
    fig.tight_layout()

    # Save to disk
    plt.savefig(model_manager.get_training_summary_plot())
