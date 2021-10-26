#! /usr/bin/env python

"""
A set of plotting utilities.
"""

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

import logging
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


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


def plot_loss(ax1, epochs1, epochs2, loss1, loss2, label1, label2):
    """Plots loss values.
    Generates a plot of accuracy values.
    :param epochs1: x values for loss 1.
    :param epochs2: x values for loss 2.
    :param accuracy1: y values for loss 1.
    :param accuracy2: y values for loss 2.
    :param label1: label for loss 1.
    :param label2: label for loss 2.
    """
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Loss', color=color)

    if loss1 is not None:
        ax2.plot(epochs1, loss1, linestyle='-', marker='', color=color, label=label1)
    if loss2 is not None:
        ax2.plot(epochs2, loss2, linestyle='--', marker='', color=color, label=label2)

    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2)


def plot_training_summary_chart(training_results, model_manager) -> None:
    """
    Plots the accuracies and losses from the training output into an image.
    :param training_results: The result of the training.
    :param model_manager: Model manager object that determines which model to process.
    """
    results = training_results['results']

    epochs = range(1, len(results['loss']) + 1)
    fig, ax1 = plt.subplots()
    plt.ylim(0.0, 1.0)

    ax1.set_xlabel('Epoch')

    # ================= Accuracy
    color = 'tab:red'
    ax1.set_ylabel('Accuracy', color=color)

    # The try/excepts were added in case data for the particular plot metric doesn't exist.
    try:
        ax1.plot(epochs, results['accuracy'], linestyle='-', marker='', color=color, label="Accuracy")
    except ValueError:
        pass
    try:
        ax1.plot(epochs, results['val_accuracy'], linestyle='--', marker='', color=color, label="Validation Accuracy")
    except ValueError:
        pass
    try:
        ax1.plot(epochs, results['sparse_categorical_accuracy'], linestyle='-', marker='', color=color,
                 label="Sparse Categorical Accuracy")
    except ValueError:
        pass
    try:
        ax1.plot(epochs, results['val_sparse_categorical_accuracy'], linestyle='-', marker='', color=color,
                 label="Validation Sparse Categorical Accuracy")
    except ValueError:
        pass

    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)

    # ================= Loss
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Loss', color=color)

    # The try/excepts were added in case data for the particular plot metric doesn't exist.
    try:
        ax2.plot(epochs, results['loss'], linestyle='-', marker='', color=color, label="Loss")
    except ValueError:
        pass
    try:
        ax2.plot(epochs, results['val_loss'], linestyle='--', marker='', color=color, label="Validation Loss")
    except ValueError:
        pass

    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2)

    # ================= General
    plt.title(f'Training results: {model_manager.model_name}')

    # otherwise the right y-label is slightly clipped
    fig.tight_layout()

    # Save to disk
    plt.savefig(model_manager.get_training_summary_plot())
    logger.info(f"Saving training summary plot to {model_manager.get_training_summary_plot()}")
