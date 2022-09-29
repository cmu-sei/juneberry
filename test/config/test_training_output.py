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

from pathlib import Path

from juneberry.config.training_output import TrainingOutputBuilder
from juneberry.config.model import ModelConfig
import utils


def test_builder(tmp_path):
    builder = TrainingOutputBuilder()

    mc = ModelConfig.from_dict(utils.make_basic_model_config())

    builder.set_from_model_config("test_config", mc)

    # The schema should require these things
    builder.output.options.num_training_images = 0
    builder.output.options.num_validation_images = 0
    builder.output.options.validation_dataset_config_path = ""
    builder.output.options.training_dataset_config_path = ""

    builder.output.results.accuracy = [0.0]
    builder.output.results.loss = [0.0]
    builder.output.results.model_hash = ""
    builder.output.results.val_accuracy = [0.0]
    builder.output.results.val_loss = [0.0]

    out_path = Path(tmp_path) / "tmp.json"
    builder.save(str(out_path))

