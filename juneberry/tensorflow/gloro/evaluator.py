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
from types import SimpleNamespace

from juneberry.config.dataset import DatasetConfig
from juneberry.config.model import ModelConfig
from juneberry.filesystem import ModelManager, EvalDirMgr
import juneberry.tensorflow.data as tf_data
import juneberry.tensorflow.evaluator

logger = logging.getLogger(__name__)


# HACK - This needs to go into the workspace
class Evaluator(juneberry.tensorflow.evaluator.Evaluator):
    def __init__(self, model_config: ModelConfig, lab, dataset: DatasetConfig, model_manager: ModelManager,
                 eval_dir_mgr: EvalDirMgr, eval_options: SimpleNamespace = None):
        super().__init__(model_config, lab, dataset, model_manager, eval_dir_mgr, eval_options)

    def obtain_model(self) -> None:
        from gloro import GloroNet
        model_file = self.model_manager.get_train_root_dir() / "model.gloronet"
        logger.info(f"Loading model {model_file}...")
        self.model = GloroNet.load_model(model_file)
        logger.info("...complete")
