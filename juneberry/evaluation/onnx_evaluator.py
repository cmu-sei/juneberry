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

import logging
import onnx
import onnxruntime as ort
import sys
from tqdm import tqdm
from types import SimpleNamespace

from juneberry.config.dataset import DatasetConfig
from juneberry.config.model import ModelConfig
from juneberry.evaluation.evaluator import Evaluator
from juneberry.filesystem import EvalDirMgr, ModelManager
from juneberry.lab import Lab
import juneberry.utils as jb_utils

logger = logging.getLogger(__name__)


class OnnxEvaluator(Evaluator):
    """
        This subclass is the ONNX-specific version of the Evaluator.
        """

    def __init__(self, model_config: ModelConfig, lab: Lab, dataset: DatasetConfig, model_manager: ModelManager,
                 eval_dir_mgr: EvalDirMgr, eval_options: SimpleNamespace = None):
        super().__init__(model_config, lab, dataset, model_manager, eval_dir_mgr, eval_options)

        self.input_data = []
        self.onnx_model = None
        self.ort_session = None

    def setup(self) -> None:
        """
        This is the ONNX version of the extension point that's responsible for setting up the Evaluator.
        :return: Nothing.
        """
        # Read the evaluation methods from the ModelConfig.
        self.eval_method = self.model_config.evaluation_procedure
        self.eval_output_method = self.model_config.evaluation_output

        # TODO: Shouldn't this be done in the lab??

        if self.model_config.hints is not None and 'num_workers' in self.model_config.hints.keys():
            num_workers = self.model_config.hints.num_workers
            logger.warning(f"Overriding number of workers. Found {num_workers} in ModelConfig")
            self.lab.num_workers = num_workers

        # Set the seeds using the value from the ModelConfig.
        jb_utils.set_seeds(self.model_config.seed)

        logger.info(f"ONNX Evaluator setup steps are complete.")

    def obtain_dataset(self) -> None:
        """
        This is the ONNX version of the extension point that's responsible for obtaining the
        dataset to be evaluated. The input_data is expected to be a list of individual tensors,
        where each tensor will be fed in to the evaluation procedure, one at a time.
        :return: Nothing.
        """
        if self.model_config.platform == "pytorch":
            from juneberry.pytorch.evaluation.pytorch_evaluator import PytorchEvaluator
            from torch import split
            evaluator = PytorchEvaluator(self.model_config, self.lab, self.eval_dataset_config, self.model_manager,
                                         self.eval_dir_mgr, None)
            evaluator.obtain_dataset()
            data_loader = evaluator.eval_loader
            self.eval_name_targets = evaluator.eval_name_targets.copy()

            logger.info(f"Converting the PyTorch dataloader into a format suitable for ONNX evaluation...")

            for i, (thing, target) in enumerate(tqdm(data_loader)):
                for item in split(thing, 1):
                    self.input_data.append(item.data.numpy())

        elif self.model_config.platform == "tensorflow":
            # TODO: Implement this.
            pass
        else:
            logger.info(f"ONNX evaluations are currently NOT supported for the {self.model_config.platform} platform.")
            sys.exit(-1)

    def obtain_model(self) -> None:
        """
        This is the ONNX version of the extension point that's responsible for obtaining the model
        to be evaluated.
        :return: Nothing.
        """
        # Load the ONNX model.
        self.onnx_model = onnx.load(self.model_manager.get_onnx_model_path())

        # Check that the ONNX model is well formed.
        onnx.checker.check_model(self.onnx_model)

        # TODO: Decide if this graph adds any value to the evaluation process.
        # logger.info(f"Graph of the ONNX model:\n{onnx.helper.printable_graph(self.onnx_model.graph)}")

    def evaluate_data(self) -> None:
        """
        This is the ONNX version of the extension point that's responsible for feeding the evaluation
        dataset into the model and obtaining the raw evaluation data. That process is usually defined in some
        external method, usually found in juneberry.evaluation.evals.
        :return: Nothing.
        """

        self.ort_session = ort.InferenceSession(str(self.model_manager.get_onnx_model_path()))

        if self.dryrun:
            logger.info(f"Dry run complete.")
            sys.exit(0)

        logger.info(f"Will evaluate model '{self.model_manager.model_name}' using {self.eval_dataset_config_path}")
        logger.info(f"Generating EVALUATION data according to {self.eval_method}")

        jb_utils.invoke_evaluator_method(self, self.eval_method)

        logger.info(f"EVALUATION COMPLETE.")

    def format_evaluation(self) -> None:
        """
        This is the Pytorch version of the extension point that's responsible for converting the raw
        evaluation data into the format the user wants. Much like evaluate_data, the actual process is
        usually defined in some external method, typically found in juneberry.pytorch.evaluation.
        :return: Nothing.
        """
        logger.info(f"Formatting raw EVALUATION data according to {self.eval_output_method}")

        jb_utils.invoke_evaluator_method(self, self.eval_output_method)

        logger.info(f"EVALUATION data has been formatted.")
