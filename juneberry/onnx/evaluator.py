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
import onnx
import onnxruntime as ort
import sys
from types import SimpleNamespace

from juneberry.config.coco_anno import CocoAnnotations
from juneberry.config.dataset import DatasetConfig
from juneberry.config.model import ModelConfig
from juneberry.evaluation.evaluator import EvaluatorBase
import juneberry.evaluation.utils as jb_eval_utils
from juneberry.filesystem import EvalDirMgr, ModelManager
from juneberry.lab import Lab
import juneberry.utils as jb_utils

import numpy as np
from PIL import Image


logger = logging.getLogger(__name__)


class Evaluator(EvaluatorBase):
    """
        This subclass is the ONNX-specific version of the Evaluator.
        """

    def __init__(self, model_config: ModelConfig, lab: Lab, model_manager: ModelManager, eval_dir_mgr: EvalDirMgr,
                 dataset: DatasetConfig, eval_options: SimpleNamespace = None, **kwargs):
        super().__init__(model_config, lab, model_manager, eval_dir_mgr, dataset, eval_options, **kwargs)

        self.input_data = []
        self.onnx_model = None
        self.ort_session = None
        self.raw_output = []
        self.eval_loader = None

    def setup(self) -> None:
        """
        This is the ONNX version of the extension point that's responsible for setting up the Evaluator.
        :return: Nothing.
        """

        # TODO: Shouldn't this be done in the lab??

        if self.model_config.hints is not None and 'num_workers' in self.model_config.hints.keys():
            num_workers = self.model_config.hints.num_workers
            logger.warning(f"Overriding number of workers. Found {num_workers} in ModelConfig")
            self.lab.num_workers = num_workers

        # Set the seeds using the value from the ModelConfig.
        jb_utils.set_seeds(self.model_config.seed)

        # Use default values if they were not provided in the model config.
        if self.eval_method is None:
            self.eval_method = "juneberry.onnx.default.OnnxEvaluationProcedure"
        if self.eval_output_method is None:
            self.eval_output_method = "juneberry.onnx.default.OnnxEvaluationOutput"

        logger.info(f"ONNX Evaluator setup steps are complete.")

    def obtain_dataset(self) -> None:
        """
        This is the ONNX version of the extension point that's responsible for obtaining the
        dataset to be evaluated. The input_data is expected to be a list of individual tensors,
        where each tensor will be fed in to the evaluation procedure, one at a time.
        :return: Nothing.
        """

        # TODO: I think there's a risk here if the datasets are too large to fit in memory.
        #  self.input_data could end up being very large.

        # If a PyTorch model is being evaluated, create a separate PyTorch specific evaluator
        # and use it to construct a PyTorch dataloader. Once the dataloader exists, convert it
        # into the format that ONNX expects.
        if self.model_config.platform == "pytorch":
            from juneberry.pytorch.evaluation.evaluator import Evaluator

            # Create a PytorchEvaluator and use it to build a PyTorch dataloader for the input data.
            evaluator = Evaluator(self.lab, self.model_config, self.model_manager, self.eval_dir_mgr,
                                  self.eval_dataset_config, None)
            evaluator.obtain_dataset()
            self.eval_loader = evaluator.eval_loader

            # Retrieve the labels for the input data.
            self.eval_name_targets = evaluator.eval_name_targets.copy()

        # This bit will be responsible for converting the TensorFlow input data into the format ONNX expects.
        elif self.model_config.platform == "tensorflow":
            from juneberry.tensorflow.evaluation.evaluator import Evaluator
            evaluator = Evaluator(self.model_config, self.lab, self.model_manager, self.eval_dir_mgr,
                                  self.eval_dataset_config, None)
            evaluator.obtain_dataset()
            self.eval_loader = evaluator.eval_loader

            self.eval_name_targets = evaluator.eval_labels
            self.eval_name_targets = [('', x) for x in self.eval_name_targets]

        elif self.model_config.platform == "onnx":
            import juneberry.data as jb_data
            from juneberry.evaluation.utils import get_histogram, populate_metrics

            label_names = jb_data.get_label_mapping(model_manager=self.model_manager, model_config=self.model_config,
                                                    train_config=None,
                                                    eval_config=self.eval_dataset_config)
            logger.info(f"Evaluating using label_names={label_names}")

            self.eval_list, coco_data = jb_data.make_eval_manifest_file(self.lab, self.eval_dataset_config,
                                                                        self.model_config, self.model_manager,
                                                                        self.use_train_split, self.use_val_split)

            # Adds the histogram data for the evaluation list to the evaluation output.
            self.output.options.dataset.classes = label_names
            self.output.options.dataset.histogram = get_histogram(self.eval_list, label_names)

            coco_anno = CocoAnnotations.construct(coco_data)
            self.eval_loader = []
            batch_size = 10
            batch = []
            for image in coco_anno.images:
                if len(batch) < batch_size:
                    batch.append((self.lab.data_root() / image.file_name, image.id))
                else:
                    self.eval_loader.append((batch, None))
                    batch = [(self.lab.data_root() / image.file_name, image.id)]

            if batch:
                self.eval_loader.append((batch, None))

            # TODO: END HERE

            # inputs_num = len(self.eval_loader)
            # import onnx_tf.backend as backend
            #
            # detections = []
            # self.onnx_model = onnx.load(self.model_manager.get_onnx_model_path())
            #
            # from tqdm import tqdm
            # for i in tqdm(range(inputs_num)):
            #     input_file, image_id = self.eval_loader[i]
            #     img = Image.open(input_file)
            #     ratio = 800.0 / min(img.size[0], img.size[1])
            #     img_data = preprocess(img)
            #     output = list(backend.run_model(self.onnx_model, img_data))
            #     det_list = convert_to_detection(output, image_id, ratio)
            #     for detection in det_list:
            #         detections.append(detection)
            #
            #     if i == 1:
            #         break
            #
            # import json
            # with open(self.eval_dir_mgr.get_detections_path(), "w") as det_file:
            #     json.dump(detections, det_file, indent=4)
            #
            # populate_metrics(self.model_manager, self.eval_dir_mgr, self.output)

            # TODO: END OF GOOD CODE
            # import glob
            # import os
            # from onnx import numpy_helper
            # import onnx_tf.backend as backend
            # import numpy as np
            #
            # model = onnx.load(self.model_manager.get_onnx_model_path())
            # test_data_dir = 'data_sets/test_data_set_0'
            #
            # # Load inputs
            # inputs = []
            # inputs_num = len(glob.glob(os.path.join(test_data_dir, 'input_*.pb')))
            # for i in range(inputs_num):
            #     input_file = os.path.join(test_data_dir, 'input_{}.pb'.format(i))
            #     tensor = onnx.TensorProto()
            #     with open(input_file, 'rb') as f:
            #         tensor.ParseFromString(f.read())
            #     inputs.append(numpy_helper.to_array(tensor))
            #
            # # Load reference outputs
            # ref_outputs = []
            # ref_outputs_num = len(glob.glob(os.path.join(test_data_dir, 'output_*.pb')))
            # for i in range(ref_outputs_num):
            #     output_file = os.path.join(test_data_dir, 'output_{}.pb'.format(i))
            #     tensor = onnx.TensorProto()
            #     with open(output_file, 'rb') as f:
            #         tensor.ParseFromString(f.read())
            #     ref_outputs.append(numpy_helper.to_array(tensor))
            #
            # # Run the model on the backend
            # outputs = list(backend.run_model(model, inputs))
            #
            # # Compare the results with reference outputs.
            # for ref_o, o in zip(ref_outputs, outputs):
            #     np.testing.assert_almost_equal(ref_o, o)

        # Handle cases where the model platform does not support an ONNX evaluation.
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

        jb_eval_utils.invoke_evaluator_method(self, self.eval_method)

        logger.info(f"EVALUATION COMPLETE.")

    def format_evaluation(self) -> None:
        """
        This is the ONNX version of the extension point that's responsible for converting the raw
        evaluation data into the format the user wants. Much like evaluate_data, the actual process is
        usually defined in some external method, typically found in juneberry.pytorch.evaluation.
        :return: Nothing.
        """
        logger.info(f"Formatting raw EVALUATION data according to {self.eval_output_method}")

        jb_eval_utils.invoke_evaluator_method(self, self.eval_output_method)

        logger.info(f"EVALUATION data has been formatted.")


def preprocess(image):
    # Resize
    ratio = 800.0 / min(image.size[0], image.size[1])
    image = image.resize((int(ratio * image.size[0]), int(ratio * image.size[1])), Image.BILINEAR)

    # Convert to BGR
    image = np.array(image)[:, :, [2, 1, 0]].astype('float32')

    # HWC -> CHW
    image = np.transpose(image, [2, 0, 1])

    # Normalize
    mean_vec = np.array([102.9801, 115.9465, 122.7717])
    for i in range(image.shape[0]):
        image[i, :, :] = image[i, :, :] - mean_vec[i]

    # Pad to be divisible of 32
    import math
    padded_h = int(math.ceil(image.shape[1] / 32) * 32)
    padded_w = int(math.ceil(image.shape[2] / 32) * 32)

    padded_image = np.zeros((3, padded_h, padded_w), dtype=np.float32)
    padded_image[:, :image.shape[1], :image.shape[2]] = image
    image = padded_image

    return image


def convert_to_detection(output, image_id, ratio):
    detections = []
    boxes, labels, scores = output
    boxes /= ratio
    for i in range(len(labels)):
        # print(type(labels[i]), type(boxes[i]), type(scores[i]))
        # input("Pause")
        detections.append({"image_id": image_id, "category_id": int(labels[i]), "bbox": boxes[i].tolist(),
                           "score": float(scores[i])})
    return detections

