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
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import sys
from torch import split
from tqdm import tqdm

from juneberry.config.training_output import TrainingOutput
from juneberry.onnx.evaluator import Evaluator
import juneberry.evaluation.utils as jb_eval_utils
import juneberry.filesystem as jbfs
import juneberry.pytorch.evaluation.utils as jb_pytorch_eval_utils

logger = logging.getLogger(__name__)


class OnnxEvaluationProcedure:
    """
    Attempt at an ONNX eval procedure.
    """

    def __call__(self, evaluator: Evaluator):
        """
        When called, this method uses the attributes of the evaluator to conduct the evaluation. The result
        of the process is raw evaluation data.
        :param evaluator: The PytorchEvaluator object managing the evaluation.
        :return: Nothing.
        """

        self.evaluator = evaluator

        input_name = evaluator.ort_session.get_inputs()[0].name

        data_loader = evaluator.eval_loader

        # output_list = list()

        for i, (batch, target) in enumerate(tqdm(data_loader)):
            if evaluator.model_config.platform == "pytorch":
                sample = self.sample_pytorch_data(batch)
            elif evaluator.model_config.platform == "tensorflow":
                sample = self.sample_tensorflow_data(batch)
            elif evaluator.model_config.platform == "onnx":
                sample = self.sample_onnx_data(batch)
            else:
                sys.exit(-1)
            for item in sample:
                if evaluator.model_config.platform == "onnx":
                    session_input, ratio, image_id = item
                    ort_out = evaluator.ort_session.run([], {input_name: session_input})
                    evaluator.raw_output.append(self.convert_to_detection(ort_out, image_id, ratio))
                else:
                    ort_out = evaluator.ort_session.run([], {input_name: item})
                    ort_out = np.array(ort_out[0]).tolist()
                    evaluator.raw_output.append(ort_out[0])

    @staticmethod
    def establish_evaluator(model_config, lab, dataset, model_manager, eval_dir_mgr, eval_options):
        return Evaluator(model_config, lab, dataset, model_manager, eval_dir_mgr, eval_options)

    @staticmethod
    def sample_pytorch_data(batch):
        # Convert the individual tensors in the batch to numpy arrays and place them in
        # the input data list.

        return_list = []
        for item in split(batch, 1):
            return_list.append(item.data.numpy())

        return return_list

    @staticmethod
    def sample_tensorflow_data(batch):
        return_list = []
        for item in np.split(batch, batch.shape[0]):
            return_list.append(item.astype(np.float32))

        return return_list

    def sample_onnx_data(self, batch):
        return_list = []

        for img_path, img_id in batch:
            logger.info(f"Working on img {img_path}")
            img = Image.open(img_path)
            proc_img, ratio = self.preprocess(img)
            return_list.append((proc_img, ratio, img_id))

        return return_list

    @staticmethod
    def preprocess(image):
        # Resize
        ratio = 800.0 / min(image.size[0], image.size[1])
        image = image.resize((int(ratio * image.size[0]), int(ratio * image.size[1])), Image.BILINEAR)

        # Convert to BGR
        try:
            image = np.array(image)[:, :, [2, 1, 0]].astype('float32')
        except IndexError:
            stacked_image = np.stack((image,)*3, axis=-1)
            image = np.array(stacked_image)[:, :, [2, 1, 0]].astype('float32')

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

        return image, ratio

    @staticmethod
    def convert_to_detection(output, image_id, ratio):
        detections = []
        boxes, labels, scores = output
        boxes /= ratio
        for i in range(len(labels)):
            detections.append({"image_id": image_id, "category_id": int(labels[i]), "bbox": boxes[i].tolist(),
                               "score": float(scores[i])})
        return detections


class OnnxEvaluationOutput:
    """
    This is the default ONNX evaluation class used for formatting raw evaluation data in Juneberry.
    """

    def __call__(self, evaluator: Evaluator):
        """
        When called, this method uses the attributes of the evaluator to format the raw evaluation data. The
        result of the process is the evaluator.output attribute will contain JSON-friendly data, which will
        then be written to a file.
        :param evaluator: The Evaluator object managing the evaluation.
        :return: Nothing.
        """

        if evaluator.model_config.task == "objectDetection":
            import json

            with open(evaluator.eval_dir_mgr.get_detections_path(), "w") as det_file:
                json.dump(evaluator.raw_output, det_file, indent=4)

            from juneberry.evaluation.utils import populate_metrics
            populate_metrics(evaluator.model_manager, evaluator.eval_dir_mgr, evaluator.output)
            input("NOW WHAT?")
            logger.error(f"OD not implemented yet.")
            sys.exit(-1)
            # This probably goes here.


        # Add the predicted labels for each image to the output.
        labels = [item[1] for item in evaluator.eval_name_targets]
        evaluator.output.results.labels = labels

        # Diagnostic for accuracy
        # TODO: Switch to configurable and standard accuracy
        is_binary = evaluator.eval_dataset_config.num_model_classes == 2
        onnx_predicted_classes = jb_eval_utils.continuous_predictions_to_class(evaluator.raw_output, is_binary)

        # Calculate the accuracy and add it to the output.
        logger.info(f"Computing the accuracy.")
        onnx_accuracy = accuracy_score(labels, onnx_predicted_classes)
        evaluator.output.results.metrics.accuracy = onnx_accuracy

        # Calculate the balanced accuracy and add it to the output.
        logger.info(f"Computing the balanced accuracy.")
        onnx_balanced_acc = balanced_accuracy_score(labels, onnx_predicted_classes)
        evaluator.output.results.metrics.balanced_accuracy = onnx_balanced_acc

        # Log the the accuracy values.
        logger.info(f"******          Accuracy: {onnx_accuracy:.4f}")
        logger.info(f"****** Balanced Accuracy: {onnx_balanced_acc:.4f}")

        # Save these as two classes if binary so it's consistent with other outputs.
        if is_binary:
            evaluator.raw_output = jb_eval_utils.binary_to_classes(evaluator.raw_output)

        # Add the prediction data to the output.
        evaluator.output.results.predictions = evaluator.raw_output

        # Add the dataset mapping and the number of classes the model is aware of to the output.
        evaluator.output.options.dataset.classes = evaluator.eval_dataset_config.label_names
        evaluator.output.options.model.num_classes = evaluator.eval_dataset_config.num_model_classes

        # Calculate the hash of the model that was used to conduct the evaluation.
        evaluated_model_hash = jbfs.generate_file_hash(evaluator.model_manager.get_onnx_model_path())

        # If Juneberry was used to train the model, retrieve the hash from the training output file
        # and verify the hash matches the hash of the model used to evaluate the data.
        training_output_file_path = evaluator.model_manager.get_training_out_file()
        if training_output_file_path.is_file():
            training_output = TrainingOutput.load(training_output_file_path)
            hash_from_output = training_output.results.onnx_model_hash
            logger.info(f"Model hash retrieved from training output: {hash_from_output}")
            if hash_from_output != evaluated_model_hash:
                logger.error(f"Hash of the model that was just evaluated: '{evaluated_model_hash}'")
                logger.error(f"The hash of the model used for evaluation does NOT match the hash in the training "
                             f"output file. EXITING.")
                sys.exit(-1)
            else:
                logger.info(f"Hashes match! Hash of the evaluated model: {evaluated_model_hash}")

        # Add the hash of the model used for evaluation to the output.
        evaluator.output.options.model.hash = evaluated_model_hash

        # If requested, get the top K classes predicted for each input.
        if evaluator.top_k:
            jb_pytorch_eval_utils.top_k_classifications(evaluator, evaluator.eval_dataset_config.label_names)

        # Save the predictions portion of the evaluation output to the appropriate file.
        logger.info(f"Saving predictions to {evaluator.eval_dir_mgr.get_predictions_path()}")
        evaluator.output_builder.save_predictions(evaluator.eval_dir_mgr.get_predictions_path())

        # Save the metrics portion of the evaluation output to the appropriate file.
        logger.info(f"Saving metrics to {evaluator.eval_dir_mgr.get_metrics_path()}")
        evaluator.output_builder.save_metrics(evaluator.eval_dir_mgr.get_metrics_path())
