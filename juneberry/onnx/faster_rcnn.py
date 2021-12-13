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

import json
import logging
import numpy as np
from PIL import Image
from tqdm import tqdm

from juneberry.config.coco_anno import CocoAnnotations
import juneberry.config.coco_utils as jb_coco_utils
import juneberry.data as jb_data
from juneberry.evaluation.utils import get_histogram, populate_metrics
from juneberry.onnx.evaluator import Evaluator

logger = logging.getLogger(__name__)


class DataLoader:

    def __call__(self, evaluator: Evaluator):
        evaluator.output.options.dataset.classes = jb_data.get_label_mapping(model_manager=evaluator.model_manager,
                                                                             model_config=evaluator.model_config,
                                                                             train_config=None,
                                                                             eval_config=evaluator.eval_dataset_config)

        logger.info(f"Evaluating using label_names={evaluator.output.options.dataset.classes}")

        evaluator.eval_list, coco_data = jb_data.make_eval_manifest_file(evaluator.lab,
                                                                         evaluator.eval_dataset_config,
                                                                         evaluator.model_config,
                                                                         evaluator.model_manager,
                                                                         evaluator.use_train_split,
                                                                         evaluator.use_val_split)

        # Adds the histogram data for the evaluation list to the evaluation output.
        evaluator.output.options.dataset.histogram = get_histogram(evaluator.eval_list,
                                                                   evaluator.output.options.dataset.classes)

        coco_anno = CocoAnnotations.construct(coco_data)

        evaluator.eval_loader = []
        batch_size = evaluator.model_config.batch_size
        batch = []
        for image in coco_anno.images:
            if len(batch) < batch_size:
                batch.append((evaluator.lab.data_root() / image.file_name, image.id))
            else:
                evaluator.eval_loader.append((batch, None))
                batch = [(evaluator.lab.data_root() / image.file_name, image.id)]

        if batch:
            evaluator.eval_loader.append((batch, None))

        return


class EvaluationProcedure:
    def __call__(self, evaluator: Evaluator):
        input_name = evaluator.ort_session.get_inputs()[0].name

        data_loader = evaluator.eval_loader

        for i, (batch, target) in enumerate(tqdm(data_loader)):
            preprocessed_imgs = self.preprocess_img_batch(batch)

            for img in preprocessed_imgs:
                img_as_np_array, ratio, image_id = img
                ort_output = evaluator.ort_session.run([], {input_name: img_as_np_array})
                evaluator.raw_output += self.convert_to_detection(ort_output, image_id, ratio)

    def preprocess_img_batch(self, batch: list):
        return_list = []

        for img_path, img_id in batch:
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
            # The boxes, labels, and scores need to be converted to a format that is JSON-serializable.
            box = boxes[i].tolist()
            category_id = int(labels[i])
            score = float(scores[i])

            # The bbox also needs to be converted from LTRB to LTWH.
            box[2] = box[2] - box[0]
            box[3] = box[3] - box[1]

            detections.append({"image_id": image_id, "category_id": category_id, "bbox": box, "score": score})
        return detections


class EvaluationOutput:
    def __call__(self, evaluator: Evaluator):

        # Write detections file.
        logger.info(f"Saving the detections to {evaluator.eval_dir_mgr.get_detections_path()}")
        with open(evaluator.eval_dir_mgr.get_detections_path(), "w") as det_file:
            json.dump(evaluator.raw_output, det_file, indent=4)
        logger.info(f"Detections saved.")

        # Calculate the metrics.
        logger.info(f"Calculating the metrics for this evaluation...")
        populate_metrics(evaluator.model_manager, evaluator.eval_dir_mgr, evaluator.output)

        # Create COCO version of detections file.
        logger.info(f"Converting the detections to COCO-style annotations...")
        jb_coco_utils.save_predictions_as_anno(data_root=evaluator.lab.data_root(),
                                               dataset_config="",
                                               predict_file=evaluator.eval_dir_mgr.get_detections_path(),
                                               output_file=evaluator.eval_dir_mgr.get_detections_anno_path(),
                                               eval_manifest_path=evaluator.eval_dir_mgr.get_manifest_path()
                                               )
        logger.info(f"Saved the COCO-style detections to {evaluator.eval_dir_mgr.get_detections_anno_path()}")

        # Sample some images from the annotations file.

        logger.info(f"Generating some sample images with bboxes on the detections...")
        sample_dir = evaluator.eval_dir_mgr.get_sample_detections_dir()
        jb_coco_utils.generate_bbox_images(evaluator.eval_dir_mgr.get_detections_anno_path(), evaluator.lab,
                                           sample_dir, sample_limit=20, shuffle=True)
        logger.info(f"Finished generating sample images.")

        # Save the eval output to file.
        logger.info(f"Saving evaluation output to {evaluator.eval_dir_mgr.get_predictions_path()}")
        evaluator.output_builder.save_predictions(evaluator.eval_dir_mgr.get_predictions_path())
        logger.info(f"Evaluation output saved.")
