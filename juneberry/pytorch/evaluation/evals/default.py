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
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import sys

from juneberry.config.training_output import TrainingOutput
import juneberry.evaluation.util as jb_eval_utils
import juneberry.filesystem as jbfs
from juneberry.pytorch.evaluation.pytorch_evaluator import PytorchEvaluator
import juneberry.pytorch.evaluation.util as jb_pytorch_eval_utils

logger = logging.getLogger(__name__)


class DefaultEvaluationProcedure:
    """
    This is the default PyTorch evaluation class used for evaluating data in Juneberry.
    """

    def __call__(self, evaluator: PytorchEvaluator):
        """
        When called, this method uses the attributes of the evaluator to conduct the evaluation. The result
        of the process is raw evaluation data.
        :param evaluator: The PytorchEvaluator object managing the evaluation.
        :return: Nothing.
        """

        # Perform the evaluation; saving the raw data to the correct evaluator attribute.
        evaluator.raw_output = jb_eval_utils.predict_classes(evaluator.eval_loader, evaluator.model, evaluator.device)

    @staticmethod
    def establish_evaluator(model_config, lab, dataset, model_manager, eval_dir_mgr, eval_options):
        return PytorchEvaluator(model_config, lab, dataset, model_manager, eval_dir_mgr, eval_options)


class DefaultEvaluationOutput:
    """
    This is the default PyTorch evaluation class used for formatting raw evaluation data in Juneberry.
    """

    def __call__(self, evaluator: PytorchEvaluator):
        """
        When called, this method uses the attributes of the evaluator to format the raw evaluation data. At the
        end of this call, the evaluator.output attribute will contain JSON-friendly data which will then be
        written to a file.
        :param evaluator: The PytorchEvaluator object managing the evaluation.
        :return: Nothing.
        """

        # Add the predicted labels for each image to the output.
        labels = [item[1] for item in evaluator.eval_name_targets]
        evaluator.output.results.labels = labels

        # Diagnostic for accuracy
        # TODO: Switch to configurable and standard accuracy
        is_binary = evaluator.eval_dataset_config.num_model_classes == 2
        predicted_classes = jb_eval_utils.continuous_predictions_to_class(evaluator.raw_output, is_binary)

        # Calculate the accuracy and add it to the output.
        logger.info(f"Computing the accuracy.")
        accuracy = accuracy_score(labels, predicted_classes)
        evaluator.output.results.metrics.accuracy = accuracy

        # Calculate the balanced accuracy and add it to the output.
        logger.info(f"Computing the balanced accuracy.")
        balanced_acc = balanced_accuracy_score(labels, predicted_classes)
        evaluator.output.results.metrics.balanced_accuracy = balanced_acc

        # Log the the accuracy values.
        logger.info(f"******          Accuracy: {accuracy:.4f}")
        logger.info(f"****** Balanced Accuracy: {balanced_acc:.4f}")

        # Save these as two classes if binary so it's consistent with other outputs.
        if is_binary:
            evaluator.raw_output = jb_eval_utils.binary_to_classes(evaluator.raw_output)

        # Add the raw prediction data to the output.
        evaluator.output.results.predictions = evaluator.raw_output

        # Add the dataset mapping and the number of classes the model is aware of to the output.
        evaluator.output.options.dataset.classes = evaluator.eval_dataset_config.label_names
        evaluator.output.options.model.num_classes = evaluator.eval_dataset_config.num_model_classes

        # Calculate the hash of the model that was used to conduct the evaluation.
        evaluated_model_hash = jbfs.generate_file_hash(evaluator.model_manager.get_pytorch_model_path())

        # If Juneberry was used to train the model, we can retrieve the hash from the training output file
        # and verify that the hash matches the model we used to evaluate the data.
        training_output_file_path = evaluator.model_manager.get_training_out_file()
        if training_output_file_path.is_file():
            training_output = TrainingOutput.load(training_output_file_path)
            hash_from_output = training_output.results.model_hash
            if hash_from_output != evaluated_model_hash:
                logger.error(f"The hash of the model used for evaluation does NOT match the hash in the training "
                             f"output file. EXITING.")
                logger.error(f"Expected: '{hash_from_output}' Found: '{evaluated_model_hash}'")
                sys.exit(-1)

        # Add the hash of the model used for evaluation to the output.
        evaluator.output.options.model.hash = evaluated_model_hash

        # If requested, get the top K classes predicted for each input.
        if evaluator.top_k:
            jb_pytorch_eval_utils.top_k_classifications(evaluator, evaluator.eval_dataset_config.label_names)

        # Save the predictions portion of the evaluation output to the appropriate file.
        evaluator.output_builder.save_predictions(evaluator.eval_dir_mgr.get_predictions_path())
        logger.info(f"Saving predictions to {evaluator.eval_dir_mgr.get_predictions_path()}")

        # Save the metrics portion of the evaluation output to the appropriate file.
        evaluator.output_builder.save_metrics(evaluator.eval_dir_mgr.get_metrics_path())
        logger.info(f"Saving metrics to {evaluator.eval_dir_mgr.get_metrics_path()}")
