#! /usr/bin/env python3

# ======================================================================================================================
# Juneberry - Release 0.5
#
# Copyright 2022 Carnegie Mellon University.
#
# NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS"
# BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER
# INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED
# FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM
# FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.
#
# Released under a BSD (SEI)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.
#
# [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution. Please see
# Copyright notice for non-US Government use and distribution.
#
# This Software includes and/or makes use of Third-Party Software each subject to its own license.
# 
# DM22-0856
#
# ======================================================================================================================

import logging
from pathlib import Path
import sys
from types import SimpleNamespace

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from torchvision import transforms

import juneberry.config.coco_utils as coco_utils
from juneberry.config.dataset import DatasetConfig
from juneberry.config.eval_output import EvaluationOutput
from juneberry.config.model import ModelConfig
import juneberry.data as jb_data
import juneberry.detectron2.data as dt2_data
from juneberry.evaluation.evaluator import EvaluatorBase
from juneberry.evaluation.utils import get_histogram, get_default_od_metrics_config, get_default_od_metrics_formatter
from juneberry.filesystem import EvalDirMgr, ModelManager
from juneberry.logging import setup_logger as jb_setup_logger, RemoveDuplicatesFilter
from juneberry.lab import Lab
from juneberry.metrics.objectdetection.metrics_manager import MetricsManager
import juneberry.pytorch.processing as processing
from juneberry.pytorch.utils import PyTorchPlatformDefinitions

logger = logging.getLogger(__name__)


class Evaluator(EvaluatorBase):
    def __init__(self, model_config: ModelConfig, lab: Lab, model_manager: ModelManager, eval_dir_mgr: EvalDirMgr,
                 dataset: DatasetConfig, eval_options: SimpleNamespace = None, log_file: str = None):
        super().__init__(model_config, lab, model_manager, eval_dir_mgr, dataset, eval_options, log_file)

        self.lab = lab
        self.dataset_config = dataset
        self.model_manager = model_manager
        self.eval_opts = eval_options
        self.data_root = lab.data_root()
        self.dt2_dataset_name = "juneberry_eval"

        # Unique to the dt2 Evaluator
        self.eval_list = None
        self.predictor = None
        self.cfg = None
        self.eval_results = None

        # Establish an evaluation output directory and save the detectron2 logging messages
        self.output_dir = self.eval_dir_mgr.root

    # ==========================================================================

    @classmethod
    def get_platform_defs(cls):
        return PyTorchPlatformDefinitions()

    # ==========================================================================

    @classmethod
    def get_eval_output_files(cls, model_mgr: ModelManager, dataset_path: str, dryrun: bool = False):
        """
        Returns a list of files to clean from the eval directory. This list should contain ONLY
        files or directories that were produced by the evaluate command. Directories in this list
        will be deleted even if they are not empty.
        :param model_mgr: A ModelManager to help locate files.
        :param dataset_path: A string indicating the name of the dataset being evaluated.
        :param dryrun: When True, returns a list of files created during a dryrun of the Evaluator.
        :return: The files to clean from the eval directory.
        """
        eval_dir_mgr = model_mgr.get_eval_dir_mgr(dataset_path)
        if dryrun:
            return [eval_dir_mgr.get_dryrun_imgs_dir(),
                    eval_dir_mgr.get_manifest_path()]
        else:
            return [eval_dir_mgr.get_sample_detections_dir(),
                    eval_dir_mgr.get_detections_path(),
                    eval_dir_mgr.get_detections_anno_path(),
                    eval_dir_mgr.get_manifest_path(),
                    eval_dir_mgr.get_instances_predictions_file(),
                    eval_dir_mgr.get_metrics_path()]

    @classmethod
    def get_eval_clean_extras(cls, model_mgr: ModelManager, dataset_path: str, dryrun: bool = False):
        """
        Returns a list of extra "evaluation" files to clean. Directories in this list will NOT
        be deleted if they are not empty.
        :param model_mgr: A ModelManager to help locate files.
        :param dataset_path: A string indicating the name of the dataset being evaluated.
        :param dryrun: When True, returns a list of files created during a dryrun of the Trainer.
        :return: The extra files to clean from the training directory.
        """
        eval_dir_mgr = model_mgr.get_eval_dir_mgr(dataset_path)
        if dryrun:
            return [eval_dir_mgr.get_dir(),
                    eval_dir_mgr.get_dir().parent]
        else:
            return [eval_dir_mgr.get_dir(),
                    eval_dir_mgr.get_dir().parent]

    @classmethod
    def get_default_metric_value(cls, eval_data: EvaluationOutput):
        """ :return: The value of the Evaluator's default metric as found in the results structure """
        return eval_data.results.metrics.bbox['mAP'], "mAP"

    # ==========================================================================
    def dry_run(self) -> None:
        self.dryrun = True
        dryrun_path = Path(self.eval_dir_mgr.get_dryrun_imgs_dir())
        dryrun_path.mkdir(parents=True, exist_ok=True)

        self.setup()
        self.obtain_dataset()
        self.obtain_model()

        # Test for the presence of the annotations file.
        anno_file = Path(self.output_dir) / self.eval_dir_mgr.get_manifest_path()
        logger.info(f"Checking for annotations file at {anno_file}")
        if anno_file.exists():
            logger.info(f"Annotations file exists. It would be deleted and regenerated during the evaluation.")
        else:
            logger.info(f"Annotations file not found. A new one would be generated during evaluation.")

        # Test the creation of the detectron2 evaluator used to perform the evaluation.
        logger.info(f"Attempting to build the evaluator.")
        try:
            evaluator = COCOEvaluator(dataset_name=dt2_data.EVAL_DS_NAME, distributed=False,
                                      output_dir=self.output_dir)
        except Exception:
            logger.exception(f"Error building the evaluator.")
            raise
        else:
            logger.info(f"Evaluator built.")

        # Test the creation of the detectron2 test loader used to perform the evaluation.
        logger.info(f"Attempting to build the loader for the evaluator.")
        try:
            mapper = dt2_data.create_mapper(self.cfg, self.model_config.evaluation_transforms, False)
            eval_loader = build_detection_test_loader(self.cfg, dt2_data.EVAL_DS_NAME, mapper=mapper)
        except Exception:
            logger.exception(f"Error building the loader for the evaluator.")
            raise
        else:
            logger.info(f"Built the loader for the evaluator.")

        # Now produce a few sample images from the evaluation dataloader.
        logger.info(f"Obtaining the first 5 sample images from the eval loader.")
        for sample_idx, image in enumerate(eval_loader):
            if sample_idx > 4:
                break

            save_path = Path(self.eval_dir_mgr.get_dryrun_imgs_dir()) / Path(image[0]['file_name']).name
            img = transforms.ToPILImage()(image[0]['image'])
            img.save(save_path)

        logger.info(f"Saved 5 sample images to {self.eval_dir_mgr.get_dryrun_imgs_dir()}")

        logger.info(f"Dryrun complete.")

    # ==========================================================================

    def check_gpu_availability(self, required: int):
        count = processing.determine_gpus(required)
        # TODO: Test to see if we can use more than 1 gpu with DP.
        if count > 1:
            logger.warning(f"The evaluator is only configured to support 1 GPU. Reducing {count} to 1.")
            count = 1
        return count

    def setup(self) -> None:
        jb_setup_logger(self.log_file_path, "", name="fvcore", level=logging.DEBUG)
        jb_setup_logger(self.log_file_path, "", name="detectron2", level=logging.DEBUG)
        jb_setup_logger(self.log_file_path, "", name="brambox", level=logging.DEBUG,
                        log_filter_class=RemoveDuplicatesFilter)

    def obtain_dataset(self) -> None:

        # TODO: Move this to a preprocessing step for distributed
        # Load the evaluation list
        label_names = jb_data.get_label_mapping(model_manager=self.model_manager, model_config=self.model_config,
                                                train_config=self.dataset_config, eval_config=self.eval_dataset_config)
        logger.info(f"Evaluating using label_names={label_names}")

        self.eval_list, _ = jb_data.make_eval_manifest_file(self.lab, self.dataset_config, self.model_config,
                                                            self.model_manager,
                                                            self.use_train_split, self.use_val_split)

        dt2_data.register_eval_manifest_file(self.lab, self.model_manager, self.dataset_config)

        # Adds the histogram data for the evaluation list to the evaluation output.
        self.output.options.dataset.classes = label_names
        self.output.options.dataset.histogram = get_histogram(self.eval_list, label_names)

    def obtain_model(self) -> None:
        # Set up the predictor
        model_arch_name = self.model_config.model_architecture.fqcn

        self.cfg = get_cfg()

        # Add in the configuration from the DT2 config file. These are in the DT2 package.
        self.cfg.merge_from_file(model_zoo.get_config_file(model_arch_name))

        # AOM - What is this for?
        # -- NV: This value is *way* too high; the blog post was kinda bogus
        # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8  # set threshold for this model

        # Identify the model file.
        model_path = self.model_manager.get_model_path(self.get_platform_defs())

        # If the model file exists, load the weights.
        if model_path.exists():
            self.cfg.MODEL.WEIGHTS = str(model_path)
            self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.dataset_config.num_model_classes

        # If the model file doesn't exist...
        else:
            # A missing model file is not a big deal for a dryrun, just inform that the weights
            # could not be loaded.
            if self.dryrun:
                logger.info(f"Did not load model weights. {model_path} does not exist.")

            # If there's no model file and it's not a dryrun, then this Evaluator will eventually
            # fail log an error and exit.
            else:
                logger.error(f"Failed to load model. File does not exist: {model_path}")
                sys.exit(-1)

        if self.num_gpus == 0:
            self.cfg.MODEL.DEVICE = "cpu"

        if hasattr(self.model_config, "detectron2"):
            if "overrides" in self.model_config.detectron2:
                self.cfg.merge_from_list(self.model_config.detectron2['overrides'])

        if model_path.exists():
            self.predictor = DefaultPredictor(self.cfg)
        else:
            logger.warning(f"Model weights could not be loaded because the following model file was not "
                           f"found: {model_path}")

    def evaluate_data(self) -> None:
        evaluator = COCOEvaluator(dataset_name=dt2_data.EVAL_DS_NAME, distributed=False, output_dir=self.output_dir)
        mapper = dt2_data.create_mapper(self.cfg, self.model_config.evaluation_transforms, False)
        eval_loader = build_detection_test_loader(self.cfg, dt2_data.EVAL_DS_NAME, mapper=mapper)
        self.eval_results = inference_on_dataset(self.predictor.model, eval_loader, evaluator)

        # Rename the results to our detections file for things like plot_pr
        det = Path(self.output_dir, "coco_instances_results.json")
        det.rename(self.eval_dir_mgr.get_detections_path())

    def populate_metrics(self) -> None:
        """
        Calculate metrics and populate the metrics output with the result.
        :return: None
        """
        metrics_config = self.model_config.evaluation_metrics
        metrics_formatter = self.model_config.evaluation_metrics_formatter

        if not metrics_config:
            metrics_config = get_default_od_metrics_config()
            metrics_formatter = get_default_od_metrics_formatter()

        metrics_mgr = MetricsManager(metrics_config, metrics_formatter)
        self.output.results.metrics = metrics_mgr.call_with_eval_dir_manager(self.eval_dir_mgr)

    def format_evaluation(self) -> None:
        out = self.eval_dir_mgr.get_detections_anno_path()

        # Find category list
        eval_manifest_path = self.eval_dir_mgr.get_manifest_path()
        category_list = jb_data.get_category_list(eval_manifest_path=eval_manifest_path,
                                                  model_manager=self.model_manager,
                                                  train_config=self.dataset_config,
                                                  eval_config=self.eval_dataset_config,
                                                  data_root=self.data_root)

        # Save as coco annotations file
        coco_utils.save_predictions_as_anno(data_root=self.data_root, dataset_config=str(self.dataset_config.file_path),
                                            predict_file=str(self.eval_dir_mgr.get_detections_path()),
                                            category_list=category_list, output_file=out,
                                            eval_manifest_path=eval_manifest_path)

        # Sample some images from the annotations file.
        sample_dir = self.eval_dir_mgr.get_sample_detections_dir()
        coco_utils.generate_bbox_images(out, self.lab, sample_dir, sample_limit=20, shuffle=True)

        # Save the eval output to file.
        logger.info(f"Saving evaluation output to {self.eval_dir_mgr.get_metrics_path()}")
        self.output_builder.save_predictions(self.eval_dir_mgr.get_metrics_path())
