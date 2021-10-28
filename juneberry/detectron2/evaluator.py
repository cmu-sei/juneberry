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
from pathlib import Path
import sys
from torchvision import transforms
from types import SimpleNamespace

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

import juneberry.config.coco_utils as coco_utils
from juneberry.config.dataset import DatasetConfig
from juneberry.config.model import ModelConfig
import juneberry.data as jb_data
import juneberry.detectron2.data as dt2_data
from juneberry.evaluation.evaluator import Evaluator
from juneberry.evaluation.utils import get_histogram
from juneberry.filesystem import EvalDirMgr, ModelManager
from juneberry.jb_logging import setup_logger as jb_setup_logger
from juneberry.lab import Lab
import juneberry.metrics.metrics as metrics
import juneberry.pytorch.processing as processing


logger = logging.getLogger(__name__)


class Detectron2Evaluator(Evaluator):
    def __init__(self, model_config: ModelConfig, lab: Lab, dataset: DatasetConfig, model_manager: ModelManager,
                 eval_dir_mgr: EvalDirMgr, eval_options: SimpleNamespace = None):
        super().__init__(model_config, lab, dataset, model_manager, eval_dir_mgr, eval_options)

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

        # TODO: The evaluator base class should pass down the log path.
        log_file_path = self.eval_dir_mgr.get_log_dryrun_path() if self.dryrun else self.eval_dir_mgr.get_log_path()

        jb_setup_logger(log_file_path, "", name="fvcore", level=logging.DEBUG)
        jb_setup_logger(log_file_path, "", name="detectron2", level=logging.DEBUG)

    def check_gpu_availability(self, required: int):
        count = processing.determine_gpus(required)
        # TODO: Test to see if we can use more than 1 gpu with DP.
        if count > 1:
            logger.warning(f"The evaluator is only configured to support 1 gpu. Reducing {count} to 1")
            count = 1
        return count

    def setup(self) -> None:
        # If dryrun mode was requested, perform a "lighter" version of this method.
        if self.eval_opts.dryrun:
            self.perform_dryrun()

    def obtain_dataset(self) -> None:

        # TODO: Move this to a preprocessing step for distributed
        # Load the evaluation list
        label_names = self.dataset_config.retrieve_label_names()
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
        model_arch_name = self.model_config.model_architecture['module']

        self.cfg = get_cfg()

        # Add in the the configuration from the DT2 config file. These are in the DT2 package.
        self.cfg.merge_from_file(model_zoo.get_config_file(model_arch_name))

        # AOM - What is this for?
        # -- NV: This value is *way* too high; the blog post was kinda bogus
        # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8  # set threshold for this model

        self.cfg.MODEL.WEIGHTS = str(self.model_manager.get_pytorch_model_path())
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.dataset_config.num_model_classes

        if self.num_gpus == 0:
            self.cfg.MODEL.DEVICE = "cpu"

        if hasattr(self.model_config, "detectron2"):
            if "overrides" in self.model_config.detectron2:
                self.cfg.merge_from_list(self.model_config.detectron2['overrides'])

        self.predictor = DefaultPredictor(self.cfg)

    def evaluate_data(self) -> None:
        evaluator = COCOEvaluator(dataset_name=dt2_data.EVAL_DS_NAME, distributed=False, output_dir=self.output_dir)
        mapper = dt2_data.create_mapper(self.cfg, self.model_config.evaluation_transforms, False)
        eval_loader = build_detection_test_loader(self.cfg, dt2_data.EVAL_DS_NAME, mapper=mapper)
        self.eval_results = inference_on_dataset(self.predictor.model, eval_loader, evaluator)

        # Rename the results to our detections file for things like plot_pr
        det = Path(self.output_dir, "coco_instances_results.json")
        det.rename(self.eval_dir_mgr.get_detections_path())
        det = self.eval_dir_mgr.get_detections_path()

        # Populate metrics
        m = metrics.Metrics.create_with_filesystem_managers(self.model_manager, self.eval_dir_mgr)
        self.output.results.metrics.bbox = m.as_dict()
        self.output.results.metrics.bbox_per_class = m.mAP_per_class

    def format_evaluation(self) -> None:
        out = self.eval_dir_mgr.get_detections_anno_path()
        coco_utils.save_predictions_as_anno(self.data_root, str(self.dataset_config.file_path),
                                            str(self.eval_dir_mgr.get_detections_path()), out)

        # Sample some images from the annotations file.
        sample_dir = self.eval_dir_mgr.get_sample_detections_dir()
        coco_utils.generate_bbox_images(out, self.lab, sample_dir, sample_limit=20, shuffle=True)

        # Save the eval output to file.
        logger.info(f"Saving evaluation output to {self.eval_dir_mgr.get_predictions_path()}")
        self.output_builder.save_predictions(self.eval_dir_mgr.get_predictions_path())

    def perform_dryrun(self):
        """
        This method is responsible for executing "dryrun" mode for the detectron2 evaluator. In this mode,
        the evaluator will attempt to load the model config and the detectron2 config, check for the
        presence of the annotations file, and create some sample images. The evaluator will then exit
        without performing any actual evaluation tasks.
        :return:
        """

        # Establish a directory for storing dry run files and create it if it doesn't exist.
        dryrun_path = Path(self.eval_dir_mgr.get_dryrun_imgs_dir())
        if not dryrun_path.exists():
            dryrun_path.mkdir(parents=True)

        # Test the loading of the model config.
        logger.info(f"Attempting to load model config: {self.model_manager.get_model_config()}")
        try:
            model_config = ModelConfig.load(self.model_manager.get_model_config())
        except Exception:
            logger.exception(f"Error loading the model config.")
            raise
        else:
            logger.info(f"Finished loading model config.")

        # Test the loading of the evaluation data.
        logger.info(f"Attempting to load evaluation data.")
        try:
            jb_data.make_eval_manifest_file(self.lab, self.dataset_config,
                                            model_config, self.model_manager,
                                            self.use_train_split, self.use_val_split)
            dt2_data.register_eval_manifest_file(self.lab, self.model_manager, self.dataset_config)
        except Exception:
            logger.exception(f"Error loading the evaluation data.")
            raise
        else:
            logger.info(f"Finished generating eval manifest.")

        # Test the loading and manipulation of the detectron2 config.
        logger.info(f"Fetching the detectron2 config.")
        model_arch_name = model_config.model_architecture['module']
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(model_arch_name))
        cfg.MODEL.WEIGHTS = str(self.model_manager.get_model_dir() / "dt2output" / "model_final.pth")
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.dataset_config.num_model_classes

        if self.num_gpus == 0:
            cfg.MODEL.DEVICE = "cpu"

        if hasattr(model_config, "detectron2"):
            if "overrides" in model_config.detectron2:
                cfg.merge_from_list(model_config.detectron2['overrides'])

        # Save the resulting detectron2 config to file.
        dryrun_cfg_path = dryrun_path / "dt2_cfg.txt"
        logger.info(f"Saving the detectron2 config to {dryrun_cfg_path}")
        with open(dryrun_cfg_path, "w") as cfg_outfile:
            cfg_outfile.write(str(cfg))

        # Test for the presence of the annotations file.
        anno_file = Path(self.output_dir) / self.model_manager.get_eval_manifest_path(self.dataset_config.file_path)
        logger.info(f"Checking for annotations file at {anno_file}")
        if anno_file.exists():
            logger.info(f"Annotations file exists. It would be deleted and regenerated during the evaluation.")
        else:
            logger.info(f"Annotations file not found. A new one would be generated during evaluation.")

        # Test the creation of the detectron2 evaluator used to perform the evaluation.
        logger.info(f"Attempting to build the evaluator.")
        try:
            evaluator = COCOEvaluator(dataset_name=dt2_data.EVAL_DS_NAME, distributed=False, output_dir=self.output_dir)
        except Exception:
            logger.exception(f"Error building the evaluator.")
            raise
        else:
            logger.info(f"Evaluator built.")

        # Test the creation of the detectron2 test loader used to perform the evaluation.
        logger.info(f"Attempting to build the loader for the evaluator.")
        try:
            mapper = dt2_data.create_mapper(cfg, model_config.evaluation_transforms, False)
            eval_loader = build_detection_test_loader(cfg, dt2_data.EVAL_DS_NAME, mapper=mapper)
        except Exception:
            logger.exception(f"Error building the loader for the evaluator.")
            raise
        else:
            logger.info(f"Built the loader for the evaluator.")

        # The final piece of the dry run is to produce a few sample images from the evaluation dataloader.
        logger.info(f"Obtaining the first 5 sample images from the eval loader.")
        for sample_idx, image in enumerate(eval_loader):
            if sample_idx > 4:
                break

            save_path = Path(dryrun_path) / Path(image[0]['file_name']).name
            img = transforms.ToPILImage()(image[0]['image'])
            img.save(save_path)

        logger.info(f"Saved 5 sample images to {dryrun_path}")
        logger.info(f"Dry run complete.")

        sys.exit(0)
