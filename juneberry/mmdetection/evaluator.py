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

from collections import OrderedDict
import itertools
import logging
import mmcv
import numpy as np
from pathlib import Path
import sys
from types import SimpleNamespace
import warnings

# Multi-gpu needs to do MMDDP
from mmcv.parallel import MMDataParallel

# Multi-gpu needs to do init_dist
# from mmcv.runner import (init_dist, load_checkpoint)
from mmcv.runner import load_checkpoint

from mmdet.apis import single_gpu_test
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.datasets.api_wrappers import COCOeval
from mmdet.datasets.coco import CocoDataset
from mmdet.models import build_detector

import juneberry.config.coco_utils as coco_utils
from juneberry.config.dataset import DatasetConfig
from juneberry.config.model import ModelConfig
import juneberry.data as jb_data
from juneberry.evaluation.evaluator import EvaluatorBase
from juneberry.evaluation.utils import get_histogram, get_metrics
import juneberry.filesystem as jbfs
from juneberry.filesystem import EvalDirMgr, ModelManager
from juneberry.jb_logging import setup_logger as jb_setup_logger
from juneberry.lab import Lab
import juneberry.mmdetection.utils as mmd_utils
import juneberry.pytorch.processing as processing

logger = logging.getLogger(__name__)


class Evaluator(EvaluatorBase):
    def __init__(self, model_config: ModelConfig, lab: Lab, model_manager: ModelManager, eval_dir_mgr: EvalDirMgr,
                 dataset: DatasetConfig, eval_options: SimpleNamespace = None, log_file: str = None):
        super().__init__(model_config, lab, model_manager, eval_dir_mgr, dataset, eval_options, log_file)

        self.mm_home = mmd_utils.find_mmdetection()

        # The mmdetection cfg
        self.cfg = None
        self.working_dir = self.eval_dir_mgr.root
        self.eval_list = None
        self.eval_coco_anno = None
        self.dataset = None
        self.data_loader = None
        self.eval_options = eval_options

    # ==========================================================================
    def dry_run(self) -> None:
        self.setup()
        self.obtain_dataset()
        self.obtain_model()

        logger.info(f"Dryrun complete.")

    # ==========================================================================

    def check_gpu_availability(self, required: int):
        count = processing.determine_gpus(required)
        # TODO: See if we can get the multigpu api working.
        if count > 1:
            logger.warning(f"The evaluator is only configured to support 1 GPU. Reducing {count} to 1.")
            count = 1
        return count

    def setup(self) -> None:
        jb_setup_logger(self.log_file_path, "", name="mmdet", level=logging.DEBUG)

        # Setup working dir to save files and logs.
        logger.info(f"Using working directory of: {self.working_dir}")
        if not self.working_dir.exists():
            logger.info(f"Making working dir {str(self.working_dir)}")
            self.working_dir.mkdir(parents=True)

        ds_cfg = self.eval_dataset_config

        # Load cfg based on what they said in the model.
        # TODO: Should we read in the one we __actually__ saved during training if available?
        cfg = mmcv.Config.fromfile(str(self.mm_home / "configs" / self.model_config.model_architecture['module']))
        cfg.work_dir = str(self.working_dir)

        # Get rid of any training config bits.
        cfg.model.train_cfg = None
        cfg.model.pretrained = None

        # ============ Setup Datasets

        # NOTE: The make_eval_manifest_file has SIDE EFFECTS on the ds_cfg and updates the classes.
        self.eval_list, self.eval_coco_anno = jb_data.make_eval_manifest_file(
            self.lab, ds_cfg, self.model_config,
            self.model_manager,
            self.use_train_split, self.use_val_split)

        # Get the class names from the dataset.  This must happen AFTER we load the eval file.
        label_names = jb_data.get_label_mapping(model_manager=self.model_manager, model_config=self.model_config,
                                                eval_config=self.eval_dataset_config_path)
        classes = list(label_names.values())
        logger.info(f"Using classes={classes}")

        cfg.dataset_type = 'COCODataset'
        cfg.data_root = str(self.lab.data_root())

        coco_path = self.model_manager.get_eval_manifest_path(ds_cfg.file_path).resolve()
        cfg.data.test.data_root = str(self.lab.data_root().resolve())
        cfg.data.test.ann_file = str(coco_path.resolve())
        cfg.data.test.img_prefix = ""
        cfg.data.test.classes = classes
        cfg.data.test.test_mode = True

        # We don't use these, so remove.
        cfg.data.train = {}
        cfg.data.val = {}

        # Similar to trainer using the predefined model architecture, but this time use the
        # training output instead of the mmdetection trained checkpoint.
        model_path = self.model_manager.get_pytorch_model_path()
        if not model_path.exists():
            logger.error(f"Trained model {model_path} does not exist. EXITING.")
            sys.exit(-1)
        # This does NOT actually get the file loaded.
        cfg.load_from = str(model_path.resolve())

        # Set seed, thus the results are more reproducible.
        mmd_utils.add_reproducibility_configuration(self.model_config, cfg)

        # For eval, we only do one gpu.
        cfg.gpu_ids = range(1)

        # These metrics works for our style of dataset.
        cfg.evaluation.metric = 'bbox'

        # TODO: Should this be in obtain_dataset?
        # TODO: They have complex logic for determining samples per gpu I need to understand.
        cfg.data.samples_per_gpu = 1

        # Add in the pipelines overrides.
        mmd_utils.adjust_pipelines(self.model_config, cfg)

        # Bring all the user defined configuration.
        mmd_utils.add_config_overrides(self.model_config, cfg)

        # This output should be EXACTLY what we used, so we should be able to feed
        # this into mmdetection's test.py.
        out_path = self.model_manager.get_platform_eval_config(self.eval_dataset_config_path, 'py')
        with open(out_path, "w") as out_cfg:
            logger.info(f"Writing out config to: {out_path}")
            out_cfg.write(cfg.pretty_text)

        self.cfg = cfg

    def obtain_dataset(self) -> None:
        # Build the dataset.
        self.dataset = build_dataset(self.cfg.data.test)

        # Add the dataset histogram information to the evaluation output.
        classes = jb_data.get_label_mapping(model_manager=self.model_manager, model_config=self.model_config,
                                            eval_config=self.eval_dataset_config)
        self.output.options.dataset.classes = classes
        self.output.options.dataset.histogram = get_histogram(self.eval_list, classes)

    def obtain_model(self) -> None:
        cfg = self.cfg

        # Build the dataloader.
        self.data_loader = build_dataloader(
            self.dataset,
            samples_per_gpu=cfg.data.samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            # dist=distributed,
            dist=False,
            shuffle=False)

        # Build the model and load checkpoint.
        model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
        model.CLASSES = self.dataset.CLASSES

        # The "load_from" does not actually load the values! WE MUST do this.
        checkpoint = load_checkpoint(model, cfg.load_from, map_location='cpu')

        # Right now we only support single CPU/GPU.
        # NOTE: If we are in CPU mode (not GPUs) this just seems to figure it out by itself.
        # So, we just always use dataparallel with a device of 0.
        # TODO: See what happens if we do NOT use MMDataParallel in CPU mode.
        # if self.num_gpus == 1:
        #     self.model = MMDataParallel(model, device_ids=[0])
        self.model = MMDataParallel(model, device_ids=[0])

    def evaluate_data(self) -> None:
        self.raw_output = single_gpu_test(model=self.model,
                                          data_loader=self.data_loader,
                                          show=False,
                                          out_dir=None)

    def format_evaluation(self) -> None:
        # The result data is in the format:
        # NOTE: It is currently unclear what the score is
        # [batch] x [num_classes] x [ bbox(l,t,r,b] + score ]

        # Grab category mapping
        eval_manifest_path = self.model_manager.get_eval_manifest_path(self.eval_dataset_config.file_path)
        category_list = jb_data.get_category_list(eval_manifest_path=eval_manifest_path,
                                                  model_manager=self.model_manager,
                                                  # train_config = self.dataset_config,
                                                  eval_config=self.eval_dataset_config,
                                                  data_root=self.lab.data_root())

        # Convert to standard coco-style annotations.
        coco_annotations = make_coco_annotations(self.eval_coco_anno, self.raw_output, category_list)

        instances_path = self.eval_dir_mgr.get_detections_path()
        logger.info(f"Saving coco detections to: {instances_path}")
        jbfs.save_json(coco_annotations['annotations'], instances_path)

        instances_anno_path = self.eval_dir_mgr.get_detections_anno_path()
        logger.info(f"Saving coco detections (anno format) to: {instances_anno_path}")
        jbfs.save_json(coco_annotations, instances_anno_path)

        # Sample some images from the annotations file.
        sample_dir = self.eval_dir_mgr.get_sample_detections_dir()
        coco_utils.generate_bbox_images(instances_anno_path, self.lab, sample_dir, sample_limit=20, shuffle=True)

        # For now just print the results.
        # Now output the bbox evaluation.
        # TODO: Look at how they bring in metrics flags; are there other ones we should accept?
        # TODO: Output metrics using brambox?
        # TODO: The logger argument to the evaluate says "Logger used for printing
        #       related information during evaluation." but it doesn't seem to come out as log entries.
        result = JBMMDCocoDataset.evaluate(self=self.dataset, results=self.raw_output,
                                           metric=self.cfg.evaluation.metric, logger=logger, classwise=True)

        self.output_builder.save_predictions(self.eval_dir_mgr.get_metrics_path())

        # TODO: Add some samples. Refactor the code out of DT2.

    def populate_metrics(self) -> None:
        metrics = get_metrics(self.model_config, self.eval_dir_mgr)["juneberry.metrics.metrics.Coco"]
        self.output.results.metrics.bbox = metrics["bbox"]
        self.output.results.metrics.bbox_per_class = metrics["bbox_per_class"]

def make_coco_annotations(input_anno, outputs, category_list):
    # Inputs are the images from the input set.
    # Outputs come from the model and look like:
    # [batch] x [num_classes] x [ bbox(l,t,r,b] + confidence ]
    annos = []
    anno_idx = 0

    for img, class_predictions in zip(input_anno['images'], outputs):
        for class_id, bbbox_scores in enumerate(class_predictions):
            for bbxsc in bbbox_scores:
                width = bbxsc[2] - bbxsc[0]
                height = bbxsc[3] - bbxsc[1]
                annos.append({
                    "id": anno_idx,
                    "image_id": img['id'],
                    "category_id": class_id,
                    "bbox": [bbxsc[0], bbxsc[1], width, height],
                    "area": width * height,
                    "iscrowd": 0,
                    "score": bbxsc[4]
                })
                anno_idx += 1

    return {'info': input_anno.get('info', {}),
            'images': input_anno['images'],
            'annotations': annos,
            'licenses': input_anno.get('licenses', []),
            'categories': category_list
            }


class JBMMDCocoDataset(CocoDataset):
    # The original version is available here:
    # https://github.com/open-mmlab/mmdetection/blob/abe9d127be0d1a5b41c6098967243ec35da2c1d2/mmdet/datasets/coco.py
    # At the time it was #L358; latest commit 38c0501

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None):

        """Evaluation in COCO protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox' or metric=='segm'``.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
        if iou_thrs is None:
            iou_thrs = np.linspace(
                .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        if metric_items is not None:
            if not isinstance(metric_items, list):
                metric_items = [metric_items]

        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        eval_results = OrderedDict()
        cocoGt = self.coco
        for metric in metrics:
            logger.info(f"Evaluating {metric}")

            if metric == 'proposal_fast':
                ar = self.fast_eval_recall(
                    results, proposal_nums, iou_thrs, logger='silent')
                log_msg = []
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
                    log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
                for msg in log_msg:
                    logger.info(f"{msg}")
                continue

            iou_type = 'bbox' if metric == 'proposal' else metric
            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            try:
                predictions = mmcv.load(result_files[metric])
                if iou_type == 'segm':
                    # Refer to https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py#L331  # noqa
                    # When evaluating mask AP, if the results contain bbox,
                    # cocoapi will use the box area instead of the mask area
                    # for calculating the instance area. Though the overall AP
                    # is not affected, this leads to different
                    # small/medium/large mask AP results.
                    for x in predictions:
                        x.pop('bbox')
                    warnings.simplefilter('once')
                    warnings.warn(
                        'The key "bbox" is deleted for more accurate mask AP '
                        'of small/medium/large instances since v2.12.0. This '
                        'does not change the overall mAP calculation.',
                        UserWarning)
                cocoDt = cocoGt.loadRes(predictions)
            except IndexError:
                logger.error("The testing results of the whole dataset is empty.")
                break

            cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
            cocoEval.params.catIds = self.cat_ids
            cocoEval.params.imgIds = self.img_ids
            cocoEval.params.maxDets = list(proposal_nums)
            cocoEval.params.iouThrs = iou_thrs
            # mapping of cocoEval.stats
            coco_metric_names = {
                'mAP': 0,
                'mAP_50': 1,
                'mAP_75': 2,
                'mAP_s': 3,
                'mAP_m': 4,
                'mAP_l': 5,
                'AR@100': 6,
                'AR@300': 7,
                'AR@1000': 8,
                'AR_s@1000': 9,
                'AR_m@1000': 10,
                'AR_l@1000': 11
            }
            if metric_items is not None:
                for metric_item in metric_items:
                    if metric_item not in coco_metric_names:
                        raise KeyError(
                            f'metric item {metric_item} is not supported')

            if metric == 'proposal':
                cocoEval.params.useCats = 0
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                if metric_items is None:
                    metric_items = [
                        'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000',
                        'AR_m@1000', 'AR_l@1000'
                    ]

                for item in metric_items:
                    val = float(
                        f'{cocoEval.stats[coco_metric_names[item]]:.3f}')
                    eval_results[item] = val
            else:
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                results_per_category = None
                if classwise:  # Compute per-category AP
                    # Compute per-category AP
                    # from https://github.com/facebookresearch/detectron2/
                    precisions = cocoEval.eval['precision']
                    # precision: (iou, recall, cls, area range, max dets)
                    assert len(self.cat_ids) == precisions.shape[2]

                    results_per_category = []
                    for idx, catId in enumerate(self.cat_ids):
                        # area range index 0: all area ranges
                        # max dets index -1: typically 100 per image
                        nm = self.coco.loadCats(catId)[0]
                        precision = precisions[:, :, idx, 0, -1]
                        precision = precision[precision > -1]
                        if precision.size:
                            ap = np.mean(precision)
                        else:
                            ap = float('nan')
                        # >>>>> SEI ADDITION -- START
                        # Multiplied the metric by 100
                        results_per_category.append(
                            (f'{nm["name"]}', f'{float(ap) * 100}'))
                        # >>>>> SEI ADDITION: -- END

                    num_columns = min(6, len(results_per_category) * 2)
                    results_flatten = list(
                        itertools.chain(*results_per_category))
                    headers = ['category', 'AP'] * (num_columns // 2)
                    results_2d = itertools.zip_longest(*[
                        results_flatten[i::num_columns]
                        for i in range(num_columns)
                    ])
                    table_data = [headers]
                    table_data += [result for result in results_2d]

                if metric_items is None:
                    metric_items = [
                        'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
                    ]

                for metric_item in metric_items:
                    key = f'{metric}_{metric_item}'
                    val = float(
                        f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}'
                    )
                    # >>>>> SEI ADDITION -- START
                    # Multiplied the metric by 100
                    eval_results[key] = val * 100
                    # >>>>> SEI ADDITION: -- END
                ap = cocoEval.stats[:6]

                # >>>>> SEI ADDITION -- START
                # Added this piece to include the per-class AP results
                # in the return value.
                if results_per_category is not None:
                    for category, result in results_per_category:
                        eval_results[f"mAP_{category}"] = result
                # >>>>> SEI ADDITION -- END

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results
