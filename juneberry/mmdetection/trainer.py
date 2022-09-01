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
from math import ceil
from pathlib import Path
import sys

import hjson
import torch

from mmcv import Config
from mmcv.utils.logging import logger_initialized

from mmdet.apis import train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector

from juneberry.config.dataset import DatasetConfig
from juneberry.config.model import ModelConfig
from juneberry.config.training_output import TrainingOutputBuilder
import juneberry.data as jb_data
from juneberry.filesystem import generate_file_hash, ModelManager
from juneberry.jb_logging import log_banner, setup_logger
from juneberry.lab import Lab
import juneberry.mmdetection.utils as mmd_utils
from juneberry.mmdetection.utils import MMDPlatformDefinitions
from juneberry.plotting import plot_training_summary_chart
import juneberry.pytorch.processing as processing
from juneberry.trainer import Trainer

logger = logging.getLogger(__name__)


class MMDTrainer(Trainer):
    def __init__(self, lab: Lab, model_manager: ModelManager, model_config: ModelConfig, dataset_config: DatasetConfig,
                 log_level):
        super().__init__(lab, model_manager, model_config, dataset_config, log_level)

        self.working_dir = model_manager.get_train_scratch_path()
        self.mm_home = mmd_utils.find_mmdetection()

        self.cfg = None
        self.datasets = None
        self.model = None

        # This is the pytorch device we are associated with.
        self.device = None
        self.dryrun = False

        logger.info(f"Using working directory of: {self.working_dir}")

        # Content that will be saved to output.json.
        self.output_builder = TrainingOutputBuilder()
        self.output = self.output_builder.output

        # Fill out some of the output fields using the model name / model config.
        self.output_builder.set_from_model_config(self.model_manager.model_name, self.model_config)

        # Initialize the output keys that will be written to as empty lists.
        # TODO: Disabled until we figure out how to get the per epoch time instead of the per iteration time.
        # self.output.times.epoch_duration_sec = []

        results_keys = ['accuracy', 'loss', 'loss_bbox', 'loss_cls', 'loss_rpn_bbox', 'loss_rpn_cls',
                        'learning_rate', 'val_loss', 'val_accuracy']

        # TODO: Use these keys instead when 'val_loss' capability is restored.
        # results_keys = ['accuracy', 'loss', 'loss_bbox', 'loss_cls', 'loss_rpn_bbox', 'loss_rpn_cls',
        #                 'learning_rate', 'val_accuracy', 'val_loss', 'val_loss_bbox', 'val_loss_cls',
        #                 'val_loss_rpn_bbox', 'val_loss_rpn_cls']
        for key in results_keys:
            self.output.results.update({key: []})

    # ==========================================================================

    @classmethod
    def get_platform_defs(cls):
        return MMDPlatformDefinitions()

    # ==========================================================================

    @classmethod
    def get_training_output_files(cls, model_mgr: ModelManager, dryrun: bool = False):
        """
        Returns a list of files to clean from the training directory. This list should contain ONLY
        files or directories that were produced by the training command. Directories in this list
        will be deleted even if they are not empty.
        :param model_mgr: A ModelManager to help locate files.
        :param dryrun: When True, returns a list of files created during a dryrun of the Trainer.
        :return: The files to clean from the training directory.
        """
        common_files = [model_mgr.get_training_data_manifest_path(),
                        model_mgr.get_validation_data_manifest_path(),
                        model_mgr.get_platform_training_config(cls.get_platform_defs()),
                        model_mgr.get_train_scratch_path()]

        if dryrun:
            return common_files
        else:
            return common_files + [model_mgr.get_model_path(cls.get_platform_defs()),
                                   model_mgr.get_training_out_file(),
                                   model_mgr.get_training_summary_plot()]

    @classmethod
    def get_training_clean_extras(cls, model_mgr: ModelManager, dryrun: bool = False):
        """
        Returns a list of extra "training" files/directories to clean. Directories in this list will NOT
        be deleted if they are not empty.
        :param model_mgr: A ModelManager to help locate files.
        :param dryrun: When True, returns a list of files created during a dryrun of the Trainer.
        :return: The extra files to clean from the training directory.
        """
        if dryrun:
            return [model_mgr.get_train_root_dir()]
        else:
            return [model_mgr.get_train_root_dir()]

    # ==========================================================================

    def dry_run(self) -> None:
        # Setup saves the config file which is really what we want for now.
        self.dryrun = True
        self.node_setup()
        self.setup()

    # ==========================

    def node_setup(self) -> None:
        log_banner(logger, "Node Setup")

        if self.num_gpus == 0 and not self.dryrun:
            logger.error("MMDetection does not support CPU mode for training. EXITING.")
            sys.exit(-1)

        # Set up working dir to save files and logs.
        if not self.working_dir.exists():
            logger.info(f"Creating working dir {str(self.working_dir)}")
            self.working_dir.mkdir(parents=True)

        # ============ Setup Datasets

        jb_data.make_split_metadata_manifest_files(self.lab, self.dataset_config, self.model_config, self.model_manager)

    def establish_loggers(self) -> None:
        # In distributed training, the root Juneberry logger must be established again for each process. In
        # non-distributed training, the Trainer can continue using the root Juneberry logger that was
        # established earlier.
        if self.distributed:
            setup_logger(self.model_manager.get_training_log(), "", dist_rank=self.gpu, level=self.log_level)

        # This will output any mmdet logging messages to console and to file. The logging
        # level is DEBUG here because in a non-verbose setting, this helps
        # distinguish between messages coming from Juneberry and mmdet.
        setup_logger(self.model_manager.get_training_log(), "", dist_rank=self.gpu, name="mmdet",
                     level=logging.DEBUG)

        # This will prevent train_detector from setting up the mmdet logger again.
        logger_initialized["mmdet"] = True

    def setup(self) -> None:

        if self.onnx:
            logger.warning(f"An ONNX model format was requested, but mmdetection training does not support saving "
                           f"model files in ONNX format. Switching to the MMD native format.")
            self.onnx = False
            self.native = True

        # Load cfg based on what they said in the model.
        cfg = Config.fromfile(str(self.mm_home / "configs" / self.model_config.model_architecture.fqcn))
        cfg.work_dir = str(self.working_dir)

        # ============ Extract the MMD configuration

        mmd_config = {}
        if hasattr(self.model_config, "mmdetection"):
            mmd_config = self.model_config.mmdetection

        # ============ Setup Processing

        self.device = processing.setup_cuda_device(self.num_gpus, self.gpu)
        if self.distributed:
            processing.setup_distributed(self.num_gpus, self.gpu)
        processing.log_cuda_configuration(self.num_gpus, self.gpu, logger)

        # Set the samples per gpu to the batch size.
        cfg.data.samples_per_gpu = self.model_config.batch_size

        # TODO: Do we need to set deterministic?

        # ============ Pipelines

        # Add in the pipelines overrides. We have to do this BEFORE we wrap datasets so the pipelines
        # get moved around.
        # TODO: Should we adjust base pipelines and just repoint? Based on
        #       https://mmdetection.readthedocs.io/en/latest/tutorials/config.html
        #       this seems the way to go.
        mmd_utils.adjust_pipelines(self.model_config, cfg)

        # ============ Setup Datasets

        # jb_data.make_split_dataset_files(self.lab, self.dataset_config, self.model_config, self.model_manager)

        # Get the class names from the dataset AFTER IT HAS BEEN PROCESSED.
        label_names = jb_data.get_label_mapping(model_manager=self.model_manager, model_config=self.model_config,
                                                train_config=self.dataset_config)
        classes = list(label_names.values())
        logger.info(f"Using classes={classes}")

        # Their ConfigDict doesn't seem to work like other attribute dicts in that it lets adding any old term.
        # So, we need to build an pure python overlay and merge.  Let's do it "mmdetect" style for "consistency".
        train_ds = dict(type='CocoDataset',
                        data_root=str(self.lab.data_root().resolve()),
                        ann_file=str(self.model_manager.get_training_data_manifest_path().resolve()),
                        img_prefix="",
                        classes=classes)

        # Wrap if needed
        if 'trainDatasetWrapper' in mmd_config:
            wrapper_cfg = mmd_config['trainDatasetWrapper']
            logger.info(f"Wrapping training dataset with {wrapper_cfg['type']}")
            # Make sure to move the pipeline "down"
            train_ds['pipeline'] = cfg.data.train.pipeline
            train_ds = dict(type=wrapper_cfg['type'],
                            ann_file="",
                            img_prefix="",
                            dataset=train_ds)
            train_ds.update(wrapper_cfg['kwargs'])
            cfg.data.train.pipeline = []

        # Now nest it at the right "level" and merge it in
        cfg.merge_from_dict(dict(data=dict(train=train_ds)))

        val_ds = dict(type='CocoDataset',
                      data_root=str(self.lab.data_root().resolve()),
                      ann_file=str(self.model_manager.get_validation_data_manifest_path().resolve()),
                      img_prefix="",
                      classes=classes)

        # TODO: This "works" for now, but I don't think it's right. If the validation pipeline is an
        #  exact copy of the training pipeline, it's going to have the same image augmentations (like flip
        #  or resize). I'm not sure we want that. The correct thing to do might be to put some work into
        #  adjust_pipelines to make sure the val.pipeline is in the state we want. As currently written,
        #  this will still produce val_loss values, but they probably aren't "correct".
        #  NOTE - VAL_LOSS IS DISABLED FOR NOW
        # cfg.data.val.pipeline = cfg.data.train.pipeline.copy()

        # Now nest it at the right "level" and merge it in
        cfg.merge_from_dict(dict(data=dict(val=val_ds)))

        # We don't use this so remove.
        cfg.data.test = {}

        # The user can specify a set of weights to load from, such as pretrained model.
        # TODO: Use 'previous model' here?
        if hasattr(self.model_config, "mmdetection"):
            if "load_from" in self.model_config.mmdetection:
                cfg.load_from = self.model_config.mmdetection['load_from']
                logger.info(f"Adding load_from model load from: {cfg.load_from}")

        # The original learning rate (LR) is set for 8-GPU training. (reference)
        # We divide it by 8 since we only use one GPU.
        cfg.optimizer.lr = 0.02 / 8
        cfg.lr_config.warmup = None

        # Set seed thus the results are more reproducible.
        mmd_utils.add_reproducibility_configuration(self.model_config, cfg)

        # We set this for non-distributed.
        cfg.gpu_ids = range(self.num_gpus)
        logger.info(f"Setting up for gpus: {cfg.gpu_ids}")

        # These metrics work for our style of dataset.
        cfg.evaluation.metric = 'bbox'
        # We can set the evaluation interval to reduce the evaluation times.
        cfg.evaluation.interval = 1
        # We can set the checkpoint saving interval to reduce the storage cost.
        cfg.checkpoint_config.interval = 1

        # Set the number of epochs.
        cfg.runner.type = 'EpochBasedRunner'
        cfg.runner.max_epochs = self.model_config.epochs

        # Add in the overrides if they have any, which they usually do.
        mmd_utils.add_config_overrides(self.model_config, cfg)

        # Save the entire config to the working dir.  At this point we should be able to
        # use the mmdetection "train.py" script with the config file.
        config_out = self.model_manager.get_platform_training_config(MMDTrainer.get_platform_defs())
        with open(config_out, "w") as out_cfg:
            logger.info(f"Writing out final config to: {config_out}")
            out_cfg.write(cfg.pretty_text)

        # TODO: There were some unanswered questions about whether or not we were obtaining
        #  the val_loss correctly, so for now our workflow is only going to include 'train'.
        cfg.workflow = [('train', 1)]
        # cfg.workflow = [('train', 1), ('val', 1)]

        if self.dryrun:
            return

        # ==========================================================

        # Build the datasets.
        logger.info("Building the training and validation datasets.")
        train_dataset = build_dataset(cfg.data.train)

        # TODO: The val_dataset must be enabled again if we restore the 'val' portion
        #  of the workflow for val_loss.
        # val_dataset = build_dataset(cfg.data.val)
        # self.datasets = [train_dataset, val_dataset]
        self.datasets = [train_dataset]

        # Set the logging interval. The idea behind this calculation is to set the logging interval such that it
        # displays a training output log message at the end of a training epoch.
        cfg.log_config.interval = ceil(len(train_dataset) / (self.num_gpus * self.model_config.batch_size))

        # Add the number of training / validation images to the training output.
        self.output.options.num_training_images = len(train_dataset)

        # TODO: Restore this when the 'val' portion is added back to the workflow for val_loss.
        self.output.options.num_validation_images = 0
        # self.output.options.num_validation_images = len(val_dataset)

        # Build the detector.
        logger.info("Building the model (build_detector)")
        self.model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
        if cfg.load_from is None or len(cfg.load_from.strip()) == 0:
            logger.info("...initing model weights.")
            self.model.init_weights()
        self.model.CLASSES = self.datasets[0].CLASSES

        # Save our config
        self.cfg = cfg

    def train(self) -> None:
        # If not epochs, just save the model
        if self.model_config.epochs == 0:
            torch.save(self.model_config.epochs, self.model_manager.get_mmd_latest_model_path())
            return

        # For now we just support multi-gpu single-process MMD with its use of DataParallel.
        # NOTE: If we ever decide to use the timestamp kwarg for train_detector, this will impact
        # get_mmd_out_log in filesystem.py. Refer to that method for more details.
        train_detector(self.model, self.datasets, self.cfg, distributed=self.distributed, validate=True)

    def finish(self) -> None:

        if self.gpu:
            logger.info(f"Only the rank 0 process performs the training finalization steps.")

        else:
            # The first task is to rename the trained model file to the Juneberry standard.
            src = self.model_manager.get_mmd_latest_model_path()
            dst = self.model_manager.get_model_path(self.get_platform_defs())
            logger.info(f"Moving output model '{src}' -> '{dst}'")
            # The resolve turns the link into the full path.
            src.resolve().rename(dst)

            # Retrieve the metrics from the mmd training log.
            if self.model_config.epochs > 0:
                self.extract_mmd_log_content(self.get_mmd_out_log())
                plot_training_summary_chart(self.output, self.model_manager)

            # Compute the model hash.
            model_path = self.model_manager.get_model_path(self.get_platform_defs())
            self.output.results.model_hash = generate_file_hash(model_path)

            # Add the training time information to the output.
            self.output.times.start_time = self.train_start_time.isoformat()
            self.output.times.end_time = self.train_end_time.isoformat()
            duration = self.train_end_time - self.train_start_time
            self.output.times.duration = duration.total_seconds()

            # Save the training output to the JSON file.
            self.output_builder.save(self.model_manager.get_training_out_file())

    # ==========================

    def check_gpu_availability(self, required: int = None):
        return processing.determine_gpus(required)

    def train_distributed(self, num_gpus) -> None:
        # This call initializes this base object before we spawn multiple processes
        # which get copies.  For the most part, everything can come through via this
        # object except we use the environment variables for the address and port
        # as is traditional.
        self.distributed = True
        self.num_gpus = num_gpus

        processing.prepare_for_distributed()

        # Indicate the number of GPUs detected and adjust the batch size so that the batch size
        # specified in the config is evenly distributed among all processes in the "world".
        # TODO: Inject learning rate scaling code
        new_batch_size = int(self.model_config.batch_size / self.num_gpus)
        # This is really for unit tests because real code has real sized batches
        if new_batch_size < 1:
            new_batch_size = 1
        if self.model_config.batch_size != new_batch_size:
            logger.info(f"Adjusting batch size from {self.model_config.batch_size} "
                        f"to {new_batch_size} for distributed training...")
            self.model_config.batch_size = new_batch_size
            self.output.options.batch_size = new_batch_size
            logger.warning("!!! NOT ADJUSTING LEARNING RATE")

        # Spawn one training process per GPU.
        processing.start_distributed(self.train_model, self.num_gpus)

    def extract_mmd_log_content(self, file: Path):
        logger.info(f"Extracting training log data from mmdetection training output log.")

        # Open the mmd training log and read each line. Each line contains a metrics summary in
        # JSON format for each phase of training that was recorded.
        with open(file, 'r') as log_file:
            for line in log_file:
                # TODO: Should this be routed through the jbfs load chokepoint?
                content = hjson.loads(line)

                # If it's a summary of training metrics, add the values to the appropriate lists.
                if content['mode'] == "train":
                    # TODO: The time reported here is actually the per iteration time, not per epoch.
                    #  Need to figure out if it's possible to get the per epoch time without tracking
                    #  the time spent for each iteration.
                    # self.output.times.epoch_duration_sec.append(content['time'])
                    self.output.results.learning_rate.append(content['lr'])
                    self.output.results.loss_rpn_cls.append(content['loss_rpn_cls'])
                    self.output.results.loss_rpn_bbox.append(content['loss_rpn_bbox'])
                    self.output.results.loss_cls.append(content['loss_cls'])
                    self.output.results.loss_bbox.append(content['loss_bbox'])
                    self.output.results.loss.append(content['loss'])
                    self.output.results.accuracy.append(content['acc'] / 100)

                # TODO: Will need to be enabled again once val_loss returns.
                # If the content isn't training metrics, they must be validation metrics. Therefore
                # the values get added to different (validation) lists.
                # else:
                #     self.output.results.val_loss_rpn_cls.append(content['loss_rpn_cls'])
                #     self.output.results.val_loss_rpn_bbox.append(content['loss_rpn_bbox'])
                #     self.output.results.val_loss_cls.append(content['loss_cls'])
                #     self.output.results.val_loss_bbox.append(content['loss_bbox'])
                #     self.output.results.val_loss.append(content['loss'])
                #     self.output.results.val_accuracy.append(content['acc'])

        # After the relevant information has been extracted, the mmd training output log serves no
        # purpose so it can be deleted.
        file.unlink()

    # ==========================

    def get_mmd_out_log(self):
        """:return: The path to the mmdetection training output log."""
        # NOTE: The name of this log file is controlled by the 'timestamp' kwarg that gets
        # passed in to the train_detector call in our mmd_trainer. Since we currently don't
        # use this kwarg, the name of the log file defaults to "None". If we use that kwarg
        # in the future, the name of the log file here would have to be updated to match
        # whatever we pass in for that kwarg.
        return self.model_manager.get_train_scratch_path() / 'None.log.json'
