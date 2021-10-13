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

import json
import math
import types
import logging
from pathlib import Path
from typing import Optional
from collections import OrderedDict

import torch
from torch.nn.parallel import DistributedDataParallel

from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.engine import launch
from detectron2.engine.defaults import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, print_csv_format
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.collect_env import collect_env_info
import detectron2.utils.comm as comm
from detectron2.utils.env import seed_all_rng
from detectron2.utils.events import EventStorage, CommonMetricPrinter, JSONWriter, TensorboardXWriter

from juneberry.config.dataset import DatasetConfig
from juneberry.config.model import ModelConfig
from juneberry.config.training_output import TrainingOutputBuilder
import juneberry.data as jb_data
import juneberry.detectron2.data as dt2_data
from juneberry.detectron2.dt2_loss_evaluator import DT2LossEvaluator
from juneberry.filesystem import generate_file_hash, ModelManager
from juneberry.jb_logging import log_banner, setup_logger
from juneberry.lab import Lab
from juneberry.plotting import plot_training_summary_chart
import juneberry.pytorch.processing as processing
from juneberry.trainer import Trainer

logger = logging.getLogger(__name__)


class Detectron2Trainer(Trainer):
    def __init__(self, lab: Lab, model_manager: ModelManager, model_config: ModelConfig, dataset_config: DatasetConfig,
                 log_level):
        super().__init__(lab, model_manager, model_config, dataset_config, log_level)

        self.iter: int = 0
        self.start_iter: int = 0
        self.max_iter: int = 0
        self.iter_per_epoch: int = 0

        self.cfg = None
        self.model = None
        self.save_model = None

        self.resume = False

        # We shouldn't need to change these.
        self.train_len = 0
        self.val_len = 0

        # We need our own data mapper for transforms.
        self.train_dataset_mapper = None
        self.test_dataset_mapper = None
        self.val_dataset_mapper = None

        # For loss evaluation
        self.loss_evaluator = None
        self.storage: EventStorage

        # Content that will be saved to output.json.
        self.output_builder = TrainingOutputBuilder()
        self.output = self.output_builder.output

        # Fill out some of the output fields using the model name / model config.
        self.output_builder.set_from_model_config(self.model_manager.model_name, self.model_config)

        self.results_keys = ['accuracy', 'false_negative', 'fg_cls_accuracy', 'loss_box_reg', 'loss_cls',
                             'loss_rpn_cls', 'loss_rpn_loc', 'learning_rate', 'num_bg_samples', 'num_fg_samples',
                             'num_neg_anchors', 'num_pos_anchors', 'timetest', 'loss', 'val_accuracy', 'val_loss']

        for key in self.results_keys:
            self.output.results.update({key: []})

    def dry_run(self) -> None:
        self.node_setup()
        self.setup()

    # ==========================

    def node_setup(self) -> None:
        log_banner(logger, "Node Setup")

        # Make sure we have the appropriate directory structure
        self.model_manager.setup()

        # NOTE: This isn't going to use proper logging
        train_list, val_list = jb_data.make_split_metadata_manifest_files(
            self.lab, self.dataset_config, self.model_config, self.model_manager)
        self.train_len = len(train_list)
        self.val_len = len(val_list)

    def establish_loggers(self) -> None:
        # Determine the rank of the current process.
        dist_rank = comm.get_local_rank()

        # In distributed training, the root Juneberry logger must be established again for each process. In
        # non-distributed training, the Trainer can continue using the root Juneberry logger that was
        # established earlier.
        if self.distributed:
            setup_logger(self.model_manager.get_training_log(), "", dist_rank=dist_rank, level=self.log_level)

        # For both distributed and non-distributed training, capture the fvcore and detectron2 logging messages
        # in the Juneberry training log files.
        setup_logger(self.model_manager.get_training_log(), "", dist_rank=dist_rank, name="fvcore",
                     level=logging.DEBUG)
        setup_logger(self.model_manager.get_training_log(), "", dist_rank=dist_rank, name="detectron2",
                     level=logging.DEBUG)

    def setup(self) -> None:

        logger.info("Setting up trainer")
        args = types.SimpleNamespace()

        # Register the datasets from our previously created manifests.
        dt2_data.register_train_manifest_files(self.lab, self.model_manager)

        # =======

        logger.warning("We do NOT support model transforms!!!")

        # Get a basic config
        cfg = get_cfg()

        # Set some basic properties
        cfg.SEED = self.model_config.seed

        # plain_train_net has loading of configs from a --config file and also from command line opts
        # We would do that by tunneling things via self.
        # cfg.merge_from_file(args.config_file)
        # cfg.merge_from_list(args.opts)

        model_arch_name = self.model_config.model_architecture['module']
        logger.info(f"Initializing detectron 2 model from: {model_arch_name}")

        cfg.merge_from_file(model_zoo.get_config_file(model_arch_name))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_arch_name)  # Let training initialize from model zoo
        # We can do a local model file if we need.
        # cfg.MODEL.WEIGHTS = "model_final_280758.pkl"

        # train root outputs, then move everything later
        #cfg.OUTPUT_DIR = str(self.model_manager.get_train_root_dir())
        cfg.OUTPUT_DIR = str(self.model_manager.get_train_scratch_path())

        # Set our datasets to the ones we registered
        cfg.DATASETS.TRAIN = [dt2_data.TRAIN_DS_NAME]
        cfg.DATASETS.TEST = [dt2_data.VAL_DS_NAME]

        # The num classes should be the same
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.dataset_config.num_model_classes

        if self.num_gpus == 0:
            cfg.MODEL.DEVICE = "cpu"

        # AOM: I have no idea of the impact of this
        if hasattr(self.model_config, "hints"):
            cfg.DATALOADER.NUM_WORKERS = self.model_config.hints.get('num_workers', 4)

        # =====================================================================
        # In JB the user specifies the number of epochs. An epoch is ONE complete pass through the data
        # in a set of batches. At the end we perform a validation at the end of each epoch.
        # A batch is how many images that will be processed _in aggregate_ before doing a model update.
        # Many of the solver properties are scaled (up or down) when we change GPUS.

        # Juneberry does things in terms of epochs not iterations.  Iterations are batch so use that.
        cfg.SOLVER.IMS_PER_BATCH = self.model_config.batch_size

        # In JB the user wants to get through the all the images some number of times.
        self.iter_per_epoch = math.ceil(self.train_len / cfg.SOLVER.IMS_PER_BATCH)
        cfg.SOLVER.MAX_ITER = self.model_config.epochs * self.iter_per_epoch

        # We want our warmup to be in terms of epochs, default of 1
        cfg.SOLVER.WARMUP_ITERS = self.iter_per_epoch * 1

        # Set it to some reasonable range
        # Detectron2 COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl uses
        #  MAX_ITER: 270000;   STEPS: [ 210000, 250000 ] ->  [ 0.7777778,  0.9259259 ] of iterations 
        cfg.SOLVER.STEPS = [math.ceil(cfg.SOLVER.MAX_ITER * 3 / 4.0), math.ceil(cfg.SOLVER.MAX_ITER * 9 / 10.0)]

        # Set a default learning rate for this reference world size
        # TODO: We should have a general purpose model config for this
        # Default for 1 GPU; according to https://github.com/facebookresearch/detectron2/blob/master/GETTING_STARTED.md
        cfg.SOLVER.BASE_LR = .0025

        # We want to evaluate after every epoch.
        # TODO: Add "validate every N epochs" type config
        cfg.TEST.EVAL_PERIOD = self.iter_per_epoch
        # We want to checkpoint every time we evaluate
        cfg.SOLVER.CHECKPOINT_PERIOD = cfg.TEST.EVAL_PERIOD

        # ===============================================
        # Okay, we need to scale everything based on GPUs.  We use detectron2 to scale everything.
        # TODO: Add Reference World Size concept
        # For now, hard code it to 1, given the learning rate of 0.0025
        cfg.REFERENCE_WORLD_SIZE = 1

        # IMS_PER_BATCH: 16
        # BASE_LR: 0.1
        # REFERENCE_WORLD_SIZE: 8
        # MAX_ITER: 5000
        # STEPS: (4000,)
        # CHECKPOINT_PERIOD: 1000

        # =================================================================

        # Now, let the user override anything they want BEFORE we scale. We want to scale what they add too.
        # They can, of course stop the scaling by setting REFERENCE_WORLD_SIZE to zero.
        if hasattr(self.model_config, "detectron2"):
            if "overrides" in self.model_config.detectron2:
                cfg.merge_from_list(self.model_config.detectron2['overrides'])

        # =================================================================
        # Now scale it it to our number of gpus.  This call scales all these based on the new
        # set of workers IF current REFERENCE_WORLD_SIZE is NOT zero.
        # NOTE: If we are cpu we don't change anything.  We get what they provided.
        # TODO: Should this be pluggable?
        if cfg.REFERENCE_WORLD_SIZE != 0 and self.num_gpus != 0:
            logger.info(f"Scaling config. REFERENCE_WORLD_SIZE original={cfg.REFERENCE_WORLD_SIZE}, "
                        f"new={self.num_gpus}")
            cfg = DefaultTrainer.auto_scale_workers(cfg, self.num_gpus)

        # ==========

        # At minimum, the iterations must be GREATER than the last step
        if cfg.SOLVER.MAX_ITER <= cfg.SOLVER.STEPS[1] + 1:
            logger.info("Adjusting MAX_ITER to be greater than the last solver step.")
            cfg.SOLVER.MAX_ITER = cfg.SOLVER.STEPS[1] + 2

        # ==========

        # This finalizes the freezes, finalizes config and sets other arguments
        # TODO: What all can we pass in as args to default?
        self.cfg = cfg
        cfg.freeze()
        default_setup(self.model_manager, cfg, args)

        # =======
        # Set up the model
        self.model = build_model(cfg)

        # If we are using DDP we don't want to serialize the fields added by DDP, so save
        # a reference to a raw model
        self.save_model = self.model
        # If distributed wrap in DDP
        if self.num_gpus > 1:
            self.model = DistributedDataParallel(
                self.model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )

        # Set our own dataset mappers to deal with transforms
        self.train_dataset_mapper = dt2_data.create_mapper(self.cfg, self.model_config.training_transforms, True)
        self.test_dataset_mapper = dt2_data.create_mapper(self.cfg, self.model_config.evaluation_transforms, False)
        # N.B. To calculate the validation loss, we need a training mapper, the test mappers remove the annotations 
        #      required for calculating the FPN loss terms:
        #      https://detectron2.readthedocs.io/en/latest/_modules/detectron2/data/dataset_mapper.html#DatasetMapper.__init__
        self.val_dataset_mapper = dt2_data.create_mapper(self.cfg, self.model_config.evaluation_transforms, True)

        # Construct the loss evaluator
        self.setup_loss_evaluator()

    def train(self) -> None:
        cfg = self.cfg
        model = self.model
        resume = self.resume

        # Establish the interval for logging the training metrics to console.
        interval = self.model_config.detectron2.metric_interval
        interval_str = "single iteration" if interval == 1 else f"{interval} iterations"
        logging.info(f"Common training metrics will be logged after every {interval_str}.")

        # NOTE: This is copied wholesale from plain_train_net.py
        # We can do our own optimizer building, etc.
        model.train()
        optimizer = build_optimizer(cfg, model)
        scheduler = build_lr_scheduler(cfg, optimizer)

        checkpointer = DetectionCheckpointer(model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler)
        # TODO: Why are these member vars instead of locals?
        self.start_iter = (checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1)
        self.max_iter = cfg.SOLVER.MAX_ITER

        periodic_checkpointer = PeriodicCheckpointer(checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=self.max_iter)
        writers = self.default_writers(cfg.OUTPUT_DIR, self.max_iter) if comm.is_main_process() else []

        # compared to "train_net.py", we do not support accurate timing and
        # precise BN here, because they are not trivial to implement in a small training loop
        data_loader = build_detection_train_loader(cfg, mapper=self.train_dataset_mapper)
        logger.info("Starting training from iteration {}".format(self.start_iter))

        self.storage = EventStorage(self.start_iter)
        with self.storage as storage:
            for data, iteration in zip(data_loader, range(self.start_iter, self.max_iter)):
                write_json = False
                self.iter = iteration
                storage.iter = iteration

                loss_dict = model(data)
                losses = sum(loss_dict.values())
                assert torch.isfinite(losses).all(), loss_dict

                loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
                scheduler.step()

                # Compute the loss for this step
                self.loss_evaluator.after_step(self)

                # and iteration != max_iter - 1
                if cfg.TEST.EVAL_PERIOD > 0 and ((iteration + 1) % cfg.TEST.EVAL_PERIOD) == 0:
                    self.do_test()
                    # Compared to "train_net.py", the test results are not dumped to EventStorage
                    comm.synchronize()
                    write_json = True

                if comm.is_main_process():
                    storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

                for writer in writers:

                    # The metrics are written to console every certain number of iterations.
                    if isinstance(writer, CommonMetricPrinter):
                        if (iteration + 1) % interval == 0:
                            writer.write()

                    # If the iteration if for the end of an epoch, write to the JSON file.
                    if isinstance(writer, JSONWriter):
                        if write_json:
                            writer.write()

                    # Write to tensorboard after every iteration.
                    if isinstance(writer, TensorboardXWriter):
                        writer.write()

                periodic_checkpointer.step(iteration)

    def do_test(self):
        cfg = self.cfg
        model = self.model

        results = OrderedDict()
        for dataset_name in cfg.DATASETS.TEST:
            data_loader = build_detection_test_loader(cfg, dataset_name, mapper=self.test_dataset_mapper)
            evaluator = COCOEvaluator(dataset_name, output_dir=cfg.OUTPUT_DIR)
            # evaluator = get_evaluator(
            #     cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            # )
            # NB inference_on_dataset puts the model in eval mode, turns off gradients, and returns the model to the
            # state it entered it
            results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)
        if len(results) == 1:
            results = list(results.values())[0]
        return results

    def finish(self) -> None:

        # The rank 0 process is responsible for renaming the detectron2 output model file to
        # the typical Juneberry output model file.

        if comm.get_local_rank():
            logger.info(f"Only the rank 0 process is responsible for saving the model file.")

        else:
            # NOTE: This is a custom path for DT2.  We know that DT2 puts the model file in the final
            # OUTPUT directory, so we rename it out of there.
            final_model_path = Path(self.cfg.OUTPUT_DIR) / 'model_final.pth'
            logger.info(f"Renaming {final_model_path} to "
                        f"{self.model_manager.get_pytorch_model_path()}")
            final_model_path.rename(self.model_manager.get_pytorch_model_path())

            # Retrieve the metrics from the dt2 metrics log.
            self.extract_dt2_log_content(self.get_dt2_metrics_log())

            plot_training_summary_chart(self.output, self.model_manager)

            # Compute the model hash.
            self.output.results.model_hash = generate_file_hash(self.model_manager.get_pytorch_model_path())

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
        self.distributed = True
        self.num_gpus = num_gpus

        # We don't support multi-machine yet
        launch(
            self.train_model,
            num_gpus,
            num_machines=1,
            machine_rank=0,
            dist_url="auto",  # See defaults for what this is
            args=(),
        )

    # ===============================

    #  ___       _                        _
    # |_ _|_ __ | |_ ___ _ __ _ __   __ _| |
    #  | || '_ \| __/ _ \ '__| '_ \ / _` | |
    #  | || | | | ||  __/ |  | | | | (_| | |
    # |___|_| |_|\__\___|_|  |_| |_|\__,_|_|

    def setup_loss_evaluator(self):
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        val_loader = build_detection_test_loader(self.cfg, self.cfg.DATASETS.TEST[0], self.val_dataset_mapper)

        self.loss_evaluator = DT2LossEvaluator(cfg.TEST.EVAL_PERIOD, self.model, val_loader)

    def default_writers(self, output_dir: str, max_iter: Optional[int] = None):

        tb_dir = self.model_manager.create_tensorboard_directory_name(
            self.lab.tensorboard) if self.lab.tensorboard is not None else output_dir

        logger.info(f"Tensorboard events for this run will be stored in {output_dir}")

        return [
            CommonMetricPrinter(max_iter),
            JSONWriter(Path(output_dir, "metrics.json")),
            TensorboardXWriter(tb_dir),
        ]

    def get_dt2_metrics_log(self):
        """:return: The path to the dt2 metrics log."""
        return self.model_manager.get_train_scratch_path() / "metrics.json"

    def append_metric(self, content, field, metric):
        """
        Appends the metric (if available) to the results field.
        """
        if metric in content:
            field.append(content[metric])

    def extract_dt2_log_content(self, file: Path):
        logger.info(f"Extracting training data from dt2 metrics log.")

        # Open the dt2 metrics log and read each line. Currently only fast_rcnn accuracies are supported.
        with open(file, 'r') as log_file:
            for line in log_file:
                content = json.loads(line)
                self.append_metric(content, self.output.results.accuracy, 'fast_rcnn/cls_accuracy')
                self.append_metric(content, self.output.results.false_negative, 'fast_rcnn/false_negative')
                self.append_metric(content, self.output.results.fg_cls_accuracy, 'fast_rcnn/fg_cls_accuracy')
                self.append_metric(content, self.output.results.loss_box_reg, 'loss_box_reg')
                self.append_metric(content, self.output.results.loss_cls, 'loss_cls')
                self.append_metric(content, self.output.results.loss_rpn_cls, 'loss_rpn_cls')
                self.append_metric(content, self.output.results.loss_rpn_loc, 'loss_rpn_loc')
                self.append_metric(content, self.output.results.learning_rate, 'lr'),
                self.append_metric(content, self.output.results.num_bg_samples, 'roi_head/num_bg_samples')
                self.append_metric(content, self.output.results.num_fg_samples, 'roi_head/num_fg_samples')
                self.append_metric(content, self.output.results.num_neg_anchors, 'rpn/num_neg_anchors')
                self.append_metric(content, self.output.results.num_pos_anchors, 'rpn/num_pos_anchors')
                self.append_metric(content, self.output.results.timetest, 'timetest')
                self.append_metric(content, self.output.results.loss, 'total_loss')
                self.append_metric(content, self.output.results.val_accuracy, 'val_accuracy')
                self.append_metric(content, self.output.results.val_loss, 'validation_loss')

        file.unlink()


def default_setup(model_manager: ModelManager, cfg, args):
    """
    Slight modification to dt2 default_setup. This implementation will not
    set up the logging, since that has already been done by Juneberry.

    Perform some basic common setups at the beginning of a job, including:

    1. Log basic information about environment, cmdline arguments, and config
    2. Backup the config to the output directory

    Args:
        cfg (CfgNode): the full config to be used
        args (argparse.NameSpace): the command line arguments to be logged
    """
    output_dir = cfg.OUTPUT_DIR
    if comm.is_main_process() and output_dir:
        output_path = Path(output_dir)
        if not output_path.exists():
            logger.info(f"Output directory {output_path} was not found, so it was created.")
            output_path.mkdir()

    rank = comm.get_rank()

    # Temporarily use the detectron2 logger, to make it seem like these logging messages are still
    # coming from detectron2.
    dt2_logger = logging.getLogger("detectron2.setup_logger.in_dt2_trainer")

    dt2_logger.info("Rank of current process: {}. World size: {}".format(rank, comm.get_world_size()))
    dt2_logger.info("Environment info:\n" + collect_env_info())

    dt2_logger.info("Command line arguments: " + str(args))
    if hasattr(args, "config_file") and args.config_file != "":
        with open(Path(args.config_file), "r") as f:
            cfg_content = f.read()
        dt2_logger.info("Contents of args.config_file={}:\n{}".format(args.config_file, cfg_content))

    dt2_logger.info("Running with full config:\n{}".format(cfg))
    if comm.is_main_process() and output_dir:
        # Note: some of our scripts may expect the existence of
        # config.yaml in output directory
        # path = Path(output_dir, "config.yaml")
        path = Path(model_manager.get_platform_training_config())
        with open(path, "w") as f:
            f.write(cfg.dump())
        dt2_logger.info("Full config saved to {}".format(path))

    # make sure each worker has a different, yet deterministic seed if specified
    seed_all_rng(None if cfg.SEED < 0 else cfg.SEED + rank)

    # cudnn benchmark has large overhead. It shouldn't be used considering the small size of
    # typical validation set.
    if not (hasattr(args, "eval_only") and args.eval_only):
        torch.backends.cudnn.benchmark = cfg.CUDNN_BENCHMARK