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

from pathlib import Path
from unittest import TestCase

import pytest

from juneberry.config.dataset import DatasetConfig
from juneberry.config.model import ModelConfig
from juneberry.detectron2.trainer import Detectron2Trainer
from juneberry.filesystem import ModelManager
from juneberry.lab import Lab
from juneberry.training.trainer_factory import TrainerFactory
import utils


class TestTrainerFactory(TestCase):
    """
    This group of tests exercise various aspects of the TrainerFactory.
    """

    @pytest.fixture(autouse=True)
    def init_fixtures(self, tmp_path, caplog):
        """
        The purpose of this method is to make certain fixtures available inside of
        the unittest.TestCase.
        """
        self.tmp_path = tmp_path
        self.caplog = caplog

        # Initialize a TrainerFactory.
        self.trainer_factory = TrainerFactory()

    def test_trainer_factory_attribute_init(self):
        """
        The purpose of this test is to verify that the TrainerFactory attributes are initialized properly.
        """
        # Confirm that all TrainerFactory attributes are set to None.
        assert self.trainer_factory.dataset_config is None
        assert self.trainer_factory.lab is None
        assert self.trainer_factory.log_level is None
        assert self.trainer_factory.model_config is None
        assert self.trainer_factory.model_manager is None

    def test_trainer_factory_set_model_config(self):
        """
        The purpose of this test is to exercise the 'set_model_config' method of the TrainerFactory under
        various conditions.
        :return:
        """
        # Verify that the TrainerFactory's model_config attribute isn't set.
        assert self.trainer_factory.model_config is None

        # Attempt to set the model_config attribute. It should fail because currently
        # there is no ModelManager associated with the TrainerFactory.
        self.trainer_factory.set_model_config()
        assert "no ModelManager associated with the TrainerFactory" in self.caplog.text

        # Create a temporary workspace in order to exercise the TrainerFactory's ability to load a
        # ModelConfig using the ModelManager.
        with utils.set_directory(self.tmp_path):

            # Create a ModelManager and associate it with the TrainerFactory.
            self.trainer_factory.model_manager = ModelManager('text_detect/dt2/ut')

            # Create a ModelConfig.
            model_config = utils.text_detect_dt2_config
            mc = ModelConfig.construct(data=model_config)

            # Save the model config file to the location that the ModelManager expects
            mc_path = self.trainer_factory.model_manager.get_model_config()
            mc_path.parent.mkdir(parents=True)
            mc.save(data_path=self.trainer_factory.model_manager.get_model_config())

            # Now attempt to set the model_config attribute again via the TrainerFactory. Since
            # the model config file exists, it should work.
            self.trainer_factory.set_model_config()

            # Verify that the TrainerFactory's model_config attribute has been set and that the
            # model_config's attributes match expectations.
            assert self.trainer_factory.model_config is not None
            assert self.trainer_factory.model_config.model_architecture.fqcn == "COCO-Detection/faster_rcnn_R_50_" \
                                                                                  "FPN_1x.yaml"
            assert self.trainer_factory.model_config.epochs == 1
            assert self.trainer_factory.model_config.validation.arguments.seed == 3554237221

            # Create a variation of the previous ModelConfig and make slight adjustments to some of
            # its attributes.
            model_config = ModelConfig.load(self.trainer_factory.model_manager.get_model_config())
            model_config.model_architecture.fqcn = "new_value"
            model_config.epochs = 10
            model_config.validation.arguments.seed = 1234

            # Attempt to set the TrainerFactory's model_config attribute to the ModelConfig variation
            # that was just created.
            self.trainer_factory.set_model_config(model_config=model_config)

            # Confirm the model_config associated with the TrainerFactory reflects the previously
            # adjusted values.
            assert self.trainer_factory.model_config is not None
            assert self.trainer_factory.model_config.model_architecture.fqcn == "new_value"
            assert self.trainer_factory.model_config.epochs == 10
            assert self.trainer_factory.model_config.validation.arguments.seed == 1234

    def test_trainer_factory_get_trainer(self):
        """
        The purpose of this test is to exercise the TrainerFactory's ability to produce Trainer objects.
        """
        # Verify that the TrainerFactory is unable to produce a Trainer when there is no model
        # config associated with the TrainerFactory
        assert self.trainer_factory.model_config is None
        trainer = self.trainer_factory.get_trainer()
        assert trainer is None

        # Create a temporary workspace so that the file loading operations will work.
        with utils.set_directory(self.tmp_path):

            # Now set the TrainerFactory attributes which are used to produce Trainers.
            self.trainer_factory.lab = Lab(workspace=self.tmp_path, data_root=self.tmp_path)
            self.trainer_factory.model_manager = ModelManager('text_detect/dt2/ut')
            self.trainer_factory.model_config = ModelConfig.construct(data=utils.text_detect_dt2_config)
            self.trainer_factory.dataset_config = DatasetConfig.construct(data=utils.text_detect_dataset_config)

            # In order to assemble the Trainer, the dataset file will need to be read from the expected location.
            ds_path = self.trainer_factory.model_config.training_dataset_config_path
            Path(ds_path).parent.mkdir(parents=True)
            self.trainer_factory.dataset_config.save(data_path=ds_path)

            # Get a Trainer from the TrainerFactory and verify some of its properties.
            trainer = self.trainer_factory.get_trainer()
            assert type(trainer) == Detectron2Trainer
            assert trainer.resume is False
            assert trainer.native is True
            assert trainer.onnx is False

            # Get another Trainer from the TrainerFactory, only this time set the parameters of the
            # 'get_trainer' method to the opposite of their default values. Verify the type of the
            # resulting Trainer, and confirm the three Trainer attributes were set correctly.
            trainer = self.trainer_factory.get_trainer(resume=True, native=False, onnx=True)
            assert type(trainer) == Detectron2Trainer
            assert trainer.resume is True
            assert trainer.native is False
            assert trainer.onnx is True
