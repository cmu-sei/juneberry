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
from argparse import Namespace
import logging
from unittest import TestCase

from juneberry.script_tools.sprout import Sprout
from juneberry.script_tools.training_sprout import TrainingSprout
from juneberry.script_tools.tuning_sprout import TuningSprout


class TestBaseSprout(TestCase):
    """
    Tests meant to exercise the Base Sprout.
    """

    @staticmethod
    def build_base_namespace():
        """
        This method returns a Namespace. In typical Juneberry scripts, you normally obtain a Namespace
        by running parse_args() on the ArgumentParser to parse the command line options being passed to
        the script.
        :return: A Namespace that sets all of the args expected by the Base Sprout to non-default values.
        """
        return Namespace(workspace='workspace_dir',
                         dataRoot='dataroot_dir',
                         tensorboard='tensorboard_dir',
                         logDir='log_dir',
                         silent=True,
                         verbose=True,
                         profileName='profile_name')

    def test_sprout_defaults(self):
        """
        This test confirms that all Base Sprout attributes are initialized to None when a Base Sprout
        is initialized.
        """
        # Initialize a Base Sprout.
        sprout = Sprout()

        # Check all of the Base Sprout attributes.
        assert sprout.workspace_dir is None
        assert sprout.dataroot_dir is None
        assert sprout.tensorboard_dir is None
        assert sprout.log_dir is None
        assert sprout.silent is None
        assert sprout.log_level is None
        assert sprout.profile_name is None

    def test_sprout_from_empty_namespace(self):
        """
        This test checks the behavior of the Base Sprout when an empty Namespace is used to set the
        attributes in the Base Sprout. For the most attributes, the default values remain the same.
        """
        # Initialize a Base Sprout and feed it an empty Namespace.
        sprout = Sprout()
        sprout.grow_from_args(Namespace())

        # Check all of the Base Sprout attributes for the expected values.
        assert sprout.workspace_dir is None
        assert sprout.dataroot_dir is None
        assert sprout.tensorboard_dir is None
        assert sprout.log_dir is None
        assert sprout.silent is False
        assert sprout.log_level == logging.INFO
        assert sprout.profile_name is None

    def test_sprout_base_namespace(self):
        """
        This test confirms every attributes in the Base Sprout is set properly when the corresponding
        arg is defined in the Namespace.
        """
        # Initialize a Base Sprout and feed it a fully-defined Namespace.
        sprout = Sprout()
        sprout.grow_from_args(self.build_base_namespace())

        # Check all of the Base Sprout attributes for the expected values.
        assert sprout.workspace_dir == "workspace_dir"
        assert sprout.dataroot_dir == "dataroot_dir"
        assert sprout.tensorboard_dir == "tensorboard_dir"
        assert sprout.log_dir == "log_dir"
        assert sprout.silent is True
        assert sprout.log_level == logging.DEBUG
        assert sprout.profile_name == "profile_name"


class TestTrainingSprout(TestCase):
    """
    Tests meant to exercise the TrainingSprout.
    """

    @staticmethod
    def build_training_namespace():
        """
        This method returns a Namespace similar to the Namespace you would received when running jb_train.
        :return: A Namespace that sets all of the args expected by the TrainingSprout to non-default values.
        """
        # Create a Namespace for args that are unique to the TrainingSprout.
        train_args = Namespace(modelName="test_model",
                               num_gpus=0,
                               dryrun=True,
                               resume=True,
                               skipNative=True,
                               onnx=True)

        # Create a Namespace for the args in the Base Sprout.
        base_args = TestBaseSprout.build_base_namespace()

        # Combine the two Namespaces into a single Namespace and return the result.
        return Namespace(**vars(train_args), **vars(base_args))

    def test_training_sprout_defaults(self):
        """
        This test confirms that all TrainingSprout attributes are initialized to None when a TrainingSprout
        is initialized.
        """
        # Create the sprout.
        sprout = TrainingSprout()

        # Check the attributes that are unique to the TrainingSprout.
        assert sprout.model_name is None
        assert sprout.num_gpus is None
        assert sprout.dryrun is None
        assert sprout.resume is None
        assert sprout.skip_native is None
        assert sprout.onnx is None

        # Also check the attributes in the base Sprout.
        assert sprout.workspace_dir is None
        assert sprout.dataroot_dir is None
        assert sprout.tensorboard_dir is None
        assert sprout.log_dir is None
        assert sprout.silent is None
        assert sprout.log_level is None
        assert sprout.profile_name is None

    def test_training_sprout_from_empty_namespace(self):
        """
        This test checks the behavior of the Training Sprout when an empty Namespace is used to set the
        attributes in the Training Sprout.
        """
        # Initialize a TrainingSprout and feed it an empty Namespace.
        sprout = TrainingSprout()
        sprout.grow_from_args(Namespace())

        # Check all of the attributes that are unique to the TrainingSprout for the expected values.
        assert sprout.model_name is None
        assert sprout.num_gpus is None
        assert sprout.dryrun is False
        assert sprout.resume is False
        assert sprout.skip_native is False
        assert sprout.onnx is False

        # Also check all of the Base Sprout attributes for the expected values.
        assert sprout.workspace_dir is None
        assert sprout.dataroot_dir is None
        assert sprout.tensorboard_dir is None
        assert sprout.log_dir is None
        assert sprout.silent is False
        assert sprout.log_level == logging.INFO
        assert sprout.profile_name is None

    def test_training_sprout_training_namespace(self):
        """
        This test confirms every attributes in the TrainingSprout is set properly when the corresponding
        arg is defined in the Namespace.
        """
        # Initialize a TrainingSprout and feed it a fully-defined Namespace.
        sprout = TrainingSprout()
        sprout.grow_from_args(self.build_training_namespace())

        # Check all of the TrainingSprout attributes for the expected values.
        assert sprout.model_name == "test_model"
        assert sprout.num_gpus == 0
        assert sprout.dryrun is True
        assert sprout.resume is True
        assert sprout.skip_native is True
        assert sprout.onnx is True

        # Check all of the Base Sprout attributes for the expected values.
        assert sprout.workspace_dir == "workspace_dir"
        assert sprout.dataroot_dir == "dataroot_dir"
        assert sprout.tensorboard_dir == "tensorboard_dir"
        assert sprout.log_dir == "log_dir"
        assert sprout.silent is True
        assert sprout.log_level == logging.DEBUG
        assert sprout.profile_name == "profile_name"


class TestTuningSprout(TestCase):
    """
    Tests meant to exercise the TuningSprout.
    """

    @staticmethod
    def build_tuning_namespace():
        """
        This method returns a Namespace similar to the Namespace you would received when running jb_tune.
        :return: A Namespace that sets all of the args expected by the TuningSprout to non-default values.
        """
        # Create a Namespace for args that are unique to the TuningSprout.
        tune_args = Namespace(dryrun=True,
                              modelName="test_model",
                              tuningConfig="test_tuning_config")

        # Create a Namespace for the args in the Base Sprout.
        base_args = TestBaseSprout.build_base_namespace()

        # Combine the two Namespaces into a single Namespace and return the result.
        return Namespace(**vars(tune_args), **vars(base_args))

    def test_tuning_sprout_defaults(self):
        """
        This test confirms that all TuningSprout attributes are initialized to None when a TuningSprout
        is initialized.
        """
        # Create the sprout.
        sprout = TuningSprout()

        # Check the attributes that are unique to the TuningSprout.
        assert sprout.dryrun is None
        assert sprout.model_name is None
        assert sprout.tuning_config is None

        # Also check the attributes in the base Sprout.
        assert sprout.workspace_dir is None
        assert sprout.dataroot_dir is None
        assert sprout.tensorboard_dir is None
        assert sprout.log_dir is None
        assert sprout.silent is None
        assert sprout.log_level is None
        assert sprout.profile_name is None

    def test_tuning_sprout_from_empty_namespace(self):
        """
        This test checks the behavior of the TuningSprout when an empty Namespace is used to set the
        attributes in the TuningSprout.
        """
        # Initialize a TuningSprout and feed it an empty Namespace.
        args = Namespace()
        sprout = TuningSprout()
        sprout.grow_from_args(args)

        # Check all of the attributes that are unique to the TuningSprout for the expected values.
        assert sprout.dryrun is False
        assert sprout.model_name is None
        assert sprout.tuning_config is None

        # Also check all of the Base Sprout attributes for the expected values.
        assert sprout.workspace_dir is None
        assert sprout.dataroot_dir is None
        assert sprout.tensorboard_dir is None
        assert sprout.log_dir is None
        assert sprout.silent is False
        assert sprout.log_level == logging.INFO
        assert sprout.profile_name is None

    def test_tuning_sprout_training_namespace(self):
        """
        This test confirms every attributes in the TuningSprout is set properly when the corresponding
        arg is defined in the Namespace.
        """
        # Initialize a TuningSprout and feed it a fully-defined Namespace.
        sprout = TuningSprout()
        sprout.grow_from_args(self.build_tuning_namespace())

        # Check all of the TuningSprout attributes for the expected values.
        assert sprout.dryrun is True
        assert sprout.model_name == "test_model"
        assert sprout.tuning_config == "test_tuning_config"

        # Check all of the Base Sprout attributes for the expected values.
        assert sprout.workspace_dir == "workspace_dir"
        assert sprout.dataroot_dir == "dataroot_dir"
        assert sprout.tensorboard_dir == "tensorboard_dir"
        assert sprout.log_dir == "log_dir"
        assert sprout.silent is True
        assert sprout.log_level == logging.DEBUG
        assert sprout.profile_name == "profile_name"
