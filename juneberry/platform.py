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

class PlatformDefinitions:
    def get_model_filename(self) -> str:
        """ :return: The name of the model file that the trainer saves and what evaluators should load"""
        pass

    def get_config_suffix(self) -> str:
        """
        Before training we emit the fully realized configuration file used by the platform. Different backend platforms
        use different file types and while Juneberry names them all "platform_config", they need to have the correct
        suffix and format. This routine returns the suffix used by the platform, such as ".json" or ".yaml." The
        default format is ".json"
        :return: The suffix used when saving the realized platform_config file before training.
        """
        return ".json"

    def has_platform_config(self) -> bool:
        # TODO: This is somewhat of a hack
        return True
