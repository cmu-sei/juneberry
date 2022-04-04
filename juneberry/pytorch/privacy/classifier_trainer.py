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
from opacus import PrivacyEngine
import sys

from juneberry.pytorch.classifier_trainer import ClassifierTrainer

logger = logging.getLogger(__name__)


class PrivacyTrainer(ClassifierTrainer):
    def setup(self):
        super().setup()

        if 'privacy_engine' in self.pytorch_options:
            engine_args = self.pytorch_options['privacy_engine']
            if 'target_delta' in engine_args:

                default_sample_rate = 1.0 / self.num_batches
                if 'sample_rate' in engine_args:
                    logger.warning(
                        f"Overriding sample rate to: {engine_args['sample_rate']} Default sample rate was 1 / Number "
                        f"of batches = 1 / {self.num_batches} = {default_sample_rate} ")
                else:
                    engine_args['sample_rate'] = default_sample_rate

                if 'epochs' in engine_args:
                    if engine_args['epochs'] == self.max_epochs:
                        logger.warning(
                            f"Max epochs from juneberry config: {self.max_epochs} vs Epochs from privacy engine: "
                            f"{engine_args['epochs']}")
                else:
                    engine_args['epochs'] = self.max_epochs

                logger.info(f"Invoking privacy engine with: {engine_args}")
                privacy_engine = PrivacyEngine(self.model, **engine_args)
                privacy_engine.attach(self.optimizer)

                self.history.update({"sigma": [], "C": [], "epsilon": [], "target_delta": [], "alpha": []})
            else:
                logger.error("pytorch.privacy_engine.target_delta required for opacus. Exiting.")
                sys.exit(-1)
        else:
            logger.error("pytorch.privacy_engine stanza required for opacus. Exiting.")
            sys.exit(-1)

    def summarize_metrics(self, train, metrics) -> None:
        super().summarize_metrics(train, metrics)

        if train:
            epsilon, alpha = self.optimizer.privacy_engine.get_privacy_spent(self.optimizer.privacy_engine.target_delta)

            self.history['sigma'].append(self.optimizer.privacy_engine.noise_multiplier)
            self.history['C'].append(self.optimizer.privacy_engine.max_grad_norm)
            self.history['epsilon'].append(epsilon)
            self.history['target_delta'].append(self.optimizer.privacy_engine.target_delta)
            self.history['alpha'].append(alpha)
