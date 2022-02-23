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

import sys

from juneberry.pytorch.classifier_trainer import ClassifierTrainer
from opacus import PrivacyEngine

import logging

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
                        f"Overriding sample rate to: {engine_args['sample_rate']} Default sample rate was 1 / Number of batches = 1 / {self.num_batches} = {default_sample_rate} ")
                else:
                    engine_args['sample_rate'] = default_sample_rate

                if 'epochs' in engine_args:
                    if engine_args['epochs'] == self.max_epochs:
                        logger.warning(
                            f"Max epochs from juneberry config: {self.max_epochs} vs Epochs from privacy engine: {engine_args['epochs']}")
                else:
                    engine_args['epochs'] = self.max_epochs

                logger.info(f"Invoking privacy engine with: {engine_args}")
                privacy_engine = PrivacyEngine(self.model, **engine_args)
                privacy_engine.attach(self.optimizer)

                self.history.update({"sigma": [], "C": [], "epsilon": [], "target_delta": [], "alpha": []})
            else:
                logger.error("pytorch.privacy_engine.target_delta required for opacus. EXITING.")
                sys.exit(-1)
        else:
            logger.error("pytorch.privacy_engine stanza required for opacus. EXITING.")
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
