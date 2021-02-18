#! /usr/bin/env python3

# ==========================================================================================================================================================
#  Copyright 2021 Carnegie Mellon University.
#
#  NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS"
#  BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER
#  INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED
#  FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM
#  FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT. Released under a BSD (SEI)-style license, please see license.txt
#  or contact permission@sei.cmu.edu for full terms.
#
#  [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see
#  Copyright notice for non-US Government use and distribution.
#
#  This Software includes and/or makes use of the following Third-Party Software subject to its own license:
#  1. Pytorch (https://github.com/pytorch/pytorch/blob/master/LICENSE) Copyright 2016 facebook, inc..
#  2. NumPY (https://github.com/numpy/numpy/blob/master/LICENSE.txt) Copyright 2020 Numpy developers.
#  3. Matplotlib (https://matplotlib.org/3.1.1/users/license.html) Copyright 2013 Matplotlib Development Team.
#  4. pillow (https://github.com/python-pillow/Pillow/blob/master/LICENSE) Copyright 2020 Alex Clark and contributors.
#  5. SKlearn (https://github.com/scikit-learn/sklearn-docbuilder/blob/master/LICENSE) Copyright 2013 scikit-learn
#      developers.
#  6. torchsummary (https://github.com/TylerYep/torch-summary/blob/master/LICENSE) Copyright 2020 Tyler Yep.
#  7. adversarial robust toolbox (https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/LICENSE)
#      Copyright 2018 the adversarial robustness toolbox authors.
#  8. pytest (https://docs.pytest.org/en/stable/license.html) Copyright 2020 Holger Krekel and others.
#  9. pylint (https://github.com/PyCQA/pylint/blob/master/COPYING) Copyright 1991 Free Software Foundation, Inc..
#  10. python (https://docs.python.org/3/license.html#psf-license) Copyright 2001 python software foundation.
#
#  DM20-1149
#
# ==========================================================================================================================================================

from torch.utils.tensorboard import SummaryWriter


class TensorBoardManager:
    """
    Responsible for logging data for TensorBoard.
    """

    def __init__(self, tb_root, model_manager):
        self.tensorboard_root = tb_root
        self.log_dir = model_manager.create_tensorboard_directory_name(tb_root)
        self.summary_writer = SummaryWriter(log_dir=self.log_dir)

        layout = {
            'Accuracy': {
                'accuracy': ['Multiline', ['accuracy/combined', 'accuracy/train', 'accuracy/val']]
            },
            'Learning Rate': {
                'learning rate': ['Multiline', []]
            },
            'Loss': {
                'loss': ['Multiline', ['loss/combined', 'loss/train', 'loss/val']]
            }
        }
        self.summary_writer.add_custom_scalars(layout)

    def update(self, history, epoch) -> None:
        """
        Write data to the tensorboard log.
        :param history: A data structure that tracks the training history
        :param epoch: An epoch number that can be used to look up a particular moment in the history.
        :return:
        """
        self.summary_writer.add_scalar('Accuracy/train', history['accuracy'][epoch], epoch)
        self.summary_writer.add_scalar('Accuracy/val', history['val_accuracy'][epoch], epoch)
        self.summary_writer.add_scalars('Accuracy/combined', {'train': history['accuracy'][epoch],
                                                              'val': history['val_accuracy'][epoch]}, epoch)
        self.summary_writer.add_scalar('Loss/train', history['loss'][epoch], epoch)
        self.summary_writer.add_scalar('Loss/val', history['val_loss'][epoch], epoch)
        self.summary_writer.add_scalars('Loss/combined', {'train': history['loss'][epoch],
                                                          'val': history['val_loss'][epoch]}, epoch)
        self.summary_writer.add_scalar('Learning Rate', history['lr'][epoch], epoch)

    def close(self) -> None:
        """
        Closes the summary writer.
        :return:
        """
        self.summary_writer.close()
