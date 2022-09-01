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
