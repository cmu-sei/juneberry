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

import tensorflow as tf


class ContinuousExponentialLrScheduler(tf.keras.callbacks.LearningRateScheduler):
    def __init__(self, schedule_string, duration):
        if schedule_string == 'fixed':
            def scheduler(epoch, lr):
                return lr
        elif schedule_string.startswith('decay_to_'):
            end_lr = float(schedule_string.split('decay_to_')[1].split('_')[0])
            if schedule_string.endswith('_after_half'):
                def scheduler(epoch, lr):
                    if epoch < duration // 2:
                        self._initial_lr = lr
                        return lr
                    else:
                        return self._initial_lr * (self._initial_lr / end_lr)**(
                            -(epoch - duration // 2) / (duration // 2))
            else:
                def scheduler(epoch, lr):
                    if epoch == 0:
                        self._initial_lr = lr
                    return self._initial_lr * (self._initial_lr / end_lr)**(
                        -epoch / duration)
        elif (schedule_string.startswith('half_') and
                schedule_string.endswith('_times')):
            times = int(schedule_string.split('half_')[1].split('_times')[0])
            period = duration // times

            def scheduler(epoch, lr):
                if epoch % period == period - 1:
                    return lr / 2.
                else:
                    return lr
        else:
            raise ValueError(f'unrecognized schedule string: {schedule_string}')
        super().__init__(scheduler, verbose=1)
