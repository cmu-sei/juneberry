README

This directory contains the unit test model that trains a subset of Imagenette (https://github.com/fastai/imagenette) 
against a ResNet18 that is pre-trained on ImageNet.

To get the Imagenette data, download the 160x160 data from:

https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz

and un-tar it in your DATA_ROOT. This should expand into a directory called imagenette2_160 and have two 
directories: train and val.

NOTE on Train/Validation split.

This test uses the train portions for train/validation and the val set as the hold out set.
