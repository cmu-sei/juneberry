Workspace and Experiment Overview
==========

# Introduction

Juneberry supports the idea of a model zoo where config files and pre-trained models can be downloaded.
Model zoo files are stored on remote servers in a hierarchy similar to how the models directory is
constructed. Given a path to a zoo file of:

"https://junebery.com/models/my-model/resnet.zip"

- https://juneberry.com/models - The base url to where models are stored on the server.
- my-model/resnet18.zip - A zip containing the model for the model "my-model/resnet18"

# Packaging a model

The contents of the zip file are the important contents of the model that is to be shared. This is usually

- config.json
- model.pt or model.h5
- (optional) hashes.json

If desired the hashes.json can be provided indicating what specific model architecture was used to generate
this model. If this file exists and the model_architecture hash embdedded insides does NOT match the hash of 
locally constructred model architecture summary, then the model will not be loaded and an error is generated.
During training the "hashes-latest.json" file is generted which contains the model_architecture hash that was
used to train the model

A convenience tool is provided that will package up the zip file. To invoke specify the model and a directory
that represents a staging are for the zoo where the zip files is to be run. The tool is expected to be run from
the roots of the worksapce.  For exampe

`python -m juneberry.zoo my-model/resnet18 ./zoo-staging`

This command would create the file "./zoo-staging/my-model/resnet18.zip" that contains the config file, the 
model.pt (asuming this is a PyTorch model) and would have added in hashes.json (if exists) or a renamed copy 
of "hashes-latest.json" if it exists and "hashes.json" doesn't.