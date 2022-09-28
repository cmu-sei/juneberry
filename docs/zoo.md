Model Zoo Overview
==========

# Introduction

Juneberry supports the idea of a model zoo, which contains config files and pre-trained models that 
can be downloaded and used in Juneberry. Model zoo files are stored on remote servers in a hierarchy 
similar to how the 'models' directory is organized. Consider the following path to a zoo file:

"https://juneberry.com/models/my-model/resnet.zip"

- https://juneberry.com/models - The base url of the server where the models are stored.
- my-model/resnet18.zip - A zip containing the model named "my-model/resnet18".

# Packaging a model

The model zip file contains any necessary files required to share the model. These typically include the 
following:

- config.json
- model.pt or model.h5
- (optional) hashes.json

When provided, the hashes.json file can confirm which model architecture was used to generate
the model. If the model_architecture hash embedded inside the hashes.json does NOT match the hash of 
locally constructed model architecture summary, then the model will not be loaded from the zoo and an 
error is generated. During training, a "hashes-latest.json" file will be produced which contains the 
model_architecture hash that was used to train the model.

A convenience tool is provided which packages up the zip file. To invoke the tool, specify the model and 
a directory representing a staging area for zip files to be uploaded to the zoo. The tool expects to be run from
the root of the workspace. Consider the following command:

`python -m juneberry.zoo my-model/resnet18 ./zoo-staging`

This command would create the file "./zoo-staging/my-model/resnet18.zip" containing the model's config file, the 
model.pt (assuming it is a PyTorch model), and a hashes.json file (if one exists) or a copy 
of "hashes-latest.json", if one exists, renamed to "hashes.json".