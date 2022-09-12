What Is Juneberry?

===========

# Introduction

Juneberry improves the experience of machine learning experimentation by providing a framework for automating the 
training, evaluation and comparison of multiple models against multiple datasets, reducing errors and improving 
reproducibility.

Other resources can be found at the [Juneberry Home Page](https://www.sei.cmu.edu/our-work/projects/display.cfm?customel_datapageid_4050=334902) 
which also has some [tutorial slides](https://resources.sei.cmu.edu/library/asset-view.cfm?assetid=875902).

# Top Down Perspective

The Juneberry system facilitates machine learning experimentation by helping users train and compare machine learning
models which may have different architectures, datasets, and/or hyperparameters. Various configuration files, in JSON
format, control the characteristics of the experiment, such as which models to use, training datasets, evaluation
datasets, and the types of reports and graphics to generate for comparison purposes.

Some of the key features of Juneberry:
* Primarily config driven, so users spend more time designing experiments in a platform independent way and less time writing/testing/debugging code
* Support for multiple backends, providing a level playing field suitable for making comparisons
* An emphasis on reproducible results
* A reduction in the tediousness and error potential when working with dozens (or hundreds) of models and their output
* Exposed execution tasks which can be invoked by existing pipeline or workflow systems
* Compatible with prototyping in Jupyter notebook environments

Juneberry focuses on handling the following machine learning tasks: loading and preparation of the data, construction 
and execution of models, inferencing against test datasets, producing reports, and providing general organization and 
management for the different types of output.

Juneberry wraps a variety of machine learning backends, such as PyTorch, Detectron2, MMDetection and Tensorflow. 
As a result, Juneberry configuration files consist of platform agnostic machine learning terms which get converted 
into the appropriate configuration settings for the desired backend. This standardizes many of the possible options 
when it comes to data loading, training loops, and random seeding, which in turn promotes more consistent cross-model 
evaluation. Since each platform may also support its own unique features, Juneberry configurations support arbitrary 
extensions which can be tailored to the needs of a particular platform.

# Bottom Up Perspective

Juneberry is a configuration based experiment management system used to ease the training and evaluation of machine 
learning models. Juneberry use platform independent model configuration files that specify the architecture, 
hyperparameters and training data pipelines to invoke various machine learning backends such as PyTorch, Detectron2, 
MMDetection or Tensorflow to produce trained models and prediction results.  Juneberry then uses supplied experiment 
outlines to train many of these models with different data sets and hyper parameters and produces sets of output that 
are combined into unified reports.

Juneberry experiment workflows can be exported to other workflow automation systems for integration into larger 
deployments across large clusters of machines.

It is configured via JSON or YAML configuration that try to be independent of the various backend systems by focusing 
on the core machine learning concepts such as loss functions, optimization functions, learning rate and input data 
characteristics. Juneberry translates these general concepts into the syntax required by each back end setting up the 
backend platform as needed.  During training Juneberry consistently handles data preparation and organization to 
feed each backend system a similar a data view as possible allowing for as even a comparison as possible. Surrounding 
the training process Juneberry organizes and marshals all the various outputs.

# Juneberry is Designed Around

* Data sets - List of input data elements such as images or tabular data and their associated meta data such as labels, bounding boxes, and segmentation masks.
* Architectures (Algorithms) and functions - Code that describes the neural network(s) to be used to produce the model and their associated functions.
* Hyperparameters - Configurations of models, functions and like.
* Models - Compositions of architectures, data sets, hyper parameters and data transforms.
* Experiments - Associations of data sets (training, retraining and test), models and the subsequent outputs.

# What Juneberry Isn't

The machine learning discipline is complex and needs a wide variety of tools. Sometimes it is easier to understaned
what something does, rather than what it does. One of the core design principles of Juneberry is to not reinvent the
wheel, but to use what already exists and fill the existing gaps. As the tool space evolves and new capabilities
become avalaible we repalce our existing custom code with other open source.

Juneberry does not do this things but relies on other open source projects for this support:

* A math or statistics package like numpy or pandas
* A machine learning package like pytorch, tensorflow, or scikit-learn
* An object detection package like torchvision, detectron2, or mmdetection
* An adversarial machine learning toolkit like ART
* An interactive platform like Jupyter notebooks
* A workflow engine like doit, snakemake or airflow
* A python environment