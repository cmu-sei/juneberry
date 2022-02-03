README
==========


# Introduction

Juneberry improves the experience of machine learning experimentation by providing a framework for automating 
the training, evaluation, and comparison of multiple models against multiple datasets, thereby reducing errors and 
improving reproducibility.

This README describes how to use the Juneberry framework to execute machine learning tasks. Juneberry follows a (mostly)
declarative programming model composed of sets of config files (dataset, model, and experiment configurations) and
Python plugins for features such as model construction and transformation.

This README contains:
* Steps for running a simple test of your Juneberry installation
* A tutorial on running a simple experiment by hand
* Details about building your own models and experiments

As an alternative to this README, there are some vignettes located in the `juneberry/documentation/vignettes` directory 
which offer more structured walkthroughs of some common Juneberry operations. A good start is 
[Replicating a Classic Machine Learning Result with Juneberry](./juneberry/documentation/vignettes/Replicating_a_Classic_Machine_Learning_Result_with_Juneberry.md).

# TL;DR

## How to Install Juneberry

To get started with Juneberry, run these steps from your terminal:
* `git clone https://github.com/cmu-sei/Juneberry.git`
* `cd juneberry`

Next, ensure you have Docker installed by following the installation steps
[here - (external link)](https://docs.docker.com/get-docker/).

Now you will need to build your Docker image. You can build the image using either
`docker build -f cudadev.Dockerfile -t juneberry/cudadev:dev .` or
`docker build -f cpudev.Dockerfile -t juneberry/cpudev:dev .`, depending on if you are working in a CPU or CUDA
environment.  There is a sample script you can use for building if you need to pass in proxy variables.  
See `docker/build.sh` for details.

After building your Docker image, either use the sample project directory layout or edit the mount points listed in the 
[`enter_juneberry_container`](./docker/enter_juneberry_container) script. Once the mount points are updated, 
you can run `./docker/enter_juneberry_container` from within the `juneberry` directory to launch a Docker
container.

If you have successfully entered the container, you will see a "CPU Development" or "CUDA Development" banner
followed by the current version. If an error occurs, double-check the paths you specified for the mount points 
in your`enter_juneberry_container` file and try again.

From inside the container, the final step is to run `pip install -e .`. 
**NOTE:** Because `enter_juneberry_container` creates a temporary container, you must perform this step every time 
you use `enter_juneberry_container`.

You can reference the [Building Juneberry Docker Containers](docs/building_docker.md) documentation for more 
information.

### Other environments

Juneberry is a fully-featured Python package and can be installed into any environment, such as venv. Once Python has
been installed, simply install Juneberry using `pip install -e .` from within the Juneberry directory. This will
set up the package, including its dependencies, and add the bin directory to your path. NOTE: This requires that you
have manually followed installation instructions for MMDetection and Detectron2.

### Using bash completion
Bash completion is available for the `jb_run_experiment`, `jb_train`, and `jb_evaluate` scripts. You can source 
bash completion using `source scripts/juneberry_completion.sh`, or source the script into your .bashrc or Dockerfile.

## How to test your installation

Run the following command from within your pipenv shell or docker container:
```jb_train -w . -d . tabular_multiclass_sample```.

This should train a basic tabular dataset, and the final epoch should report a training accuracy of 97.58%.

# Experiment Overview

The [Workspace and Experiment Overview](docs/overview.md) documentation contains information about 
the structure of the Juneberry workspace and how to organize experiments.

# Experiment Tutorial

The [Juneberry Basic Tutorial](docs/tutorial.md) describes how to create a model, train the model, 
and run an experiment.

# Special Configuration Variables

## JUNEBERRY_CUDA_MEMORY_SUMMARY_PERIOD

When this environment variable is set, the `torch.cuda.memory_summary()` will appear during training after 
the model is loaded and again after the specified period of epochs starting at epoch one.  
In other words, if JUNEBERRY_CUDA_MEMORY_SUMMARY_PERIOD is set to 10, the memory summary will be emitted 
after model load and again after epochs 1, 11, 21, 31, etc.

## Known Warnings

During normal use of Juneberry, you may encounter warning messages. The
[Known Warnings in Juneberry](docs/known_warnings.md) documentation contains information about known warning 
messages and what (if anything) should be done about them.

## Further Reading

As a reminder, the `juneberry/documentation/vignettes` directory contains more detailed walkthroughs of various 
Juneberry tasks. The vignettes provide helpful examples of how to construct various Juneberry configuration files, 
including datasets, models, and experiments. A good start is 
[Replicating a Classic Machine Learning Result with Juneberry](./juneberry/documentation/vignettes/Replicating_a_Classic_Machine_Learning_Result_with_Juneberry.md).


# Copyright

Copyright 2021 Carnegie Mellon University.  See LICENSE.txt file for license terms.
