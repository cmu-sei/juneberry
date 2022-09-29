<img align="right" src="docs/logo.png">

README
==========

# Introduction

Juneberry improves the experience of machine learning experimentation by providing a framework for automating 
the training, evaluation, and comparison of multiple models against multiple datasets, thereby reducing errors and 
improving reproducibility.

This README describes how to use the Juneberry framework to execute machine learning tasks. Juneberry follows a (mostly)
declarative programming model composed of sets of config files (dataset, model, and experiment configurations) and
Python plugins for features such as model construction and transformation.

If you're looking for a slightly more in depth description of Juneberry see [What Is Juneberry](docs/whatis.md).

Other resources can be found at the [Juneberry Home Page](https://www.sei.cmu.edu/our-work/projects/display.cfm?customel_datapageid_4050=334902) 

# Supporting Documentation

## How to Install Juneberry

The [Getting Started](docs/getting_started.md) documentation explains how to install Juneberry. It also 
includes a simple test command you can use to verify the installation.

## Experiment Overview

The [Workspace and Experiment Overview](docs/overview.md) documentation contains information about 
the structure of the Juneberry workspace and how to organize experiments.

## Experiment Tutorial

The [Juneberry Basic Tutorial](docs/tutorial.md) describes how to create a model, train the model, 
and run an experiment.

## Configuring Juneberry

The [Juneberry Configuration Guide](docs/configuring.md) describes various ways to configure Juneberry.

## Known Warnings

During normal use of Juneberry, you may encounter warning messages. The
[Known Warnings in Juneberry](docs/known_warnings.md) documentation contains information about known warning 
messages and what (if anything) should be done about them.

## Further Reading

The [vignettes](docs/vignettes) directory contains detailed walkthroughs of various Juneberry tasks. 
The vignettes provide helpful examples of how to construct various Juneberry configuration files, 
including datasets, models, and experiments. A good start is 
[Replicating a Classic Machine Learning Result with Juneberry](docs/vignettes/vignette1/Replicating_a_Classic_Machine_Learning_Result_with_Juneberry.md).

# Copyright

Copyright 2021 Carnegie Mellon University.  See LICENSE.txt file for license terms.
