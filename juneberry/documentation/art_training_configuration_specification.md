ART Training Configuration Specification
==========


# Introduction

This document describes the JSON format that controls the configuration 
options for performing adversarial training using the multi-class classifier 
script. This format only contains parameters that are specific to adversarial 
training.

# Schema
```
{
    "epsilon": <float>,
    "epsStep": <float>,
    "maxIter": <integer>,
    "ratio": <float between 0 and 1>,
}
```

# Details
This section provides the details of each of the fields.

## epsilon
The maximum perturbation that the attacker can introduce.

## epsStep
Attack step size (input variation) at each iteration.

## maxIter
The maximum number of iterations.

## ratio
The proportion of samples in each batch to be replaced with their 
adversarial counterparts. Setting this value to 1 performs training
using only adversarial examples.

# Copyright

Copyright 2021 Carnegie Mellon University.  See LICENSE.txt file for license terms.
