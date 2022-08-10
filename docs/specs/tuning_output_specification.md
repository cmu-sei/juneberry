Tuning Output Specification
==========

# Introduction

This document describes the JSON format that summarizes the results after using Juneberry to 
conduct hyperparameter tuning on a model. In a tuning output file, you'll find information such 
as which tuning options were used, how long it took tune, and a summary of some tuning results 
data. These files are typically found inside a model's tuning directory, inside the subdirectory 
associated with a particular tuning run.

# Schema
```
{
    "format_version": <linux style version string of the format of this file>,
    "options": {
        "model_name": <string indicating the name of the model that was tuned>,
        "tuning_config": <string indicating the name of the tuning configuration file that 
        defined how the tuning run was conducted>
    },
    "results": {
        "best_trial_id": <string indicating the ID of the "best" trial, as determined by the Tuner>,
        "best_trial_params": <dictionary indicating the hyperparameters selections used during the "best" trial>,
        "trial_results": [
            {
                "directory": <string indicating the name of the directory where you can find all of the 
                files related to this trial>,
                "id": <string indicating the ID Ray Tune used to refer to this trial>,
                "num_iterations": <integer indicating how many times metrics were reported to the Tuner 
                during this trial>,
                "params": {<dictionary which hyperparameters were modified, and what the values of those 
                hyperparameters were during this Trial>},
                "result_data": {<dictionary containing a record of all the data stored in a Ray Tune
                result.json file, but presented in a slightly different format here>}
            }
        ]
    },
    "times": {
        "duration": <integer indicating how many seconds it took for the tuning run to complete>,
        "end_time": <string indicating the timestamp when tuning ended>,
        "start_time": <string indicating the timestamp when tuning started>
    }
}
```

# Details
This section provides more information about each of the fields in the schema. 

## format_version
Linux style version of **format** of the file. Not the version of 
the data, but the version of the semantics of the fields of this file. 
Current: 0.1.0

## options
The fields inside this section describe which model was tuned, and what tuning 
configuration file was used to configure the tuning run.

### model_name
This string indicates which model was tuned.

### tuning_config
This string indicates which tuning configuration file was used to define the tuning 
configuration options that were used during this tuning run.

## results
The fields inside this section contain data from the Trials that took place during the 
tuning run.

### best_trial_id
This string indicates the ID of the trial that the Tuner determined was the "best", according 
to the criteria specified in the tuning config.

### best_trial_parameters
This dictionary describes the hyperparameter settings that were used during the "best" trial. Each 
key in the dictionary corresponds to the name of a hyperparameter, while the value indicates what 
the hyperparameter was set to during the trial.

### trial_results
This field is a list of information about all the trials that took place during the tuning run. Each 
element in this list will correspond to one trial.

#### directory
This string indicates the name of a directory where you can find all the files related to the trial.

#### id
This string indicates the ID that Ray Tune assigned to the trial.

#### num_iterations
This integer indicates how many rounds of metrics were reported to the Tuner during the trial. Later, in 
the result_data property, you should find that the data lists associated with each will contain a number 
of data points equal to the number of iterations specified by this property.

#### params
This dictionary describes the hyperparameter settings that were used during this trial. If the ID of 
the current trial matches the ID of the "best" trial, then you would expect the value of this field to 
be identical to the value of "best_trial_parameters".

#### result_data
This dictionary presents an alternate depiction of the information found inside the Ray Tune result.json 
file that's typically found inside the trial's directory. In this dictionary, you will find a key matching 
all the keys in the Ray Tune result.json. Each value of the key is set to a list, and the lists are 
populated by what the value for that key was during each iteration of the trial. So if N number of 
trial iterations are found in the Ray Tune result.json file, then each list in this field's dictionary 
will contain N elements.

## times
This section provides timing information describing the tuning run.

### duration
The time (in seconds) spent tuning the model. Calculated by end_time - start_time.

### end_time
A timestamp indicating when the tuning run ended.

### start_time
A timestamp indicating when the tuning run started.

# Copyright

Copyright 2022 Carnegie Mellon University. See LICENSE.txt file for license terms.