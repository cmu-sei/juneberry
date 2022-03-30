Juneberry Basic Tutorial
==========

# Step 1 - Create a dataset config file
Example dataset config file: [data_sets/imagenette_unit_train.json](../data_sets/imagenette_unit_train.json)

For this example you will need the Imagenette data, which you can obtain by following the 
steps [here](../models/imagenette_160x160_rgb_unit_test_pyt_resnet18/README.md).

Dataset configs follow the format described in 
[dataset_configuration_specification.md](specs/dataset_configuration_specification.md) in the
Juneberry documentation directory. These config files are used to describe a dataset 
which is composed of image directories, labels, sampling criteria
and **desired** image properties. Remember, the paths in the config are relative
to an externally specified `data_root`.

# Step 2 - Create a model config file
Example model config file: 
[models/imagenette_160x160_rgb_unit_test_pyt_resnet18/config.json](../models/imagenette_160x160_rgb_unit_test_pyt_resnet18/config.json)

Model configs follow the format described in 
[model_configuration_specification.md](specs/model_configuration_specification.md) in
the Juneberry documentation directory. Model configs must be named "config.json" and placed in a model sub-directory 
inside the "models" directory. These config files specify details such as model architecture, hyperparameters, datasets,
transforms, etc. See the documentation for details describing supported model config properties.

# Step 3 - Train
This step demonstrates how to use the training script to train a model. The most commonly
used training script is [jb_train](../bin/jb_train), which requires a
model name (i.e. the name of a sub-directory in the "models" directory containing a valid config.json 
file) as input. The output of the training process is a trained model and training metrics in an 
output JSON file.

**NOTE:** To train with TensorBoard activated, either set a system environment variable
```JUNEBERRY_TENSORBAORD="path/to/dir"``` or configure the tensorboard root via `-t` when calling the 
training script.

The training script also needs to know which workspace to use. By default, most scripts use the current working
directory as the workspace, unless another workspace has been specified via `-w`. By default, the data root is
assumed to be a peer directory to the workspace called `dataroot`, but it can also be specified via `-d`.

If your environment adheres to the common lab layout structure, you simply need to execute the following command 
within your juneberry directory to train the sample model:

```jb_train imagenette_160x160_rgb_unit_test_pyt_resnet18```

If you need to specify a workspace and dataroot outside the common lab layout structure, the command would 
take the following form:

```jb_train -w <path-to-workspace> -d <path-to-dataroot> imagenette_160x160_rgb_unit_test_pyt_resnet18```

The script provides ongoing status to the console (which can be silenced with the `-s` flag) 
as well as the model, log files, and visual summaries. Some output files you may see in the
`imagenette_160x160_rgb_unit_test_pyt_resnet18` model folder include:

* 📝**model.pt:** Your trained model with all parameters.
* 📁`train` A new folder containing training data
    * 📝**log.txt:** A text file with the command line output which you saw during training.
    * 📝**output.json:** The training data organized by type for all epochs in json format.
    * 🖼️**output.png:** An image showcasing the training and validation loss & accuracy of the most recent run. 

"dryrun" mode offers an opportunity to observe what actions the training script would 
perform without actually performing the training. You can initiate dryrun mode by 
adding the `--dryrun` argument to the training command. In dryrun mode, Juneberry 
logs messages about the actions it would have taken using a `<<DRY_RUN>>` prefix. 
The dryrun log output will be saved to a logfile named `log_dryrun.txt`.

## Notes on Dataset Selection (Images)

During training, images are loaded based on the specified dataset config file. The dataset config describes the sources
of data to pull from, their labels, and how the data should be sampled. The model config also defines the composition of
the validation dataset.  The order in which these data items are processed is impacted by the `seed` value specified in
the config. This seed value is used when setting up randomization.

The dataset construction process works as follows:

1) The directory specified in each `data` stanza is scanned for files.
1) Each `data` file list is individually sampled based on the sampling stanza in the dataset config. 
If two or more stanzas have the same label, the sampling is per stanza file list not per the
aggregate label. 
Thus, if the two stanzas are for label 0 with a shuffling of "random_quantity" with size 10, then 20 
images with label 0 in total will be selected.
If the sampling involves random operations, then a seed may be specified for the sampling process.
1) The file list in each data section is then split into training and validation sets 
based on the validation split stanza in the model config.
If the validation selection involves random operations then a seed may be specified 
for the process.
1) The training and validation sets from each entry are then merged into one list for
training and one for validation, respectively.
1) The file lists are provided to other parts of the system for cache checks, shuffling, 
loading, and transformation.

# Step 4 - Evaluation
The next step is to evaluate a test set against the trained model and see
how well the model performs. Evaluation require a trained model, like the one from step 3, 
and a dataset config describing the evaluation dataset. As with the other commands, the evaluation 
script requires workspace and data roots to be set either `-w`/`-d` switches or via environment variables.

This example command demonstrates how to test the model trained in the previous step on a 
new dataset it has not seen before. The workspace root and data root are not shown in this command:

```jb_evaluate imagenette_160x160_rgb_unit_test_pyt_resnet18 data_sets/imagenette_unit_test.json```

This script produces the following files in the `imagenette_160x160_rgb_unit_test_pyt_resnet18` model folder:

* 📁`eval` A new folder containing eval data
    * 📁`imagenette_unit_test` A sub-directory indicating whose name matches the dataset that was evaluated 
        * 📝`log.txt` : A text file with the command line output generated during the evaluation
        * 📝`metrics.json` : The evaluation metrics in JSON format
        * 📝`predictions.json` : The evaluation predictions in JSON format
    
Evaluation output files follow the format described in 
[eval_output_specification.md](specs/eval_output_specification.md) in
the Juneberry documentation directory.

# Step 5 - Format Report
The data from a predictions file can be used to construct ROC plots. The example command below demonstrates 
how to plot the ROC data for particular labels in a predictions file. 

```
jb_plot_roc 
   -f models/imagenette_160x160_rgb_unit_test_pyt_resnet18/predictions_imagenette_unit_test.json 
   -p "0,217,482,491,497,566,569,571,574,701" 
   models/imagenette_160x160_rgb_unit_test_pyt_resnet18/myplot.png
```
This example command adheres to the following structure:
```
jb_plot_roc -f [prediction file path] -p [classes] [output location]
```

The `-f` switch indicates which predictions file the data should come from. The `-c` switch indicates 
integer labels of the classes to include on the plot. The `myplot.png` indicates the filename to use when 
saving the plot. It is possible to plot data from multiple predictions files 
on the same plot. To do so, simply add another `-f` switch and the desired file, along with another 
`-c` switch with the desired class integers.

# Step 6 - Building experiments

## Using jb_run_experiment:
Example experiment config file: 
[experiments/classificationSmokeTest/config.json](../experiments/smokeTests/classify/config.json)

Much like the "models" directory, sub-directories in the "experiments" directory describe separate experiments to run. 
Each experiment directory contains a `config.json` defining the actions in the experiment, such as which models to 
train, how those models should be evaluated, and which reports should be generated. The experiment config file follows 
the format specified in 
[experiment_configuration_specification.md](specs/experiment_configuration_specification.md). 

## Experiment configs:
An experiment config file consists of two primary sections:

1. The "models" section: lists each model to be trained and the set of datasets to be tested against each 
   model. Each evaluation receives a tag for identification purposes for reference in the reports section.

2. The "reports" section: lists a series of reports to be generated where each report has a type (e.g. ROC
   curve), and a list of tags from the models section. Optionally, a set of classes can be specified for the report
   for when large datasets (e.g. imagenet) are used in the experiment.

Once an experiment config is created, it can be executed with the [jb_run_experiment](../bin/jb_run_experiment) 
command. As with other scripts, the workspace root and data root need to be properly configured.

## Doit backend
Behind the scenes, `jb_run_experiment` uses the [Doit]("https://pydoit.org/) task runner. `jb_run_experiment`
generates two Doit files based on the experiment config file:
1) `main_dodo.py`: this file executes the workflow specified in the experiment config file 
2) `dryrun_dodo.py`: this file describes in-depth the actions that the `main` experiment would perform without actually
   executing them.
   
## Execution flags
* The default behavior of `jb_run_experiment` is to run in preview mode. This mode lists the tasks that would be
  executed by the workflow, along with the corresponding status messages.
  
* To actually execute these tasks and save your changes, use the `--commit` or `-X` flag. Note: Only tasks identified 
  as not "up-to-date" will be executed.

* To use dryrun (as described above in "Step 3") on the models, use the `--dryrun` or `-D` flag.

* To clean the files associated with either the main or dryrun tasks, use the `--clean` or `-C` flag.

* The `--processes` or `-N` flag specifies the number of GPUs on which you would like to run parallel model
  trainings. The default behavior runs one training process at a time across all available GPUs.
  
* To regenerate experiment config files from an experiment outline, use the `--regen` or `-R` flag.

* The abbreviated, single-character versions of these flags can be appended to form one flag, e.g. `--regen --commit
  --clean` can be specified using `-RXC`.
  
## Sample commands
The example commands below demonstrate how to use the various flags to run and clean an experiment. The
classificationSmokeTest trains three models and generates three reports plus one summary report.

Running the `dryrun` of the workflow:
* Run this command to run the classificationSmokeTest experiment in `dryrun`:
```jb_run_experiment classificationSmokeTest --commit --dryrun```
* Run this command to preview which tasks from dryrun can be cleaned:
```jb_run_experiment classificationSmokeTest --dryrun --clean```
* Run this command to actually clean the dryrun tasks:
```jb_run_experiment classificationSmokeTest -XDC```
  
Running the `main` workflow:
* Run this command to preview which Doit tasks the main workflow would execute:
```jb_run_experiment classificationSmokeTest```
* Run this command to execute the main workflow:
```jb_run_experiment classificationSmokeTest --commit```
* Run this command to clean the main tasks:
```jb_run_experiment classificationSmokeTest --commit --clean```
