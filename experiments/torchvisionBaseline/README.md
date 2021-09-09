# Experiment to test juneberry with the baseline torchvision models

This experiment compares the validation accuracy from the training call `jb_train` to the validation accuracy from the evaluation call to `jb_evaluate` to check that both parts of the pipeline are functioning as expected. We are using the baseline pytorch models, documented at https://pytorch.org/vision/stable/models.html, as exemplars. We attempt to match the reported Top 1 accuracy by matching their transform pipeline for evaluation. The experiment does not attempt to re-train these models. 

## Methods

* Write a torchvision wrapper script to load the pre-trained torchvision models into juneberry. It is implemented in `juneberry/juneberry/architectures/pytorch/torchvision.py`.
* Setup a base config, implemented in `juneberry/models/model_tests/torchvison_baselines/resnet18/config.json` such that 
    * the LR is zero, so that no actual training happens, 
    * the training set, implemented in `juneberry/data_sets/imagenet_single_batch_128.json`, is a single batch of size 128,
    * the model is trained for only a single epoch, and 
    * the validation set is the whole of the ImageNet validation set, implemented in `juneberry/data_sets/imagenet_validation.json` so that the validation accuracy reported during training should match both the Top 1 accuracy reported by torchvision and the Top 1 accuracy reported by `jb_evaluate`. 

The experiment outline is implemented in `juneberry/experiments/torchvisionBaseline/experiment_outline.json`. 

Instructions:

1. Install juneberry. 
2. Generate the experiments from the experiment_outline:
```
juneberry$ jb_generate_experiments -w . -d /datasets/ torchvisionBaseline
```
3. Generate the pydoit scripts
```
juneberry$ jb_experiment_to_pydoit torchvisionBaseline
```
4. Execute the pydoit scripts 
```
juneberry$ doit -f experiments/torchvisionBaseline2/dodo.py -c -d .
 ```
5. You can watch the progress as it runs with a little shell foo. During training: 
```
juneberry$ find ./models/torchvisionBaseline/ -name output.json -printf "%p", -exec jq '[.options.nnArchitecture.args.className, .results.valAccuracy[0] ] | @csv' {} \; | sort
```
And during the evaluation step:
```
juneberry]$ find ./models/torchvisionBaseline/ -name eval_log_imagenet_validation.txt -printf "%p," -exec grep '******          Accuracy' {} \; | sed 's/,\./,\n\./g' | sort
```

## Results

| Path                                             | Model                 | Training Validation | Evaluation Validation | Torchvision Reported | Train - Eval | Train - Torch |
| ------------------------------------------------ | --------------------- | ------------------- | --------------------- | -------------------- | ------------ | ------------- |
| ./torchvisionBaseline/model_0/output.json     | resnet18              |             0.6969  | 0.69758               | 0.6976               | -0.07%       | -0.07%        |
| ./torchvisionBaseline/model_1/output.json     | alexnet               |             0.5652  | 0.56522               | 0.5652               | 0.00%        | 0.00%         |
| ./torchvisionBaseline/model_2/output.json     | squeezenet1_0         |             0.5809  | 0.58092               | 0.5809               | 0.00%        | 0.00%         |
| ./torchvisionBaseline/model_3/output.json     | vgg16                 |             0.7159  | 0.71592               | 0.7159               | 0.00%        | 0.00%         |
| ./torchvisionBaseline/model_6/output.json     | googlenet             |             0.6968  | 0.69778               | 0.6978               | -0.10%       | -0.10%        |
| ./torchvisionBaseline/model_7/output.json     | shufflenet_v2_x1_0    |             0.6918  | 0.69362               | 0.6936               | -0.18%       | -0.18%        |
| ./torchvisionBaseline/model_8/output.json     | mobilenet_v2          |             0.7174  | 0.71878               | 0.7188               | -0.14%       | -0.14%        |
| ./torchvisionBaseline/model_9/output.json     | resnext50_32x4d       |             0.7751  | 0.77618               | 0.7931               | -0.11%       | -1.8%         |
| ./torchvisionBaseline/model_10/output.json    | wide_resnet50_2       |             0.7833  | 0.78468               | 0.7847               | -0.14%       | -0.14%        |
| ./torchvisionBaseline/model_11/output.json    | mnasnet1_0            |             0.7346  | 0.73456               | 0.7346               | 0.00%        | 0.00%         |

## Discussion

Overall, the `Train - Eval` results match quite nicely, in all cases within 0.2%. The results also match nicely with the results reported by torchvision, with the exception of `wide_resnet50_2`, which is off by almost 2%. It is unclear why this is the case.  

# Copyright

Copyright 2021 Carnegie Mellon University.  See LICENSE.txt file for license terms.
