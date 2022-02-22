Known Warnings in Juneberry
============

# Overview

This file documents known warning messages you may encounter when using Juneberry. Each section should 
contain some example text of the warning that you may see in a log file, along with a possible explanation 
for what's causing the issue. Not all warnings necessarily represent something that must be corrected.

## The System Test

### General Warnings

These warning may apply to multiple sections of the system test.

#### Validation dataset is Length 0

##### Message

```
WARNING Validation dataset is length 0.
```

##### Response

This warning occurs after a dataset is loaded, but there is no data in the portion of the dataset 
that was split from the main portion. This situation is commonly encountered when a dataset config 
without any sampling is chosen for evaluation. This is not a concern, because this means that every 
element of the dataset would be used during the evaluation.

### Detectron2

#### Category IDs in annotations

##### Message
```
WARNING (detectron2.data.datasets.coco:92): 
Category ids in annotations are not in [1, #categories]! We'll apply a mapping for you.
```

##### Response
Detectron2 expects COCO format metadata annotations to have IDs that start at 1 and remain 
contiguous until the full range of classes is represented. When these conditions aren't met, 
detectron2 applies a mapping which converts between the ID values that were provided, and the 
converted values that detectron2 expects. In the case of the text_detect dataset used by the 
system test, you can see from the dataset config file that the provided label IDs in the dataset 
are 0, 1, 2, thus violating the range that detectron2 expects. Therefore, a mapping will be applied 
whenever this dataset is used by detectron2.

#### Skip loading parameter

##### Message(s)
```
WARNING (fvcore.common.checkpoint:338): Skip loading parameter 
'roi_heads.box_predictor.cls_score.weight' to the model due to incompatible shapes: (81, 1024) 
in the checkpoint but (4, 1024) in the model! You might want to double check if this is expected.

WARNING (fvcore.common.checkpoint:338): Skip loading parameter 
'roi_heads.box_predictor.cls_score.bias' to the model due to incompatible shapes: (81,) 
in the checkpoint but (4,) in the model! You might want to double check if this is expected.

WARNING (fvcore.common.checkpoint:338): Skip loading parameter 
'roi_heads.box_predictor.bbox_pred.weight' to the model due to incompatible shapes: (320, 1024) 
in the checkpoint but (12, 1024) in the model! You might want to double check if this is expected.

WARNING (fvcore.common.checkpoint:338): Skip loading parameter 
'roi_heads.box_predictor.bbox_pred.bias' to the model due to incompatible shapes: (320,) 
in the checkpoint but (12,) in the model! You might want to double check if this is expected.
```

##### Response
https://github.com/facebookresearch/detectron2/issues/196 suggests this behavior is expected 
because the number of classes in the dataset differs from the number of classes from the pre-trained 
model. 

#### Model parameters or buffers not found in checkpoint

##### Message

```
WARNING (fvcore.common.checkpoint:350): Some model parameters or buffers are not found in the checkpoint:
roi_heads.box_predictor.bbox_pred.{bias, weight}
roi_heads.box_predictor.cls_score.{bias, weight}
```

##### Response
https://github.com/facebookresearch/detectron2/issues/803 suggests this warning occurs because 
the ImageNet pre-trained model being used to conduct the system test does not have model parameters 
for this detection model. Therefore, this warning message is expected, and probably safe to ignore. 
Furthermore, due to warning messages shown in the "Skip loading parameter" section, you can see there's 
a relationship between the model parameters mentioned here that were not found in the checkpoint, and the 
model parameters from the earlier warnings whose loading was skipped due to incompatible shapes.

#### Checkpoint contains keys that are not used by the model

##### Message

```
WARNING (fvcore.common.checkpoint:352): The checkpoint state_dict contains keys that are not 
used by the model:
  proposal_generator.anchor_generator.cell_anchors.{0, 1, 2, 3, 4}
```

##### Response
https://github.com/facebookresearch/detectron2/issues/803 suggests this warning occurs because 
the ImageNet pre-trained model being used to conduct the system test does not have model parameters 
for this detection model. Therefore, this warning message is expected, and probably safe to ignore.

### MMDetection

#### Model and State Dict Mismatch

##### Message

```
2022-02-07 19:00:07,287 WARNING (mmdet:104): The model and loaded state dict do not match exactly
```

##### Response

https://github.com/open-mmlab/mmdetection/issues/1151 suggests this warning occurs when the layers 
of the pre-trained model being used to conduct the system test has fully connected layers that go 
unused during the run of the system test. This should not affect training and can be ignored.

#### Testing Results of the Whole Dataset is Empty

##### Message

```
ERROR (mmdet:101): The testing results of the whole dataset is empty.
```

##### Response

https://mmdetection.readthedocs.io/en/latest/_modules/mmdet/datasets/coco.html indicates this error 
message can appear during the evaluation of a COCO dataset, which does occur during the system test. 
This error message appears to trigger if an IndexError happens while loading the predictions or 
cocoDt (detections?) when metrics are being calculated.
