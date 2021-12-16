README
======

The purpose of this model is to demonstrate how to Juneberry can be used to evaluate an ONNX model 
that was trained outside of Juneberry. There are some preparation tasks you will need to perform in 
order for this evaluation to succeed. 

 - Download a copy of the Faster R-CNN model from the ONNX model zoo. You can obtain the model here: 
https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/faster-rcnn and place
it in your models/onnx/FasterRCNN-10 directory with the name "model.onnx".
```
curl --output model.onnx https://media.githubusercontent.com/media/onnx/models/master/vision/object_detection_segmentation/faster-rcnn/model/FasterRCNN-10.onnx
```
 - Obtain a copy of the COCO 2017 validation data and place it in your data root in the directory `coco2017/val2017`.
This is the dataset used to achieve the published accuracy metric, an mAP of 0.35.
 - Note: If you obtain the COCO2017 validation data from some external source, you may need to 
adjust the label numbers in the annotations file to match the model's view of the labels. 
You can see the model labels at the following location: 
https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/faster-rcnn/dependencies/coco_classes.txt. 
The difference between the two sets of labels is subtle: the standard COCO labels sparsely cover the range from 1-90. 
The model labels remove all gaps in the numbering and use the numbers 1-80. So if you use an annotations file with 
label numbers that go up to 90, you will need to adjust those label numbers to the equivalent label number in the 
1-80 set.