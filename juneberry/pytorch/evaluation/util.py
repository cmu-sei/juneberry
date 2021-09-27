from torch import FloatTensor
from torch.nn.functional import softmax


def classify_inputs(eval_name_targets, predictions, classify_topk, dataset_mapping, model_mapping):
    """
    Determines the top-K predicted classes for a list of inputs.
    :param eval_name_targets: The list of input files and their true labels.
    :param predictions: The predictions that were made for the inputs.
    :param classify_topk: How many classifications we would like to show.
    :param dataset_mapping: The mapping of class integers to human readable labels that the DATASET is aware of.
    :param model_mapping: The mapping of class integers to human readable labels that the MODEL is aware of.
    :return: A list of which classes were predicted for each input.
    """
    # Some tensor operations on the predictions; softmax converts the values to percentages.
    prediction_tensor = FloatTensor(predictions)    # TODO: This is PyTorch specific
    predict = softmax(prediction_tensor, dim=1)     # TODO: This is also PyTorch specific
    values, indices = predict.topk(classify_topk)
    values = values.tolist()
    indices = indices.tolist()

    classification_list = []

    # Each input should have a contribution to the classification list.
    for i in range(len(eval_name_targets)):

        class_list = []
        for j in range(classify_topk):
            try:
                label_name = dataset_mapping[indices[i][j]]
            except KeyError:
                label_name = model_mapping[str(indices[i][j])] if model_mapping is not None else ""

            individual_dict = {'label': indices[i][j], 'labelName': label_name, 'confidence': values[i][j]}
            class_list.append(individual_dict)

        try:
            true_label_name = dataset_mapping[eval_name_targets[i][1]]
        except KeyError:
            true_label_name = model_mapping[str(eval_name_targets[i][1])] if model_mapping is not None else ""

        classification_dict = {'file': eval_name_targets[i][0], 'actualLabel': eval_name_targets[i][1],
                               'actualLabelName': true_label_name, 'predictedClasses': class_list}
        classification_list.append(classification_dict)

    return classification_list
