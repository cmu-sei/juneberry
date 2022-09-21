from juneberry.evaluation import utils as jb_eval_utils


# lifted from pytorch.evaluation.utils.compute_accuracy
def format_input(y_true, y_pred, binary):
    # Convert the continuous predictions to single class predictions
    singular_y_pred = jb_eval_utils.continuous_predictions_to_class(y_pred, binary)

    # Now call the function
    return y_true, singular_y_pred


# TODO don't call formatting functions that noop
def format_output(result):
    return result
