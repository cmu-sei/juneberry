import torch

from juneberry.evaluation import utils as jb_eval_utils


def format_input(y_true, y_pred, binary):
    # lifted from pytorch.evaluation.utils.compute_accuracy
    with torch.set_grad_enabled(False):
        # The with clause should turn off grad, but for some reason I still get the error:
        # RuntimeError: Can't call numpy() on Variable that requires grad. Use var.detach().numpy() instead.
        # So I am including detach. :(
        if binary:
            np_y_true = y_true.type(torch.DoubleTensor).unsqueeze(1).cpu().numpy()
            np_y_pred = y_pred.type(torch.DoubleTensor).cpu().detach().numpy()
        else:
            np_y_true = y_true.cpu().numpy()
            np_y_pred = y_pred.cpu().detach().numpy()

        # Convert the continuous predictions to single class predictions
        singular_y_pred = jb_eval_utils.continuous_predictions_to_class(np_y_pred, binary)

        # Now call the function
        return np_y_true, singular_y_pred


def format_output(result):
    return result
