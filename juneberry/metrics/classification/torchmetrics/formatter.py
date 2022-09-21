import torch

# torchmetrics requires tensors as input
def format_input(target, preds):
    return torch.LongTensor(target), torch.FloatTensor(preds)


# TODO don't call formatting functions that noop
def format_output(result):
    return result # noop
