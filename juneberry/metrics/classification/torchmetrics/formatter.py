# numpy arrays -> torch tensors
import torch

def format_input(target, preds):
    return torch.LongTensor(target), torch.FloatTensor(preds)


def format_output(result):
    return torch.Tensor.tolist(result)