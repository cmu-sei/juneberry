# numpy arrays -> torch tensors
import torch

def format_input(target, preds):
    #return torch.IntTensor(target), torch.FloatTensor(preds)
    return target, preds # noop


def format_output(result):
    return result # noop
    #return torch.Tensor.tolist(result)