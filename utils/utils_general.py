import torch
import torch.utils.data as data
import torch.nn as nn
from utils.config import *

def _cuda(x):
    # if USE_CUDA:
    #     return x.cuda()
    # else:
    #     return x
    if isinstance(x, torch.Tensor):
        return x.cuda()
    else:
        return torch.Tensor(x).cuda()
