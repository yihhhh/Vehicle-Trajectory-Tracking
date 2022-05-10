import os
import torch
import numpy as np
from scipy.stats import truncnorm

from drl.utils.config import Config

def tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    x = np.asarray(x, dtype=np.float)
    x = torch.tensor(x, device=Config.DEVICE, dtype=torch.float32)
    return x

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)