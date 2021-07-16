import torch
import torch.nn as nn

class AdaIN(nn.Module):
    def __init__(self):
        super().__init__()
        pass
    def forwad(x, miu, sigma):
        assert(len(x.shape)==4) # must be 4D tensor
        assert(len(miu.shape)==4)
        assert(len(sigma.shape)==4)
        mean_ = torch.mean(x, dim=[1, 2], keepdim=True)
        std_ = torch.std(x, dim=[1,2], keepdim=True) + 1e-6
        return ((x - mean_)/std_)*sigma + miu
