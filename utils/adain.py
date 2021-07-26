import torch
import torch.nn as nn

class AdaIN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, miu, sigma):
        assert(len(x.shape)==4) # must be 4D tensor
        assert(len(miu.shape)==2)
        assert(len(sigma.shape)==2)
        miu = miu.view(miu.size()[0], miu.size()[1], 1, 1)
        sigma = sigma.view(sigma.size()[0], sigma.size()[1], 1, 1)
        mean_ = torch.mean(x, dim=[2, 3], keepdim=True) # NCHW order tenosr
        std_ = torch.std(x, dim=[2, 3], keepdim=True) + 1e-6
        return ((x - mean_)/std_)*sigma + miu
