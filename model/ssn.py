# Stereo Siamese Net

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

class StereoSiamNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = []

    def forward(self, x):
        pass

    def dump_info(self):
        return 'nothing yet'