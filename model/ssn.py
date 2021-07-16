# Stereo Siamese Net
import sys
sys.path.append('..')
from utils import AdaIN

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T


class StereoSiamNet(nn.Module):
    '''
    Params:
    @texture_encoder_config   [[4, 3, 2], [4, 3, 1]]
    this texture encoder config shows a 2 group of 4-layer conv net which firstly applys 4 filters with 3x3 kernel
    to the input gray image, and then use 1x1 kernel filters to reduce channels to 2, and then a similar
    group of ops are applied after this, while they finally reduce channels to 1.
    @style_encoder_config   [[128, 96], [8, 3], [16, 3], [32, 3], [64, 3]]
    this style encoder config shows a 4-group of 4-layers conv net followed by 1 mean-pooling and 1 fully connected
    layer to reshape 4D feature tensor into 2D(1 for batch, 1 for channel), the final channel dimension is determined
    by the decoder config.
    @decoder_config 
    '''
    def __init__(self, texture_encoder_config, style_encoder_config, decoder_config):
        super().__init__()
        # A gray image is sampled and separated into 2 parts
        # The first part is the detail part containing all the details of textures and objects.
        # The second part is the style part containing the neccessary style information of input image.
        # The overall 2 parts together reconstruct the input image.
        # *One is able to modify this into version of RGB/YUV input.

        # As the detail part should not loss details, we apply all strides to be 1;
        # To ensure it can run on low-computation-power devices, we use bottleneck convolution style.
        # For example, if the input is single channel(Gray image), we then apply a channel multiplication
        # K to get K feature maps, and then shrink down channel number by a pointwise convolution(1x1).
        # This is much different from Depthwise Separable Convolution. Almost all the computation is 
        # spent on NxN(N>=3) filtering, which matches import patterns, and avoids parameter growing and
        # memory expension.
        
        # the first part of encoder: texture encoder
        layers = []
        assert texture_encoder_config is not None
        assert isinstance(texture_encoder_config, tuple) or isinstance(texture_encoder_config, list)
        for i in range(len(texture_encoder_config)):
            head_ = nn.Conv2d(1, texture_encoder_config[i][0], texture_encoder_config[0][1], 1, 'same') # should be NxN big kernel ops
            layers.append(head_)
            layers.append(nn.LeakyReLU(0.01, inplace=True))
            tail_ = nn.Conv2d(1, 1, texture_encoder_config[0][2], 1, 'same') # should be 1x1 small kernel ops
            layers.append(tail_)
            layers.append(nn.LeakyReLU(0.01, inplace=True))
        self.texture_encoder = nn.Sequential(tuple(layers))

        # the second part of encoder: style encoder
        layers = []
        assert style_encoder_config is not None
        assert isinstance(style_encoder_config, tuple) or isinstance(style_encoder_config, list)
        for i in range(len(style_encoder_config)): 
            
            head_ = nn.Conv2d


    def forward(self, x):
        for i in range(len(self.texture_encoder)):
            

    def dump_info(self):
        return 'nothing yet'