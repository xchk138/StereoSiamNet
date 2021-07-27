# Stereo Siamese Net
import sys
sys.path.append('..')
from utils import AdaIN, adain

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T


class StereoSiamNet(nn.Module):
    '''
    Params:
    @input_channel  1(for gray image) or 3(for RGB image)
    @texture_encoder_config   [[4, 3, 2], [4, 3, 1]]
    this texture encoder config shows a 2 group of 2-layer conv net which firstly applys 4 filters with 3x3 kernel
    to the input gray image, and then use 1x1 kernel filters to reduce channels to 2, and then a similar
    group of ops are applied after this, while they finally reduce channels to 1.
    @style_encoder_config   [[8, 3], [16, 3], [32, 3], [64, 3]]
    this style encoder config shows a 4-group of 2-layers conv net followed by 1 mean-pooling and 1 fully connected
    layer to reshape 4D feature tensor into 2D(1 for batch, 1 for channel), the final channel dimension is determined
    by the decoder config.
    @decoder_config [[4, 3, 2], [4, 3, 1]]
    '''
    def __init__(
        self
        , input_channel = 1
        , texture_encoder_config = [[4, 3, 2], [4, 3, 1]]
        , style_encoder_config = [[8, 3], [16, 3], [32, 3], [64, 3]]
        , decoder_config = [[4, 3, 2], [4, 3, 1]]
        ):
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
        layers = [] # [[4, 3, 2], [4, 3, 1]]
        assert texture_encoder_config is not None
        assert isinstance(texture_encoder_config, tuple) or isinstance(texture_encoder_config, list)
        for i in range(len(texture_encoder_config)):
            chn_last = 0
            if i==0:
                chn_last = input_channel
            else:
                chn_last = texture_encoder_config[i-1][-1]
            head_ = nn.Conv2d(chn_last, texture_encoder_config[i][0], texture_encoder_config[i][1], 1, (1,1)) # should be NxN big kernel ops
            layers.append(head_)
            layers.append(nn.LeakyReLU(0.01, inplace=True))
            tail_ = nn.Conv2d(texture_encoder_config[i][0], texture_encoder_config[i][2], 1, 1, (0,0)) # should be 1x1 small kernel ops
            layers.append(tail_)
            layers.append(nn.LeakyReLU(0.01, inplace=True))
        self.texture_encoder = nn.Sequential(*layers)

        # the second part of encoder: style encoder
        layers = [] # [[8, 3], [16, 3], [32, 3], [64, 3]]
        assert style_encoder_config is not None
        assert isinstance(style_encoder_config, tuple) or isinstance(style_encoder_config, list)
        for i in range(len(style_encoder_config)):
            chn_last = 0
            if i==0:
                chn_last = input_channel
            else:
                chn_last = style_encoder_config[i-1][0]
            head_ = nn.Conv2d(chn_last, style_encoder_config[i][0], style_encoder_config[i][1], 2, (1,1))
            layers.append(head_)
            layers.append(nn.LeakyReLU(0.01, inplace=True))
            tail_ = nn.Conv2d(style_encoder_config[i][0], style_encoder_config[i][0], 1, 1, (1,1))
            layers.append(tail_)
            layers.append(nn.LeakyReLU(0.01, inplace=True))
        # append the final mean pooling layer to make it 2D tensor
        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.style_encoder = nn.Sequential(*layers)

        # the final part is the decoding network
        # restore images from given texture tensors and style tensors
        assert decoder_config is not None
        assert isinstance(decoder_config, tuple) or isinstance(decoder_config, list)
        self.decoder_layers = [] # [[4, 3, 2], [4, 3, 1]]
        self.decoder_miu = [] # AdaIN layers applied to each conv layer
        self.decoder_sigma = [] # AdaIN layers applied to each conv layer
        for i in range(len(decoder_config)):
            chn_last = 0
            if i==0:
                chn_last = texture_encoder_config[-1][-1]
            else:
                chn_last = decoder_config[i-1][-1]
            # generate AdaIN parameters
            fc_miu = nn.Sequential(
                nn.Linear(style_encoder_config[-1][0], 64, True),
                nn.LeakyReLU(0.01),
                nn.Linear(64, 32, True),
                nn.LeakyReLU(0.01),
                nn.Linear(32, chn_last, True)
            )
            self.decoder_miu.append(fc_miu)

            fc_sigma = nn.Sequential(
                nn.Linear(style_encoder_config[-1][0], 64, True),
                nn.LeakyReLU(0.01),
                nn.Linear(64, 32, True),
                nn.LeakyReLU(0.01),
                nn.Linear(32, chn_last, True)
            )
            self.decoder_sigma.append(fc_sigma)

            conv_ = nn.Conv2d(chn_last, decoder_config[i][0], decoder_config[i][1], 1, (1,1))
            relu_ = nn.LeakyReLU(0.01, inplace=True)
            self.decoder_layers.append(nn.Sequential(conv_, relu_))

            # generate AdaIN parameters
            fc_miu = nn.Sequential(
                nn.Linear(style_encoder_config[-1][0], 64, True),
                nn.LeakyReLU(0.01),
                nn.Linear(64, 32, True),
                nn.LeakyReLU(0.01),
                nn.Linear(32, decoder_config[i][0], True)
            )
            self.decoder_miu.append(fc_miu)

            fc_sigma = nn.Sequential(
                nn.Linear(style_encoder_config[-1][0], 64, True),
                nn.LeakyReLU(0.01),
                nn.Linear(64, 32, True),
                nn.LeakyReLU(0.01),
                nn.Linear(32, decoder_config[i][0], True)
            )
            self.decoder_sigma.append(fc_sigma)

            conv_ = nn.Conv2d(decoder_config[i][0], decoder_config[i][2], 1, 1, (0,0))
            relu_ = nn.LeakyReLU(0.01, inplace=True)
            self.decoder_layers.append(nn.Sequential(conv_, relu_))
        
        # register those modules
        assert len(self.decoder_layers) == len(self.decoder_miu)
        for i in range(len(self.decoder_layers)):
            self.add_module('decoder_layers_%d' % i, self.decoder_layers[i])
            self.add_module('decoder_miu_%d' % i, self.decoder_miu[i])
            self.add_module('decoder_sigma_%d' % i, self.decoder_sigma[i])

        self.mix_style = AdaIN()

    def forward(self, x1, x2):
        # texture part
        t1 = self.texture_encoder(x1)
        t2 = self.texture_encoder(x2)
        # style part
        # resize the input to 256x192
        x1s = F.interpolate(x1, (256, 192))
        x2s = F.interpolate(x2, (256, 192))
        s1 = self.style_encoder(x1s)
        s2 = self.style_encoder(x2s)
        s1 = s1.view(s1.size()[0], s1.size()[1])
        s2 = s2.view(s2.size()[0], s2.size()[1])
        # cross decoding
        assert len(self.decoder_miu) == len(self.decoder_layers)
        assert len(self.decoder_sigma) == len(self.decoder_layers)
        x1r = None # cross-decoded image
        x2r = None# cross-decoded image
        for i in range(len(self.decoder_layers)):
            # cross decode the left image: mix t2 with s1
            miu1 = self.decoder_miu[i](s1)
            sigma1 = self.decoder_sigma[i](s1)
            if i==0:
                x1r = self.mix_style(t2, miu1, sigma1)
            else:
                x1r = self.mix_style(x1r, miu1, sigma1)
            x1r = self.decoder_layers[i](x1r)
            # cross decode the right image: mix t1 with s2
            miu2 = self.decoder_miu[i](s2)
            sigma2 = self.decoder_sigma[i](s2)
            if i==0:
                x2r = self.mix_style(t1, miu2, sigma2)
            else:
                x2r = self.mix_style(x2r, miu2, sigma2)
            x2r = self.decoder_layers[i](x2r)
        return x1r, x2r, t1, t2

    def dump_info(self):
        return {
            'texture_encoder': self.texture_encoder,
            'style_encoder': self.style_encoder,
            'decoder_layers': self.decoder_layers,
            'decoder_miu': self.decoder_miu,
            'decoder_sigma': self.decoder_sigma,
            'mix_style': self.mix_style
            }