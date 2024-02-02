import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ColorNetwork(nn.Module):
    def __init__(self,
                 d_in,
                 rgbnet_width,
                 rgbnet_depth,
                 #bias=0.5, 
                 scale=1):
        super(ColorNetwork, self).__init__()
        
        self.scale = scale
        self.rgbnet = nn.Sequential(
                nn.Linear(d_in, rgbnet_width), nn.LayerNorm(rgbnet_width), #nn.ReLU(inplace=True),
                *[
                    nn.Sequential(nn.Linear(rgbnet_width, rgbnet_width), nn.LayerNorm(rgbnet_width)) #, nn.ReLU(inplace=True))
                    for _ in range(rgbnet_depth-2)
                ],
                nn.Linear(rgbnet_width, 3),
            )
        nn.init.constant_(self.rgbnet[-1].bias, 0)
        print('dvgo: mlp', self.rgbnet)


    def forward(self, inputs):
        rgb_logit = self.rgbnet(inputs)
        return rgb_logit #orch.sigmoid(rgb_logit)

    def rgb(self, x):
        return self.forward(x)
