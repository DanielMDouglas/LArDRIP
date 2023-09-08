import MinkowskiEngine as ME

import torch
import torch.nn as nn
import torch.optim as optim

from layers import *

import numpy as np
import yaml
import tqdm
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')        

class model (ME.MinkowskiNetwork):
    def __init__(self, D = 3):
        super(model, self).__init__(D)

        self.layers = [Identity()]
        self.network = nn.Sequential(*self.layers)
        
    def forward(self, x):
        return self.network(x)

    def make_checkpoint(self, filename):
        print ("saving checkpoint ", filename)
        torch.save(dict(model = self.state_dict()), filename)

    def load_checkpoint(self, filename):
        print ("loading checkpoint ", filename)
        with open(filename, 'rb') as f:
            checkpoint = torch.load(f,
                                    map_location = device)
            self.load_state_dict(checkpoint['model'], strict=False)

class encoder (model):
    def __init__(self):
        super(encoder, self).__init__()

        self.layers = [ResNetEncoder(1, depth = 4)]
        self.network = nn.Sequential(*self.layers)

class maskTokenGenerator (model):
    def __init__(self):
        super(maskTokenGenerator, self).__init__()

class decoder (model):
    def __init__(self):
        super(decoder, self).__init__()
