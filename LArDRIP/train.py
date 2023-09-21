import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

import tqdm

from dataloader import *
from model import encoder, maskTokenGenerator, decoder

import sys
sys.path.append('./mae')
import models_mae

def main(args):
    dl = MAEdataloader_dense2d('../data/example/dense_2d_voxEdep.h5',
                       batchSize = 1,
                       )
    
    # enc = encoder()
    # enc.train()

    # # these two objects need to be defined!
    # mtg = maskTokenGenerator()
    # mtg.train()
    
    # dec = decoder()
    # dec.train()
    # model = models_mae.mae_vit_large_patch16()
    model = models_mae.MaskedAutoencoderViT(img_size=112, in_chans=1)
    print (model)

    lr = 1.e-4

    optimizer = optim.Adam(model.parameters(),
                           # enc.parameters(),
                           # mtg.parameters(),
                           # dec.parameters(),
                           lr = lr)

    criterion = nn.MSELoss()

    for imageBatch in dl:
        imageBatch = torch.einsum('nhwc->nchw', imageBatch)
        loss, out, mask = model(imageBatch.float(), mask_ratio = 0.5)
        print (out)

        # loss = criterion(maskedPatches, inferrence_in_masked_regions)

        # loss.backward()
        # optimizer.step()    

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser("Train the DUNE ND convMAE thing!")
    # parser.add_argument("inputFile",
    #                     nargs = '+',
    #                     help = "input 2x2 larcv or voxelized edep-sim file")
    # parser.add_argument("preppedOutput",
    #                     help = "output patched images [hdf5]")

    args = parser.parse_args()
    main(args)
