import numpy as np
import matplotlib
matplotlib.use('AGG') 
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import tqdm
import yaml
import os

from dataloader import *
from logger import *
from model import *
from utils import *

def main(args):
    with open(args.manifest) as mf:
        manifest = yaml.load(mf, Loader = yaml.FullLoader)

    dl = MAEdataloader_dense2d(manifest['trainData'],
                               batchSize = manifest['batchSize'])

    if not os.path.exists(manifest['outdir']):
        os.mkdir(manifest['outdir'])

    logger = logManager(os.path.join(manifest['outdir'],
                                     'lossHist.dat'))
    
    model = MAEViT_network(manifest).to(device)

    if args.load:
        model.load_checkpoint(args.load)
    checkpoint_out = os.path.join(manifest['outdir'],
                                  'weights.ckpt')
    plot_out = os.path.join(manifest['outdir'],
                            'test_image.png')

    optimizer = optim.Adam(model.parameters(),
                           lr = manifest['lr'])

    for n_epoch in range(manifest['maxEpoch']):
        pbar = tqdm.tqdm(enumerate(dl))
        for n_iter, imageBatch in pbar:
            imageBatch = imageBatch.to(device)
            imageBatch = torch.einsum('nhwc->nchw', imageBatch)
            loss, out, mask = model(imageBatch.float(), mask_ratio = 0.7)

            pbarMessage = " ".join(["epoch:",
                                    str(n_epoch),
                                    "iter:",
                                    str(n_iter),
                                    "loss:",
                                    str(round(loss.item(), 4))])
            pbar.set_description(pbarMessage)

            logger.update(loss)
                       
            loss.backward()
            optimizer.step()    

            if n_iter%1000 == 0:
                logger.write_to_disk()
                model.save_checkpoint(checkpoint_out)

    plot_single_image_from_batch(model, imageBatch, out, mask, plot_out)

    model.save_checkpoint(checkpoint_out)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser("Train the DUNE ND convMAE thing!")
    parser.add_argument("-l", "--load",
                        help = "load a previous model weight checkpoint file")
    parser.add_argument("-m", "--manifest",
                        help = "manifest describing a set of configuration paramters")

    args = parser.parse_args()
    main(args)
