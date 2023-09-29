import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import tqdm

from dataloader import *
from logger import *
# from model import encoder, maskTokenGenerator, decoder

import sys
sys.path.append('./mae')
import models_mae

def map_EtoX(im, alpha):
    # map an image from energy space [0, inf],
    # to prediction space [-inf, inf], mu = 0, std \approx 1
    # epsilon = 0.001
    # return torch.atanh(2*alpha*(im + epsilon) - 1)
    return alpha*im

def map_XtoE(im, alpha):
    # map an image from prediction space [-inf, inf], mu = 0, std \approx 1,
    # to energy space [0, inf]    
    # epsilon = 0.001
    # return (torch.tanh(im) + 1)/(2*alpha) - epsilon
    return im/alpha

def main(args):
    dl = MAEdataloader_dense2d('../data/example/dense_2d_voxEdep.h5',
                               batchSize = 50,
                               sequentialLoad = True,
                           )

    logger = logManager('dl_training/lossHist.dat')
    
    # model = models_mae.MaskedAutoencoderViT(img_size=112, in_chans=1).to(device)
    model = models_mae.MaskedAutoencoderViT(img_size=112, patch_size=8, in_chans=1,
                                            embed_dim=512, depth=12, 
                                            decoder_embed_dim=256, decoder_depth=4).to(device)

    if args.load:
        checkpoint = torch.load(args.load, map_location = device)
        model.load_state_dict(checkpoint['model'], strict = False)

    lr = 1.e-5

    optimizer = optim.Adam(model.parameters(),
                           lr = lr)

    # for now, let's train on a single image...
    # for imageBatch in dl:

    imageBatch = next(dl.__iter__()).to(device)
    # alpha = 1./torch.max(imageBatch)
    # print (alpha)
    # print(imageBatch)
    # imageBatch = map_EtoX(imageBatch, alpha)
    # print(imageBatch)

    imageBatch = torch.einsum('nhwc->nchw', imageBatch)
    pbar = tqdm.tqdm(range(args.max_iter))
    for n_iter in pbar:
        loss, out, mask = model(imageBatch.float(), mask_ratio = 0.7)
        
        pbarMessage = " ".join(["iter:",
                                str(n_iter),
                                "loss:",
                                str(round(loss.item(), 4))])
        pbar.set_description(pbarMessage)

        logger.update(loss)
                       
        loss.backward()
        optimizer.step()    

        if n_iter%1000 == 0:
            logger.write_to_disk()
            torch.save(dict(model = model.state_dict()), args.checkpoint)

    out = model.unpatchify(out)
    # out = map_XtoE(out, alpha)
    out = torch.einsum('nchw->nhwc', out).detach().cpu()
    # out = out.detach().cpu()

    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 * model.num_chans)
    mask = model.unpatchify(mask)
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

    # image = map_XtoE(imageBatch, alpha)
    image = torch.einsum('nchw->nhwc', imageBatch).cpu()

    im_masked = image * (1 - mask)

    im_paste = image * (1 - mask) + out * mask

    fig = plt.figure()
    fig.set_figheight(20)
    fig.set_figwidth(80)

    ax = fig.add_subplot(141)
    draw_dense_image(ax, image[0])

    ax = fig.add_subplot(142)
    draw_dense_image(ax, im_masked[0])

    ax = fig.add_subplot(143)
    draw_dense_image(ax, out[0])

    ax = fig.add_subplot(144)
    draw_dense_image(ax, im_paste[0])

    fig.savefig(args.output)

    torch.save(dict(model = model.state_dict()), args.checkpoint)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser("Train the DUNE ND convMAE thing!")
    # parser.add_argument("inputFile",
    #                     nargs = '+',
    #                     help = "input 2x2 larcv or voxelized edep-sim file")
    # parser.add_argument("preppedOutput",
    #                     help = "output patched images [hdf5]")
    parser.add_argument("-n", "--max_iter",
                        type = int,
                        help = "maximum number of training iterations")
    parser.add_argument("-l", "--load",
                        help = "load a previous model weight checkpoint file")
    parser.add_argument("-c", "--checkpoint",
                        help = "save a model weight checkpoint file")
    parser.add_argument("-o", "--output",
                        help = "output image destination")

    args = parser.parse_args()
    main(args)
