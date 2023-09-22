import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    
    # model = models_mae.MaskedAutoencoderViT(img_size=112, in_chans=1).to(device)
    model = models_mae.MaskedAutoencoderViT(img_size=112, in_chans=3).to(device)

    lr = 1.e-4

    optimizer = optim.Adam(model.parameters(),
                           lr = lr)

    # for now, let's train on a single image...
    # for imageBatch in dl:
    # imageBatch = next(dl.__iter__()).to(device)
    # print ('shape', imageBatch.shape)
    from matplotlib.image import imread
    # imageBatch = torch.tensor(np.expand_dims(np.mean(imread('../data/example/fox.jpg'), axis = -1), (0, -1)))
    imageBatch = torch.tensor(np.expand_dims(imread('../data/example/fox.jpg'), 0))/256
    imageBatch = imageBatch.to(device)
    print ('shape', imageBatch.shape)
    # imageBatch -= torch.mean(imageBatch)
    # imageBatch /= torch.std(imageBatch)

    imageBatch = torch.einsum('nhwc->nchw', imageBatch)
    max_iter = 10
    pbar = tqdm.tqdm(range(max_iter))
    for n_iter in pbar:
        # imageBatch = torch.einsum('nhwc->nchw', imageBatch)
        loss, out, mask = model(imageBatch.float(), mask_ratio = 0.7)

        pbarMessage = " ".join(["iter:",
                                str(n_iter),
                                "loss:",
                                str(round(loss.item(), 4))])
        pbar.set_description(pbarMessage)
                        
        loss.backward()
        optimizer.step()    

    out = model.unpatchify(out)
    out = torch.einsum('nchw->nhwc', out).detach().cpu()
    # out = out.detach().cpu()

    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 * model.num_chans)
    mask = model.unpatchify(mask)
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

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

    fig.savefig('finalPred.png')


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
