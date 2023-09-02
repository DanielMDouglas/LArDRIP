import MinkowskiEngine as ME

import torch
import torch.nn as nn
import torch.optim as optim

from .layers import *

import numpy as np
import yaml
import tqdm
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')        

class encoder(ME.MinkowskiNetwork):
    def __init__(self, D):
        super(encoder, self).__init__(D)

        in_feat = 1

        self.n_epoch = 0
        self.n_iter = 0

        self.criterion = nn.MSELoss() # need to define the loss function!!!

        self.layers = [ResNetEncoder(1, depth = 4)]
        self.network = nn.Sequential(*self.layers)

        self.lr = 1.e-4

        self.optimizer = optim.Adam(self.parameters(),
                                    lr = self.lr)
        
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

    def trainLoop(self, dataLoader, dropout = False, nEpochs = 10):
        """
        page through a training file, do forward calculation, evaluate loss, and backpropagate
        """

        self.train()
        
        report = False
        prevRemainder = 0

        for i in tqdm.tqdm(range(nEpochs)):
            pbar = tqdm.tqdm(enumerate(dataLoader.load()),
                             total = dataLoader.batchesPerEpoch)
            for j, (inpt, truth) in pbar:
                self.optimizer.zero_grad()
                    
                output = self.forward(inpt)
                
                loss = self.criterion(output, truth)
 
                pbarMessage = " ".join(["epoch:",
                                        str(self.n_epoch),
                                        "loss:",
                                        str(round(loss.item(), 4))])
                pbar.set_description(pbarMessage)

                loss.backward()
                self.optimizer.step()        

                self.n_iter += 1
                        
            self.n_epoch += 1
            self.n_iter = 0

        print ("final loss:", loss.item())        

    def evalLoop(self, dataLoader, nBatches = 50, evalMode = 'eval'):
        """
        page through a test file, do forward calculation, evaluate loss and accuracy metrics
        do not update the model!
        """

        evalBatches = nBatches
       
        self.eval()
        
        lossList = []
        
        pbar = tqdm.tqdm(enumerate(dataLoader.load()),
                         total = evalBatches)
        for i, (inpt, truth) in pbar:
            if i >= evalBatches:
                break # we're done here

            output = self.forward(inpt)
            loss = self.criterion(output, truth)

            pbar.set_description("loss: "+str(round(loss.item(), 4)))

            lossList.append(loss.item())

        return lossList
