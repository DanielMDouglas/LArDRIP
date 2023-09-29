import torch
import torch.nn as nn
import torch.optim as optim
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from functools import partial

import sys
sys.path.append('./mae')
import models_mae

class MAEViT_network (models_mae.MaskedAutoencoderViT):
    def __init__(self, manifest):
        super().__init__(in_chans = 1, norm_layer=partial(nn.LayerNorm, eps=1.e-6),
                         **manifest['config'])
        self.manifest = manifest

        # self.model = models_mae.MaskedAutoencoderViT(img_size=56, patch_size=8, in_chans=1,
        #                                              embed_dim=512, depth=12, 
        #                                              decoder_embed_dim=256, decoder_depth=4).to(device)

    def load_checkpoint(self, checkpointFile):
        checkpoint = torch.load(checkpointFile, map_location = device)
        self.load_state_dict(checkpoint['model'], strict = False)

    def save_checkpoint(self, checkpointFile):
        torch.save(dict(model = self.state_dict()), checkpointFile)
