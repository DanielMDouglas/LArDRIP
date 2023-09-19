import h5py
import numpy as np

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from LArDRIP.utils import *

class MAEdataloader:
    def __init__(self, patched_image_input, maskFraction = 0.5, batchSize = 1):
        with h5py.File(patched_image_input, 'r') as f:
            self.patchSizes = f['patchSize'][:]
            self.patchBounds = f['patchBounds'][:]
            self.patchData = f['patchedData'][:]

        imageIndices = np.unique(self.patchData['imageInd'])

        self.imageLoadOrder = np.random.choice(len(imageIndices),
                                               size = len(imageIndices),
                                               replace = False)

        # sequential load order for testing
        # self.imageLoadOrder = np.arange(len(imageIndices))
                                       
        self.maskFraction = maskFraction
        self.batchSize = batchSize
            
    def masker(self, patches):
        # take a list of jumbled patches and select a fraction
        # of them
        # return the unmasked patches and their coordinate key
        # and return the coordinate keys of the masked patches

        # return unmaskedPatches, unmaskedKeys, maskedKeys

        patchIndices = np.unique(patches['patchInd'])
        nPatches = len(patchIndices)
        nKept = int((1 - self.maskFraction)*nPatches)
        
        patchChoice = np.random.choice(patchIndices,
                                       size = nKept,
                                       replace = False)

        patchMask = np.array([thisPatch['patchInd'] in patchChoice
                              for thisPatch in patches])
        keptPatches = patches[patchMask]
        maskedPatches = patches[~patchMask]

        return keptPatches, maskedPatches
    
    def load_image(self, idx):
        imageIndex = self.imageLoadOrder[idx]
        imagePatchMask = self.patchData['imageInd'] == imageIndex 

        imagePatches = self.patchData[imagePatchMask]

        return imagePatches

    def __getitem__(self, idx):
        imagePatches = self.load_image(idx)
        # densePatches = densify(imagePatches, self.patchSizes)
        ravelledPatches = ravel_patches(densePatches, self.patchBounds)
        maskedPatches = self.masker(imagePatches)

        return maskedPatches
        
    def __iter__(self):
        patchBatch = []
        for imageIndex in self.imageLoadOrder:
            imagePatchMask = self.patchData['imageInd'] == imageIndex 

            imagePatches = self.patchData[imagePatchMask]
            # densePatches = densify(imagePatches, self.patchSizes)
            # ravelledPatches = ravel_patches(densePatches, self.patchBounds)
            maskedPatches = self.masker(ravelledPatches)

            patchBatch.append(maskedPatches)

            if len(patchBatch) == self.batchSize:
                yield patchBatch
                patchBatch = []
                
if __name__ == '__main__':
    # as an example, this is how you can initialize and
    # get batches from the data loader
    # each batch is a list of masked + patched images

    # the first element is a list of kept patches and
    # the second element is a list of the masked ones

    # each patch is defined in a sparse representation
    # the elements are
    # (imageInd, patchInd, voxel_x, voxel_y, voxel_z, voxel_value)
    # imageInd is largely useless (just for internal tracking)
    # patchInd is the static positional encoding of a patch within
    # an image (defined in the patchBounds)

    dl = MAEdataloader('../data/example/patched_voxEdep.h5')

    patches = dl.load_image(2)
    patches = densify(patches, dl.patchBounds, dl.patchSizes)
    patches = ravel_patches(patches, dl.patchBounds)
    kept, masked = dl.masker(patches)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    draw_patched_image(ax, kept, dl.patchBounds,
                       color = 'red', label = 'kept',
                       unravelled = False)
    draw_patched_image(ax, masked, dl.patchBounds,
                       color = 'blue', label = 'masked',
                       unravelled = False)
    for thisPatchBound in dl.patchBounds:
        draw_rect_bounds(ax,
                         [[thisPatchBound['xmin'], thisPatchBound['xmax']],
                          [thisPatchBound['ymin'], thisPatchBound['ymax']]],
                         color = 'green'
                         )
    fig.savefig('imView.png')
