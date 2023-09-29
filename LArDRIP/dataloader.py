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

class MAEdataloader_dense2d:
    def __init__(self, image_input, batchSize = 1, sequentialLoad = False):
        with h5py.File(image_input, 'r') as f:
            self.images = f['images'][:]

        self.imageSize = self.images.shape[1:]

        self.batchSize = batchSize

        self.sequentialLoad = sequentialLoad

    def gen_load_order(self):
        imageIndices = np.arange(len(self.images))

        if self.sequentialLoad:
            self.imageLoadOrder = np.arange(len(imageIndices))
        else:
            self.imageLoadOrder = np.random.choice(len(imageIndices),
                                                   size = len(imageIndices),
                                                   replace = False)
        
    def load_image(self, idx):
        imageIndex = self.imageLoadOrder[idx]

        image = self.images[imageIndex]

        return image

    def __getitem__(self, idx):
        image = self.load_image(idx)

        return image

    def __iter__(self):
        imageBatch = torch.empty((0, self.imageSize[0], self.imageSize[1], 1))
        self.gen_load_order()
        for imageIndex in self.imageLoadOrder:
            image = self.load_image(imageIndex)
            image = torch.tensor(np.expand_dims(image, 0))
            imageBatch = torch.cat((imageBatch, image))

            if len(imageBatch) == self.batchSize:
                yield imageBatch
                imageBatch = torch.empty((0, self.imageSize[0], self.imageSize[1], 1))

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

    dl = MAEdataloader_dense2d('../data/example/dense_2d_voxEdep.h5')

    thisImage = dl.load_image(2)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    draw_dense_image(ax, thisImage,
                     )
    fig.savefig('imView.png')
