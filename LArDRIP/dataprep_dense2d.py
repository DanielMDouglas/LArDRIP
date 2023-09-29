import numpy as np
import h5py
from tqdm import tqdm

from sklearn.cluster import DBSCAN

from utils import *

class dataset_larcv:
    def __init__(self, inputFiles):
        from larcv import larcv
        from ROOT import TChain

        self.chain = TChain("sparse3d_packets_tree")
        for inputFile in inputFiles:
            self.chain.AddFile(inputFile)

        self.n_entries = self.chain.GetEntries()

    def __getitem__(self, eventIndex):

        self.chain.GetEntry(eventIndex)
        
        sparse_event = getattr(self.chain, 'sparse3d_packets_branch')

        num_point = sparse_event.as_vector().size()

        np_voxels = np.empty(shape=(num_point, 3), dtype = np.int32)
        larcv.fill_3d_voxels(sparse_event, np_voxels)

        np_data = np.empty(shape=(num_point, 1), dtype = np.float32)
        larcv.fill_3d_pcloud(sparse_event, np_data)

        return np_voxels, np_data

    def __iter__(self):
        # more fancy sampler to come...
        # sequential sampling for testing
        # sampleOrder = np.arange(self.n_entries)
        sampleOrder = np.random.choice(self.n_entries,
                                       size = self.n_entries,
                                       replace = False)
        # add augmentation?
        # reflect x, y, z
        # swap x <-> z
        for idx in sampleOrder:
            yield self[idx]

class dataset_voxedep:
    def __init__(self, inputFiles):
        self.files = inputFiles

        self.n_files = len(inputFiles)

        # voxel coordinates are in cm,
        # in 3.8 mm (pixel pitch) voxels
        self.pitch = 0.38 

    def load_next_file(self, fileInd):
        self.voxels = h5py.File(self.files[fileInd])['track_voxels'][:]

        self.eventIDs = np.unique(self.voxels['eventID'])
        self.n_entries = len(self.eventIDs)

    def __getitem__(self, eventIndex):

        eventMask = self.voxels['eventID'] == self.eventIDs[eventIndex]
        eventVoxels = self.voxels[eventMask]

        np_voxels = np.array([np.cast[int](eventVoxels['xBin']/self.pitch),
                              np.cast[int](eventVoxels['yBin']/self.pitch)]).T
        np_data = np.array([eventVoxels['dE']]).T

        return np_voxels, np_data

    def __iter__(self):
        # more fancy sampler to come...
        # add augmentation?
        # reflect x, y, z
        # swap x <-> z

        fileLoadOrder = np.arange(self.n_files)
        for fileIDX in fileLoadOrder:
            self.load_next_file(fileIDX)
            # sequential sampling for testing 
            # (if sampling is done again in dataloader, this is fine)
            sampleOrder = np.arange(self.n_entries)
            # sampleOrder = np.random.choice(self.n_entries,
            #                                size = self.n_entries,
            #                                replace = False)
            for idx in sampleOrder:
                yield self[idx]

class imagePrepper:
    def __init__(self, rawDataset):
        self.rawDataset = rawDataset

        # corners of "module" boundaries
        # These are the regions without any real dead space
        # chosen to be a few voxels inboard of the actual boundary
        
        # these are just contiguous 100x200x100 voxel spaces
        # NOTE: not used currently
        self.moduleBounds = [[[25, 125], [50, 250], [25, 125]],
                             [[25, 125], [50, 250], [175, 275]],
                             [[175, 275], [50, 250], [25, 125]],
                             [[175, 275], [50, 250], [175, 275]],
        ]

        self.imageSize = [56, 56]
        
        # how to chunk up the module volume
        # number of patches per image edge
        
    def imageSelector(self, vox, data):
        # pick an image volume within this event that has
        # some filled voxels

        # find an interesting region within an event window
        # do this using a very loose (~ the image size) DBSCAN
        # take the biggest cluster and center the image on that
        # clustering = DBSCAN(eps = self.imageSize[0],
        #                     min_samples = 1).fit(vox)
        clustering = DBSCAN(eps = 3,
                            min_samples = 1).fit(vox)
        maxHits = 0
        for clusterLabel in np.unique(clustering.labels_):
            nHitsPerCluster = sum(clustering.labels_ == clusterLabel)
            if nHitsPerCluster > maxHits:
                maxHits = nHitsPerCluster
                largestCluster = clusterLabel

        clusterMask = clustering.labels_ == largestCluster
        clusterVox = vox[clusterMask]
        clusterData = data[clusterMask]
        
        CoM = np.average(clusterVox,
                         weights = clusterData[:,0],
                         axis = 0)
        # TODO: if the image bounds overlap with the module bounds
        # (i.e. real dead space is included)
        # shift the image so that it doesn't overlap
        imageBounds = [[int(CoM[0] - self.imageSize[0]/2), int(CoM[0] + self.imageSize[0]/2)],
                       [int(CoM[1] - self.imageSize[1]/2), int(CoM[1] + self.imageSize[1]/2)],
                       ]

        vox, data = select_within_box(vox, data, imageBounds)
        # subtract the coordinates of the image window
        # so that voxel indices are relative to the image
        vox -= np.array([imageBounds[0][0],
                         imageBounds[1][0],
                         ])
        return vox, data

    def densifier(self, vox, data):
        dense_image = np.zeros(shape = self.imageSize)
        dense_image = np.expand_dims(dense_image, -1)
        for vox_i, data_i in zip(vox, data):
            dense_image[vox_i[0], vox_i[1], 0] += data_i
        
        return dense_image
    
    def __getitem__(self, idx):
        vox, data = self.rawDataset[idx]
        vox, data = self.imageSelector(vox, data)
        img = self.densifier(vox, data)
        
        return img
        
    def __iter__(self):
        for vox, data in self.rawDataset:
            try:
                vox, data = self.imageSelector(vox, data)
                if len(data) > 5:
                    img = self.densifier(vox, data)
                    yield img
            except ValueError:
                continue

def main(args):
    if '.root' in args.inputFile[0]:
        d = dataset_larcv(args.inputFile)
    elif '.h5' in args.inputFile[0]:
        d = dataset_voxedep(args.inputFile)
    else:
        raise TypeError ("not a valid input filetype!")

    ip = imagePrepper(d)
    
    print ("recording dense image data...")
    imageData = np.empty((0,
                          ip.imageSize[0],
                          ip.imageSize[1],
                          1))
    for imageInd, image in tqdm(enumerate(ip)):
        image = np.expand_dims(image, 0)
        imageData = np.concatenate((imageData, image))

    with h5py.File(args.preppedOutput, 'a') as f:
        f['images'] = imageData
        
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser("Make training images from protoDUNE-ND 2x2 simulation")
    parser.add_argument("inputFile",
                        nargs = '+',
                        help = "input 2x2 larcv or voxelized edep-sim file")
    parser.add_argument("preppedOutput",
                        help = "output patched images [hdf5]")

    args = parser.parse_args()
    main(args)
