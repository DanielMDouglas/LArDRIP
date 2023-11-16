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
        self.infileList = inputFiles

        self.fileOrder = np.random.choice(len(self.infileList),
                                          size = len(self.infileList),
                                          replace = False)
        # voxel coordinates are in cm,
        # in 3.8 mm (pixel pitch) voxels
        self.pitch = 0.38 

    def load_next_file(self, fileInd):
        thisInputFile = self.infileList[fileInd]
        self.voxels = h5py.File(thisInputFile)['track_voxels'][:]

        self.eventIDs = np.unique(self.voxels['eventID'])
        self.n_entries = len(self.eventIDs)

        # sequential sampling for testing
        # sampleOrder = np.arange(self.n_entries)
        self.sampleOrder = np.random.choice(self.n_entries,
                                            size = self.n_entries,
                                            replace = False)

    def __getitem__(self, eventIndex):

        eventMask = self.voxels['eventID'] == self.eventIDs[eventIndex]
        eventVoxels = self.voxels[eventMask]

        np_voxels = np.array([np.cast[int](eventVoxels['xBin']/self.pitch),
                              np.cast[int](eventVoxels['yBin']/self.pitch),
                              np.cast[int](eventVoxels['zBin']/self.pitch)]).T
        np_data = np.array([eventVoxels['dE']]).T

        return np_voxels, np_data

    def __iter__(self):
        # more fancy sampler to come...
        for i in self.fileOrder:
            self.load_next_file(i)

            # add augmentation?
            # reflect x, y, z
            # swap x <-> z
            for idx in self.sampleOrder:
                yield self[idx]

class imagePrepper:
    def __init__(self, rawDataset, imageSize = 1024):
        self.rawDataset = rawDataset

        self.imageSize = imageSize
                
    def imageSelector(self, coo, feat):
        # pick an image volume within this event that has
        # some filled voxels

        # find an interesting region within an event window
        # do this using a very loose (~ the image size) DBSCAN
        # take the biggest cluster and center the image on that
        # clustering = DBSCAN(eps = self.imageSize[0],
        #                     min_samples = 1).fit(vox)
        # fig = plt.figure()
        
        clustering = DBSCAN(eps = 3,
                            min_samples = 1).fit(coo)

        # ax = fig.add_subplot(131, projection = '3d')
        # ax.scatter(*coo.T, c = clustering.labels_)

        maxHits = 0
        for clusterLabel in np.unique(clustering.labels_):
            nHitsPerCluster = sum(clustering.labels_ == clusterLabel)
            if nHitsPerCluster > maxHits:
                maxHits = nHitsPerCluster
                largestCluster = clusterLabel

        clusterMask = clustering.labels_ == largestCluster
        clusterCoo = coo[clusterMask]
        clusterFeat = feat[clusterMask]

        # print (np.unique(clustering.labels_, return_counts = True))
        # print ("in cluster", len(clusterFeat))

        clusterInd, clusterCount = np.unique(clustering.labels_, return_counts = True)
        smallClusters = clusterInd[clusterCount <= 10]
        
        # smallClusterMask = clustering.labels_ in smallClusters
        smallClusterMask = np.array([thisLabel not in smallClusters
                                     for thisLabel in clustering.labels_])

        # print (smallClusters)
        # print (coo.shape)
        # print (smallClusterMask.shape)
        # print (smallClusterMask)

        coo = coo[smallClusterMask]
        feat = feat[smallClusterMask]

        if len(coo) == 0:
            raise ValueError ("what the fuck")
        
        # ax = fig.add_subplot(132, projection = '3d')
        # ax.scatter(*clusterCoo.T, c = clusterFeat[:,0])

        CoM = np.average(clusterCoo,
                         weights = clusterFeat[:,0],
                         axis = 0)

        CoMdist = np.sqrt(np.sum(np.power(coo - CoM, 2), axis = -1))

        # maxDist = np.quantile(CoMdist, frac, method = 'inverted_cdf')

        # this method tends to grab too many points
        # adjust the number of points afterwards by
        # choosing a point to leave out
        nKept = self.imageSize
        frac = nKept/len(CoMdist)
        maxDist = np.quantile(CoMdist, frac)
        mask = CoMdist <= maxDist

        cooSelected = coo[mask]
        featSelected = feat[mask]

        if len(featSelected) > self.imageSize:
            excess = len(featSelected) - self.imageSize
            cooSelected = cooSelected[:-excess]
            featSelected = featSelected[:-excess]
        
        # ax = fig.add_subplot(133, projection = '3d')
        # ax.scatter(*coo[mask].T, c = feat[mask,0])
        # plt.show()

        return cooSelected, featSelected
    
    def __getitem__(self, idx):
        vox, data = self.rawDataset[idx]
        vox, data = self.imageSelector(vox, data)
        
        return vox, data
        
    def __iter__(self):
        for vox, data in self.rawDataset:
            try:
                vox, data = self.imageSelector(vox, data)
                yield vox, data
            except ValueError:
                continue

image_dtype = np.dtype([("imageInd", "u4"),
                        ("voxx", "i4"),
                        ("voxy", "i4"),
                        ("voxz", "i4"),
                        ("voxq", "f4"),
                        ])
            
def main(args):
    if '.root' in args.inputFile[0]:
        d = dataset_larcv(args.inputFile)
    elif '.h5' in args.inputFile[0]:
        d = dataset_voxedep(args.inputFile)
    else:
        raise TypeError ("not a valid input filetype!")

    ip = imagePrepper(d)

    print ("recording patch data...")
    croppedData = np.empty((0,), dtype = image_dtype)
    for imageInd, image in tqdm(enumerate(ip)):
        # print (imageInd)
        coo, feat = image
        thisImage = np.empty((len(feat),), dtype = image_dtype)
        thisImage["imageInd"][:] = imageInd
        thisImage['voxx'][:] = coo[:,0]
        thisImage['voxy'][:] = coo[:,1]
        thisImage['voxz'][:] = coo[:,2]
        thisImage['voxq'][:] = feat[:, 0]

        croppedData = np.concatenate((croppedData, thisImage))
        
    with h5py.File(args.preppedOutput, 'a') as f:
        f['images'] = croppedData
            
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
