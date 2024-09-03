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
    def __init__(self, inputFiles, pitch):
        self.infileList = inputFiles

        self.fileOrder = np.random.choice(len(self.infileList),
                                          size = len(self.infileList),
                                          replace = False)

        self.pitch = pitch

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
        np_data = np.array([eventVoxels['dE'],
                            eventVoxels['PID'],
                            eventVoxels['t0']]).T

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
    def __init__(self, rawDataset, image_size = None, mode = 'hits'):
        self.rawDataset = rawDataset
        
        # self.imageSize = [50, 50, 50]

        # valid modes are:
        # 'hits' -- fixed number of hits per image, set by image_size
        # 'voxels' -- fixed voxels per image edge, set by image_size
        self.mode = mode
            
        self.minHitsPerCluster = 20
        self.minHitsPerImage = 512
        
        self.image_size = image_size

    def imageSelector(self, vox, data):
        # pick an image volume within this event that has
        # some filled voxels

        # find an interesting region within an event window
        # do this using a very loose (~ the image size) DBSCAN
        # take the biggest cluster and center the image on that
        # clustering = DBSCAN(eps = self.imageSize[0],
        #                     min_samples = 1).fit(vox)
        
        # find clusters within the image
        clustering = DBSCAN(eps = 3,
                            min_samples = 1).fit(vox)
        
        maxHits = 0
        
        minHits = self.minHitsPerCluster
        smallClusters = []

        # find the cluster with the greatest number of hits
        for clusterLabel in np.unique(clustering.labels_):
            nHitsPerCluster = sum(clustering.labels_ == clusterLabel)
            if nHitsPerCluster > maxHits:
                maxHits = nHitsPerCluster
                largestCluster = clusterLabel
            # track small clusters
            if nHitsPerCluster < minHits:
                smallClusters.append(clusterLabel)

        # get the largest cluster in the event
        clusterMask = clustering.labels_ == largestCluster
        clusterVox = vox[clusterMask]
        clusterData = data[clusterMask]

        # find the center of the largest cluster
        CoM = np.average(clusterVox,
                         weights = clusterData[:,0],
                         axis = 0)

        # remove small clusters (< 10 voxels)
        clusterSizeMask = np.array([thisLabel not in smallClusters
                                    for thisLabel in clustering.labels_])
        # print ("small cluster mask", clusterSizeMask)

        # all clusters in the image, except for the very small ones
        clusterVox = vox[clusterSizeMask]
        clusterData = data[clusterSizeMask]

        # select the N voxels closest to the image center
        displacement_to_com = clusterVox - CoM
        distance_to_com = np.sum(np.power(displacement_to_com, 2), axis = -1)

        # each prepped image should have at least (default 512) hits
        assert len(clusterData) > self.minHitsPerImage

        # this mode will keep a fixed number of hits per image
        if self.mode == 'hits':
            assert self.image_size < len(clusterData)
            fracKept = (self.image_size + 1)/len(clusterData)
            maxDist = np.quantile(distance_to_com, fracKept)

            closestMask = distance_to_com <= maxDist

            clusterVox = clusterVox[closestMask]
            clusterData = clusterData[closestMask]
        
            if len(clusterData) > self.image_size:
                choice = np.random.choice(len(clusterData), 
                                          size = self.image_size,
                                          replace = False)
                clusterVox = clusterVox[choice]
                clusterData = clusterData[choice]
            assert len(clusterData) == self.image_size
            
            # shift the image so that the minimum value is (0, 0, 0)
            # can also shift so that the earliest hit is at (0, 0, 0)
            clusterVox -= np.array([np.min(clusterVox[:,0]),
                                    np.min(clusterVox[:,1]),
                                    np.min(clusterVox[:,2]),
            ])

        # this mode will produce an image with a given size
        elif self.mode == 'voxels':
            xMin = CoM[0] - self.image_size/2
            xMax = CoM[0] + self.image_size/2

            yMin = CoM[1] - self.image_size/2
            yMax = CoM[1] + self.image_size/2

            zMin = CoM[2] - self.image_size/2
            zMax = CoM[2] + self.image_size/2

            # windowMask = (clusterVox[:,0] > xMin)
            # windowMask *= (clusterVox[:,0] <= xMax)
            # windowMask *= (clusterVox[:,1] > yMin)
            # windowMask *= (clusterVox[:,1] <= yMax)
            # windowMask *= (clusterVox[:,2] > zMin)
            # windowMask *= (clusterVox[:,2] <= zMax)

            # print (np.sum(windowMask), windowMask.shape)

            windowMask = (vox[:,0] >= xMin)
            windowMask *= (vox[:,0] < xMax)
            windowMask *= (vox[:,1] >= yMin)
            windowMask *= (vox[:,1] < yMax)
            windowMask *= (vox[:,2] >= zMin)
            windowMask *= (vox[:,2] < zMax)

            clusterVox = vox[windowMask]
            clusterData = data[windowMask]

            # shift the image so that the minimum value is (0, 0, 0)
            clusterVox -= np.array([int(xMin+1),
                                    int(yMin+1),
                                    int(zMin+1),
            ])

            # print (np.min(clusterVox[:,0]),
            #        np.max(clusterVox[:,0]),
            #        )

        return clusterVox, clusterData
    
    def __getitem__(self, idx):
        vox, data = self.rawDataset[idx]
        vox, data = self.imageSelector(vox, data)
        
        return vox, data
        
    def __iter__(self):
        for vox, data in self.rawDataset:
            try:
                vox, data = self.imageSelector(vox, data)
                yield vox, data
            except AssertionError:
                continue

image_dtype = np.dtype([("imageInd", "u4"),
                        ("voxx", "i4"),
                        ("voxy", "i4"),
                        ("voxz", "i4"),
                        ("voxdE", "f4"),
                        ("voxPID", "u4"),
                        ("voxt0", "f4"),
                    ])
            
def main(args):
    if '.root' in args.inputFile[0]:
        d = dataset_larcv(args.inputFile)
    elif '.h5' in args.inputFile[0]:
        d = dataset_voxedep(args.inputFile, args.voxel_pitch)
    else:
        raise TypeError ("not a valid input filetype!")

    ip = imagePrepper(d, args.image_size, mode = 'voxels')

    print ("recording image data...")
    imageData = np.empty((0,), dtype = image_dtype)
    for imageInd, (imageVox, imageFeats) in tqdm(enumerate(ip)):
        thisImage = np.empty((len(imageFeats),), dtype = image_dtype)

        thisImage["imageInd"][:] = imageInd
        thisImage["voxx"][:] = imageVox[:,0]
        thisImage["voxy"][:] = imageVox[:,1]
        thisImage["voxz"][:] = imageVox[:,2]
        thisImage["voxdE"][:] = imageFeats[:,0]
        thisImage["voxPID"][:] = imageFeats[:,1]
        thisImage["voxt0"][:] = imageFeats[:,2]
        imageData = np.concatenate((imageData, thisImage))

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
    parser.add_argument("-s", "--image_size",
                        type = int,
                        help = "enforce a strict number of hits per image",
                        default = None)
    parser.add_argument("-p", "--voxel_pitch",
                        type = float,
                        help = "voxel pitch in cm (default: 0.1)",
                        default = 0.1)

    args = parser.parse_args()
    main(args)
