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
        # just read one file for now.  FIX THIS!
        self.voxels = h5py.File(inputFiles[0])['track_voxels'][:]

        self.eventIDs = np.unique(self.voxels['eventID'])
        self.n_entries = len(self.eventIDs)

        # voxel coordinates are in cm,
        # in 3.8 mm (pixel pitch) voxels
        self.pitch = 0.38 

    def __getitem__(self, eventIndex):

        eventMask = self.voxels['eventID'] == self.eventIDs[eventIndex]
        eventVoxels = self.voxels[eventMask]

        np_voxels = np.array([np.cast[int](eventVoxels['xBin']/self.pitch),
                              np.cast[int](eventVoxels['yBin']/self.pitch)]).T
        np_data = np.array([eventVoxels['dE']]).T

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

class patchPrepper:
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

        self.imageSize = [50, 50]
        
        # how to chunk up the module volume
        # number of patches per image edge
        self.patchScheme = [5,
                            5,
        ]

        self.patchSize = [imgSize/nPatches
                          for imgSize, nPatches in zip(self.imageSize, self.patchScheme)
                          ]
        
        xPatchEdges = np.linspace(0, self.imageSize[0],
                                  self.patchScheme[0] + 1,
                                  dtype = np.int32)
        yPatchEdges = np.linspace(0, self.imageSize[1],
                                  self.patchScheme[1] + 1,
                                  dtype = np.int32)

        self.patchBounds = [[[xPatchEdges[xInd], xPatchEdges[xInd+1]],
                             [yPatchEdges[yInd], yPatchEdges[yInd+1]]]
                            for xInd in range(self.patchScheme[0])
                            for yInd in range(self.patchScheme[1])]
        
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

    def patcher(self, vox, data):
        # this function will take the whole sparse
        # array and break it up into patches according
        # to the patcher scheme
        # the order of these patches should be
        # randomized and then the jumbled list is returned

        patches = []
        for patchIndex, thisPatchBound in enumerate(self.patchBounds):
            maskedVox, maskedData = select_within_box(vox, data, thisPatchBound)

            # subtract the patch lower left corner so that
            # the voxel indices are relative to the patch
            maskedVox -= np.array([thisPatchBound[0][0],
                                   thisPatchBound[1][0],
                                   ])
            if len(maskedVox):
                patches.append((patchIndex, maskedVox, maskedData))
            
        return patches
    
    def __getitem__(self, idx):
        vox, data = self.rawDataset[idx]
        vox, data = self.imageSelector(vox, data)
        patches = self.patcher(vox, data)
        
        return patches
        
    def __iter__(self):
        for vox, data in self.rawDataset:
            try:
                vox, data = self.imageSelector(vox, data)
                patches = self.patcher(vox, data)
                if len(patches) > 2:
                    yield patches
            except ValueError:
                continue

patchBounds_dtype = np.dtype([("patchInd", "u4"),
                              ("xmin", "i4"),
                              ("xmax", "i4"),
                              ("ymin", "i4"),
                              ("ymax", "i4"),
                              ])

patchSize_dtype = np.dtype([("nVoxX", "u4"),
                            ("nVoxY", "u4"),
                            ])

imagePatch_dtype = np.dtype([("imageInd", "u4"),
                             ("patchInd", "u4"),
                             ("voxx", "i4"),
                             ("voxy", "i4"),
                             ("voxq", "f4"),
                             ])
            
def main(args):
    if '.root' in args.inputFile[0]:
        d = dataset_larcv(args.inputFile)
    elif '.h5' in args.inputFile[0]:
        d = dataset_voxedep(args.inputFile)
    else:
        raise TypeError ("not a valid input filetype!")

    pp = patchPrepper(d)

    print ("recording patch bounds...")
    patchBounds = np.empty((0,), dtype = patchBounds_dtype)
    for patchInd, patchBound in tqdm(enumerate(pp.patchBounds)):
        theseBounds = np.empty((1,), dtype = patchBounds_dtype)
        theseBounds["patchInd"] = patchInd
        theseBounds["xmin"] = patchBound[0][0]
        theseBounds["xmax"] = patchBound[0][1]
        theseBounds["ymin"] = patchBound[1][0]
        theseBounds["ymax"] = patchBound[1][1]
        patchBounds = np.concatenate((patchBounds, theseBounds))

    print ("recording patch sizes...")
    patchSizes = np.empty((1,), dtype = patchSize_dtype)
    patchSizes["nVoxX"] = pp.patchSize[0]
    patchSizes["nVoxY"] = pp.patchSize[1]
    
    print ("recording patch data...")
    patchedData = np.empty((0,), dtype = imagePatch_dtype)
    for imageInd, patchedImage in tqdm(enumerate(pp)):
        for i, thisPatchData in enumerate(patchedImage):
            thisPatch = np.empty((len(thisPatchData[1]),), dtype = imagePatch_dtype)
            thisPatch["imageInd"][:] = imageInd
            thisPatch["patchInd"][:] = thisPatchData[0]
            thisPatch["voxx"][:] = thisPatchData[1][:,0]
            thisPatch["voxy"][:] = thisPatchData[1][:,1]
            thisPatch["voxq"][:] = thisPatchData[2][0]
            patchedData = np.concatenate((patchedData, thisPatch))

    with h5py.File(args.preppedOutput, 'a') as f:
        f['patchBounds'] = patchBounds
        f['patchedData'] = patchedData
        f['patchSize'] = patchSizes
        
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
