import numpy as np
import h5py
from tqdm import tqdm

from sklearn.cluster import DBSCAN

from larcv import larcv
from ROOT import TChain

def select_within_box(vox, data, bounds):
    keptVoxelMask = vox[:, 0] >= bounds[0][0]
    keptVoxelMask = np.logical_and(keptVoxelMask, vox[:, 0] < bounds[0][1])
    keptVoxelMask = np.logical_and(keptVoxelMask, vox[:, 1] >= bounds[1][0])
    keptVoxelMask = np.logical_and(keptVoxelMask, vox[:, 1] < bounds[1][1])
    keptVoxelMask = np.logical_and(keptVoxelMask, vox[:, 2] >= bounds[2][0])
    keptVoxelMask = np.logical_and(keptVoxelMask, vox[:, 2] < bounds[2][1])

    maskedVox = vox[keptVoxelMask]
    maskedData = data[keptVoxelMask]

    return maskedVox, maskedData
    
class dataset:
    def __init__(self, inputFiles):
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

        self.imageSize = [30, 30, 30]
        
        # how to chunk up the module volume
        self.patchScheme = [6,
                            6,
                            6,
        ]
        
        xPatchEdges = np.linspace(0, self.imageSize[0],
                                  self.patchScheme[0] + 1,
                                  dtype = np.int32)
        yPatchEdges = np.linspace(0, self.imageSize[1],
                                  self.patchScheme[1] + 1,
                                  dtype = np.int32)
        zPatchEdges = np.linspace(0, self.imageSize[2],
                                  self.patchScheme[2] + 1,
                                  dtype = np.int32)

        self.patchBounds = [[[xPatchEdges[xInd], xPatchEdges[xInd+1]],
                             [yPatchEdges[yInd], yPatchEdges[yInd+1]],
                             [zPatchEdges[zInd], zPatchEdges[zInd+1]]]
                            for xInd in range(self.patchScheme[0])
                            for yInd in range(self.patchScheme[1])
                            for zInd in range(self.patchScheme[2])]
        
    def imageSelector(self, vox, data):
        # pick an image volume within this event that has
        # some filled voxels

        # find an interesting region within an event window
        # do this using a very loose (~ the image size) DBSCAN
        # take the biggest cluster and center the image on that
        clustering = DBSCAN(eps = self.imageSize[0],
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
                       [int(CoM[2] - self.imageSize[2]/2), int(CoM[2] + self.imageSize[2]/2)],
                       ]

        vox, data = select_within_box(vox, data, imageBounds)
        # subtract the coordinates of the image window
        # so that voxel indices are relative to the image
        vox -= np.array([imageBounds[0][0],
                         imageBounds[1][0],
                         imageBounds[2][0],
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
                                   thisPatchBound[2][0],
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

                yield patches
            except ValueError:
                continue

patchBounds_dtype = np.dtype([("patchInd", "u4"),
                              ("xmin", "i4"),
                              ("xmax", "i4"),
                              ("ymin", "i4"),
                              ("ymax", "i4"),
                              ("zmin", "i4"),
                              ("zmax", "i4"),
                              ])

imagePatch_dtype = np.dtype([("imageInd", "u4"),
                             ("patchInd", "u4"),
                             ("voxx", "i4"),
                             ("voxy", "i4"),
                             ("voxz", "i4"),
                             ("voxq", "f4"),
                             ])
            
def main(args):
    d = dataset(args.inputRoot)

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
        theseBounds["zmin"] = patchBound[2][0]
        theseBounds["zmax"] = patchBound[2][1]
        patchBounds = np.concatenate((patchBounds, theseBounds))

    print ("recording patch data...")
    patchedData = np.empty((0,), dtype = imagePatch_dtype)
    for imageInd, patchedImage in tqdm(enumerate(pp)):
        thisPatch = np.empty((len(patchedImage),), dtype = imagePatch_dtype)
        thisPatch["imageInd"][:] = imageInd
        for i, thisPatchData in enumerate(patchedImage):
            thisPatch["patchInd"][i] = thisPatchData[0]
            for vox, data in zip(thisPatchData[1], thisPatchData[2]):
                thisPatch["voxx"][i] = vox[0]
                thisPatch["voxy"][i] = vox[1]
                thisPatch["voxz"][i] = vox[2]
                thisPatch["voxq"][i] = data[0]
        patchedData = np.concatenate((patchedData, thisPatch))

    with h5py.File(args.preppedOutput, 'a') as f:
        f['patchBounds'] = patchBounds
        f['patchedData'] = patchedData
            
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser("Make training images from protoDUNE-ND 2x2 simulation")
    parser.add_argument("inputRoot",
                        nargs = '+',
                        help = "input 2x2 larcv file")
    parser.add_argument("preppedOutput",
                        help = "output patched images [hdf5]")

    args = parser.parse_args()
    main(args)
