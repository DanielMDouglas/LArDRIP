![alt text](https://github.com/DanielMDouglas/LArDRIP/blob/main/logo.png?raw=true)

# LArDRIP - Liquid Argon Dead Region Inference Project

This package is a work-in-progress for generating inferred signals in dead regions of a DUNE-ND-like liquid argon time projection chamber (LArTPC).  Right now, the targeted architecture for this model is a [masked autoencoder](https://arxiv.org/abs/2111.06377), which is to be adapted to sparse 3D images.

The framework is being developed with the DUNE-ND 2x2 prototype in mind, and development is utilizing existing 2x2 simulation starting from the `larnd2supera` stage.  Small subrun samples can be found on SLAC's SDF computing system in `/sdf/group/neutrino/cyifan/larnd2supera/larcv_output/output_00679-larcv.root` (thank you to Yifan and others for providing these samples!)

## Data Preparation

To pre-process the `larnd2supera` simulation into patched images, the `dataprep.py` script is provided.  This script should be called like

```
python dataprep.py [-h] inputRoot [inputRoot ...] preppedOutput
```

This will iterate through 2x2 images, find small images (30x30x30 voxels, this is configurable), and then apply a patching scheme (6x6x6 patches per image, so that each patch is 5x5x5 voxels, this is also subject to optimization in the future).  The resulting patches are saved in a sparse representation along with a record of the patching scheme to an `hdf5` file for faster reading by the train-time data loader.

## Data Loader

A simple data loader is defined in `dataloader.py` which will read from this prepped hdf5 file and apply a run-time mask to the patches.  The probability to keep or mask a given patch is given as an argument to the dataloader at initialization.  A very simple example of iterating through batches with this dataloader is included with the definition.  Constructing sparse tensors from these sparse patches is still an open question, but a utility function is included.
