# Code and data setup

## Requirements
- [Kaolin](https://github.com/NVIDIAGameWorks/kaolin) (tested on commit [e7e5131](https://github.com/NVIDIAGameWorks/kaolin/tree/e7e513173bd4159ae45be6b3e156a3ad156a3eb9))
- Python >= 3.6
- PyTorch >= 1.2
- CUDA >= 10.0 (you won't be able to build kaolin with CUDA 9)

To run the code, you also need to install the following packages (you can easily do so via pip): `packaging`, `nltk` (for models conditioned on captions), and `tensorboard` (if you want to use this feature).

Note that, although Kaolin only officially supports PyTorch versions between 1.2 and 1.4, our code dynamically patches some functions to make them work with newer PyTorch versions. Currently, inference code has been successfully tested with PyTorch 1.6.


## Minimal setup (evaluating pretrained models)
This step involves setting up the pretrained models, precomputed statistics for FID evaluation, and precomputed pose metadata. No dataset setup is involved.
With this setup, you will be able to evaluate our pretrained models (FID scores and mesh export), but you will not be able to train a new model from scratch. 

You can download the pretrained models and cache directory from the [Releases](https://github.com/dariopavllo/convmesh/releases) section of this repository. It suffices to extract the archives to the root directory of this repo.

## Full dataset setup (training from scratch)
If you have not already done so, set up the precomputed metadata as described in the step above.

Then, to set up the CUB dataset, download [CUB images](http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz) and [segmentations](http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/segmentations.tgz) ([source](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)) and extract them so that your directory tree looks like this:
```
datasets/cub/CUB_200_2011/
datasets/cub/CUB_200_2011/segmentations/
datasets/cub/data/
datasets/cub/sfm/
```
Creating symbolic links is also a good idea if you have a copy of the dataset somewhere else.

For Pascal3D+ Cars, download [PASCAL3D+_release1.1.zip](ftp://cs.stanford.edu/cs/cvgl/PASCAL3D+_release1.1.zip) ([source](https://cvgl.stanford.edu/projects/pascal3d.html)) and set up your directory tree like this:
```
datasets/p3d/PASCAL3D+_release1.1/
datasets/p3d/data/
datasets/p3d/sfm/
datasets/p3d/p3d_labels.csv
```