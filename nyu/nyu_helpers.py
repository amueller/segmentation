import numpy as np
#import os
#from glob import glob

from sklearn.externals.joblib import Memory, Parallel, delayed
from scipy.misc import imread
#import matplotlib.pyplot as plt

from datasets.nyu import NYUSegmentation

from slic_python import slic_n
from latent_crf_experiments.utils import (DataBunchNoSP, DataBunch, gt_in_sp,
                                          probabilities_on_sp)
from skimage.segmentation import slic
from skimage import morphology

memory = Memory(cachedir="/home/data/amueller/cache")


def get_probabilities(file_name, path):
    probabilities = []
    for label in xrange(1, 5):
        f = ("%s/prediction_all/%s_lab_image_label_%d.png"
             % (path, file_name, label))
        probabilities.append(imread(f)[:, :, 0])
    probabilities = np.dstack(probabilities).astype(np.float)
    return probabilities / 255.


def load_single_file(dataset, file_name, n_sp=300, sp='rgb'):
    print(file_name)
    image = dataset.get_image(file_name)
    if sp == 'rgb':
        sps = slic_n(image, n_superpixels=n_sp, compactness=10)
    elif sp == 'rgb-skimage':
        sps = slic(image, n_segments=n_sp, compactness=10, multichannel=True)
        sps = remove_small_sp(morphology.label(sps))
    elif sp == 'rgbd':
        depth = dataset.get_depth(file_name)
        depth -= depth.min()
        depth /= depth.max()
        rgbd = np.dstack([image / 255., depth])
        sps = slic(rgbd, n_segments=n_sp, compactness=.1, convert2lab=False, multichannel=True)
    else:
        raise ValueError("Expected sp to be 'rgb' or 'rgbd' got %d" % sp)

    gt = gt_in_sp(dataset, file_name, sps)
    probs = get_probabilities(file_name, dataset.directory)
    probs_sp = probabilities_on_sp(dataset, probs, sps)
    return probs_sp, gt, sps


@memory.cache
def load_nyu(ds='train', n_sp=300, sp='rgb'):
    # trigger cache?
    dataset = NYUSegmentation()
    file_names = dataset.get_split(ds)
    # load image to generate superpixels
    result = Parallel(n_jobs=-1)(delayed(load_single_file)(dataset, f, n_sp, sp)
                                 for f in file_names)
    X, Y, superpixels = zip(*result)

    #file_names, X, Y, superpixels = [], [], [], []
    #for file_name in dataset.get_split(ds):
        #sp, gt, probs = load_single_file(dataset, file_name)
        ### load image to generate superpixels
        #Y.append(gt)
        #superpixels.append(sp)
        #file_names.append(file_name)
        #X.append(probs)

    return DataBunch(X, Y, file_names, superpixels)


@memory.cache
def load_nyu_pixelwise(ds='train'):
    dataset = NYUSegmentation()
    # load image to generate superpixels
    file_names, X, Y = [], [], []
    for file_name in dataset.get_split(ds):
        print(file_name)
        file_names.append(file_name)
        gt = dataset.get_ground_truth(file_name)
        prediction = get_probabilities(file_name, dataset.directory)
        Y.append(gt)
        X.append(prediction)
    return DataBunchNoSP(X, Y, file_names)
