import numpy as np
#import os
#from glob import glob

from sklearn.externals.joblib import Memory
from scipy.misc import imread
#import matplotlib.pyplot as plt
#from skimage.segmentation import mark_boundaries

from datasets.nyu import NYUSegmentation

from slic_python import slic_n
from latent_crf_experiments.utils import (DataBunchNoSP, DataBunch, gt_in_sp,
                                          probabilities_on_sp)

memory = Memory(cachedir="/tmp/cache")


def get_probabilities(file_name, path):
    probabilities = []
    for label in xrange(1, 5):
        f = ("%s/prediction_all/%s_lab_image_label_%d.png"
             % (path, file_name, label))
        probabilities.append(imread(f)[:, :, 0])
    probabilities = np.dstack(probabilities).astype(np.float)
    return probabilities / 255.


@memory.cache
def load_nyu(ds='train'):
    dataset = NYUSegmentation()
    # load image to generate superpixels
    file_names, X, Y, superpixels = [], [], [], []
    for file_name in dataset.get_split(ds):
        print(file_name)
        ## load image to generate superpixels
        image = dataset.load_image(file_name)
        sp = slic_n(image, n_superpixels=100, compactness=10)
        gt = gt_in_sp(dataset, file_name, sp)
        Y.append(gt)
        superpixels.append(sp)
        file_names.append(file_name)
        probs = get_probabilities(file_name, dataset.directory)
        X.append(probabilities_on_sp(dataset, probs, sp))

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
