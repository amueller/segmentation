import numpy as np
#import os
#from glob import glob

from sklearn.externals.joblib import Memory, Parallel, delayed
from scipy.misc import imread
#import matplotlib.pyplot as plt
#from skimage.segmentation import mark_boundaries

from datasets.nyu import NYUSegmentation

from slic_python import slic_n
from latent_crf_experiments.utils import (DataBunchNoSP, DataBunch, gt_in_sp,
                                          probabilities_on_sp)

memory = Memory(cachedir="/home/data/amueller/cache")


def get_probabilities(file_name, path):
    probabilities = []
    for label in xrange(1, 5):
        f = ("%s/prediction_all/%s_lab_image_label_%d.png"
             % (path, file_name, label))
        probabilities.append(imread(f)[:, :, 0])
    probabilities = np.dstack(probabilities).astype(np.float)
    return probabilities / 255.


def load_single_file(dataset, file_name, n_sp=300, add_covariance=False):
    print(file_name)
    image = dataset.get_image(file_name)
    sp = slic_n(image, n_superpixels=n_sp, compactness=10)
    gt = gt_in_sp(dataset, file_name, sp)
    probs = get_probabilities(file_name, dataset.directory)
    probs_sp = probabilities_on_sp(dataset, probs, sp,
                                   add_covariance=add_covariance)
    return probs_sp, gt, sp


@memory.cache
def load_nyu(ds='train', n_sp=300, add_covariance=False):
    dataset = NYUSegmentation()
    file_names = dataset.get_split(ds)
    # load image to generate superpixels
    result = Parallel(n_jobs=-1)(delayed(load_single_file)(dataset, f, n_sp, add_covariance)
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
