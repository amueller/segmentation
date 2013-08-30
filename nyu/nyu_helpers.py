import numpy as np
#import os
#from glob import glob

from sklearn.externals.joblib import Memory
from scipy.misc import imread
#import matplotlib.pyplot as plt
#from skimage.segmentation import mark_boundaries

from datasets.nyu import NYUSegmentation

#from slic_python import slic_n
from latent_crf_experiments.utils import DataBunchNoSP

memory = Memory(cachedir="/tmp/cache")


def get_probabilities(file_name, path):
    probabilities = []
    for label in xrange(1, 5):
        f = ("%s/prediction_all/%s_lab_image_label_%d.png"
             % (path, file_name, label))
        probabilities.append(imread(f)[:, :, 0])
    probabilities = np.dstack(probabilities)
    return probabilities


#@memory.cache
def load_nyu(ds='train'):
    if ds not in ['train', 'val']:
        raise ValueError("ds should be 'train' or 'val', got %s." % ds)
    ## load image to generate superpixels
    #if ds == 'train':
        #image_path = data_path + "training/"
    #else:
        #image_path = data_path + "validation/"
    #for image_file in glob(image_path + "*_lab_image.png")[:10]:
        #print(image_file)
        #image = imread(image_file)
        #segments = slic_n(image, n_superpixels=100, compactness=10)
        #boundary_image = mark_boundaries(image, segments)
        #plt.figure()
        #plt.imshow(boundary_image)
    #plt.show()


@memory.cache
def load_nyu_pixelwise(ds='train'):
    dataset = NYUSegmentation()
    if ds not in ['train', 'val']:
        raise ValueError("ds should be 'train' or 'val', got %s." % ds)
    # load image to generate superpixels
    file_names = []
    X = []
    Y = []
    for file_name in dataset.get_split(ds):
        print(file_name)
        file_names.append(file_name)
        gt = dataset.get_ground_truth(file_name)
        prediction = get_probabilities(file_name, dataset.directory)
        Y.append(gt)
        X.append(prediction)
    return DataBunchNoSP(X, Y, file_names)
