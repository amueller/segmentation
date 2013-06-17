from glob import glob

from scipy.misc import imread
#import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries

from slic_python import slic_n

data_path = "/home/data/amueller/nyu_depth_forest/fold4/"


def load_nyu(ds='train'):
    if ds not in ['train', 'val']:
        raise ValueError("ds should be 'train' or 'val', got %s." % ds)
    # load image to generate superpixels
    if ds == 'train':
        image_path = data_path + "training/"
    else:
        image_path = data_path + "validation/"
    for image_file in glob(image_path + "*_lab_image.png")[:10]:
        print(image_file)
        image = imread(image_file)
        segments = slic_n(image, n_superpixels=100, compactness=10)
        boundary_image = mark_boundaries(image, segments)
        plt.figure()
        plt.imshow(boundary_image)
    plt.show()


def nyu_superpixels():
    pass
