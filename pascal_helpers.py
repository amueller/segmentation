from collections import namedtuple

import numpy as np
from scipy.misc import imread

pascal_path = "/home/local/datasets/VOC2011/TrainVal/VOCdevkit/VOC2011"

# stores information that was COMPUTED from the dataset + file names for
# correspondence
DataBunch = namedtuple('DataBunch', 'X, Y, file_names')


def load_image(filename):
    return imread(pascal_path + "/JPEGImage/%s.jpg" % filename)


def get_ground_truth(filename):
    return imread(pascal_path + "/SegmentationClass/%s.png" % filename)


def load_kraehenbuehl(filename):
    path = "/home/user/amueller/local/voc_potentials_kraehenbuehl/unaries/"
    with open(path + filename + ".unary") as f:
        size = np.fromfile(f, dtype=np.uint32, count=3).byteswap()
        data = np.fromfile(f, dtype=np.float32).byteswap()
        img = data.reshape(size[1], size[0], size[2])
    return img


def load_pascal(which='train', year="2010"):
    if which not in ["train", "val"]:
        raise ValueError("Expected 'which' to be 'train' or 'val', got %s." %
                         which)
    split_file = pascal_path + "/ImageSets/Segmentation/%s.txt" % which
    files = np.loadtxt(split_file, dtype=np.str)
    files = [f for f in files if f.split("_")[0] <= year]
    X, Y = [], []
    for f in files:
        X.append(load_kraehenbuehl(f))
        Y.append(get_ground_truth(f))

    return DataBunch(X, Y, files)
