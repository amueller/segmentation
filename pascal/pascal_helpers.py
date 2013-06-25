from collections import namedtuple

import numpy as np
from scipy import sparse

from sklearn.externals.joblib import Memory
from slic_python import slic_n

from datasets.pascal import PascalSegmentation
from latent_crf_experiments.utils import gt_in_sp


memory = Memory(cachedir="/tmp/cache")
pascal_path = "/home/local/datasets/VOC2011/TrainVal/VOCdevkit/VOC2011"


# stores information that was COMPUTED from the dataset + file names for
# correspondence
DataBunch = namedtuple('DataBunch', 'X, Y, file_names, superpixels')
DataBunchNoSP = namedtuple('DataBunchNoSP', 'X, Y, file_names')


def load_kraehenbuehl(filename):
    path = "/home/user/amueller/local/voc_potentials_kraehenbuehl/unaries/"
    with open(path + filename + ".unary") as f:
        size = np.fromfile(f, dtype=np.uint32, count=3).byteswap()
        data = np.fromfile(f, dtype=np.float32).byteswap()
        img = data.reshape(size[1], size[0], size[2])
    return img


@memory.cache
def load_pascal_pixelwise(which='train', year="2010"):
    pascal = PascalSegmentation()
    if which not in ["train", "val"]:
        raise ValueError("Expected 'which' to be 'train' or 'val', got %s." %
                         which)
    split_file = pascal_path + "/ImageSets/Segmentation/%s.txt" % which
    files = np.loadtxt(split_file, dtype=np.str)
    files = [f for f in files if f.split("_")[0] <= year]
    X, Y = [], []
    for f in files:
        X.append(load_kraehenbuehl(f))
        Y.append(pascal.get_ground_truth(f))

    return DataBunchNoSP(X, Y, files)


def generate_pascal_split():
    # split the training set into train1 and train2 for validation
    base_path = pascal_path + "/ImageSets/Segmentation/"
    files = np.loadtxt(base_path + "train.txt", dtype=np.str)
    np.random.seed(0)
    inds = np.random.permutation(len(files))
    n_train2 = len(files) // 5
    np.savetxt(base_path + "train1.txt", files[inds > n_train2], fmt="%s")
    np.savetxt(base_path + "train2.txt", files[inds < n_train2], fmt="%s")


@memory.cache
def load_pascal(which='train', year="2010"):
    pascal = PascalSegmentation()
    files = pascal.get_split(which=which, year=year)
    X, Y, superpixels = [], [], []
    for f in files:
        image = pascal.get_image(f)
        superpixels.append(slic_n(image, n_superpixels=100, compactness=10))
        X.append(get_kraehenbuehl_pot_sp(f, superpixels[-1]))
        if which != "test":
            Y.append(gt_in_sp(pascal, f, superpixels[-1]))

    return DataBunch(X, Y, files, superpixels)


def get_kraehenbuehl_pot_sp(filename, superpixels):
    probs = load_kraehenbuehl(filename)
    # accumulate votes in superpixels
    # interleaved repeat
    class_indices = np.repeat(np.arange(21)[np.newaxis, :],
                              superpixels.size, axis=0).ravel()
    # non-interleaved repeat
    superpixel_indices = np.repeat(superpixels.ravel(), 21)
    sp_probs = sparse.coo_matrix((probs.ravel(), (superpixel_indices,
                                                  class_indices)))
    sp_probs = sp_probs.toarray()
    # renormalize (same as dividing by sp sizes)
    return sp_probs / sp_probs.sum(axis=-1)[:, np.newaxis]
