from collections import namedtuple
#from glob import glob

import cPickle

import numpy as np
#from scipy.misc import imread
from scipy import sparse
import matplotlib.pyplot as plt

from sklearn.externals.joblib import Memory
#from sklearn.metrics import confusion_matrix
from sklearn.kernel_approximation import AdditiveChi2Sampler


from datasets.msrc import MSRC21Dataset
#from latent_crf_experiments.utils import (add_edges, get_edge_contrast,
                                          #get_edge_directions)
from latent_crf_experiments.utils import (get_edge_contrast,
                                          get_edge_directions)

# stores information that was COMPUTED from the dataset + file names for
# correspondence
DataBunch = namedtuple('DataBunch', 'X, Y, file_names, superpixels')

base_path = "/home/user/amueller/datasets/aurelien_msrc_features/msrc/"
memory = Memory(cachedir="/tmp/cache")


class PixelwiseScorer(object):
    def __init__(self, data):
        self.data = data
        self.greater_is_better = True

    def __call__(self, estimator, X, y):
        result = eval_on_pixels(self.data,
                                [estimator.predict(x) for x in self.data.X])
        return result['average']


class SimpleSplitCV():
    def __init__(self, n_train, n_test):
        self.n_train = n_train
        self.n_test = n_test

    def __iter__(self):
        mask_train = np.zeros(self.n_train + self.n_test, dtype=np.bool)
        mask_test = mask_train.copy()
        mask_train[:self.n_train] = True
        mask_test[self.n_train:] = True
        yield mask_train, mask_test


def load_data(dataset="train", which="bow"):
    if which == "bow":
        filename = ("/home/user/amueller/checkout/superpixel_crf/"
                    "data_%s_1000_color.pickle" % dataset)
    elif which == "bow_old":
        filename = ("/home/user/amueller/checkout/superpixel_crf/"
                    "data_%s_1000_color_old.pickle" % dataset)
    elif which == "bow_new":
        filename = ("/home/user/amueller/checkout/superpixel_crf/"
                    "data_%s_1000_color_new.pickle" % dataset)
    elif which == "bow_5k":
        filename = ("/home/user/amueller/checkout/superpixel_crf/"
                    "data_%s_5000_color.pickle" % dataset)
    elif which == "piecewise":
        filename = ("/home/user/amueller/checkout/superpixel_crf/"
                    "data_probs_%s_cw_2.pickle" % dataset)
    elif which == "piecewise_trainval":
        filename = ("/home/user/amueller/checkout/superpixel_crf/"
                    "data_probs_%s_cw_trainval.pickle" % dataset)
    elif which == "piecewise_new":
        filename = ("/home/user/amueller/checkout/superpixel_crf/"
                    "data_probs_%s_new.pickle" % dataset)
    else:
        raise ValueError("'which' should be 'bow' or 'piecewise'")

    with open(filename) as f:
            data = cPickle.load(f)
    if which in ["bow", "bow_old", "bow_new", "bow_5k"]:
        data = transform_chi2(data)
    if which == "piecewise_new":
        X = [sigm(x) for x in data.X]
        data = DataBunch(X, data.Y, data.file_names, data.superpixels)
    return data


#def load_data_aurelien(dataset="train", independent=False):
    #mountain_idx = np.where(classes == "mountain")[0]
    #horse_idx = np.where(classes == "horse")[0]
    #void_idx = np.where(classes == "void")[0]

    #ds_dict = dict(train="Train", val="Validation", test="Test")
    #if dataset not in ds_dict.keys():
        #raise ValueError("dataset must be one of 'train', 'val', 'test',"
                         #" got %s" % dataset)
    #ds_path = base_path + ds_dict[dataset]
    #file_names, all_superpixels = [], []
    #X, Y = [], []
    #for f in glob(ds_path + "/*.dat"):
        #name = os.path.basename(f).split('.')[0]
        #img = imread("%s/%s.bmp" % (ds_path, name))
        #labels = np.loadtxt(base_path + "labels/%s.txt" % name, dtype=np.int)
        #file_names.append(name)
        ## features
        #feat = np.hstack([np.loadtxt("%s/%s.local%s" % (ds_path, name, i)) for
                          #i in xrange(1, 7)])
        ## superpixels
        #superpixels = np.fromfile("%s/%s.dat" % (ds_path, name),
                                  #dtype=np.int32)
        #superpixels = superpixels.reshape(img.shape[:-1][::-1]).T - 1
        #all_superpixels.append(superpixels)
        ## make horse and mountain to void
        #labels[labels == mountain_idx] = void_idx
        #labels[labels == horse_idx] = void_idx
        #Y.append(labels)
        #X.append(feat)
    #data = DataBunch(X, Y, file_names, all_superpixels)
    #data = add_edges(data, independent=independent)
    #return data


def concatenate_datasets(data1, data2):
    X = data1.X + data2.X
    Y = data1.Y + data2.Y
    file_names = np.hstack([data1.file_names, data2.file_names])
    superpixels = data1.superpixels + data2.superpixels
    return DataBunch(X, Y, file_names, superpixels)


@memory.cache
def transform_chi2(data):
    chi2 = AdditiveChi2Sampler(sample_steps=2)
    if isinstance(data.X[0], np.ndarray):
        X_new = [chi2.fit_transform(x).astype(np.float32) for x in data.X]
    elif len(data.X[0]) == 2:
        X_new = [(chi2.fit_transform(x[0]), x[1]) for x in data.X]
    elif len(data.X[0]) == 3:
        X_new = [(chi2.fit_transform(x[0]), x[1], x[2]) for x in data.X]
    else:
        raise ValueError("len(x) is weird: %d" % len(data.X[0]))

    return DataBunch(X_new, data.Y, data.file_names, data.superpixels)


def load_kraehenbuehl(filename, which="train"):
    if which == "train":
        path = "/home/user/amueller/datasets/kraehenbuehl_potentials_msrc/out/"
    elif which == "trainval":
        path = ("/home/user/amueller/datasets/kraehenbuehl_potentials_msrc/"
                "textonboost_trainval/")
    elif which == "train_30px":
        path = ("/home/user/amueller/datasets/kraehenbuehl_potentials_msrc/"
                "train_30px/")
    elif which == "trainval_30px":
        path = ("/home/user/amueller/datasets/kraehenbuehl_potentials_msrc/"
                "trainval_30px/")
    else:
        raise ValueError("Unexpected which in load_kraehenbuehl: %s" % which)
    #path = "/home/local/datasets/MSRC_ObjCategImageDatabase_v2/asdf/"
    with open(path + filename + ".unary") as f:
        size = np.fromfile(f, dtype=np.uint32, count=3).byteswap()
        data = np.fromfile(f, dtype=np.float32).byteswap()
        img = data.reshape(size[1], size[0], size[2])
    return img


@memory.cache
def get_kraehenbuehl_pot_sp(data, which="train"):
    feats = []
    for x, filename, superpixels in zip(data.X, data.file_names,
                                        data.superpixels):
        probs = load_kraehenbuehl(filename, which=which)
        if which != "train":
            # softmax normalization
            probs -= np.max(probs, axis=-1)[:, :, np.newaxis]
            probs = np.exp(probs)
            probs /= probs.sum(axis=-1)[:, :, np.newaxis]

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
        feats.append(sp_probs / sp_probs.sum(axis=-1)[:, np.newaxis])
    return feats


def sigm(x):
    return 1. / (1 + np.exp(-x))


def add_kraehenbuehl_features(data, which="train", replace=False):
    sp_probas = get_kraehenbuehl_pot_sp(data, which=which)
    if replace:
        X = [probas
             for probas in sp_probas]
        return DataBunch(X, data.Y, data.file_names, data.superpixels)
    if isinstance(data.X[0], np.ndarray):
        X = [np.hstack([x, probas])
             for x, probas in zip(data.X, sp_probas)]
    else:
        X = [(np.hstack([x[0], probas]), x[1])
             for x, probas in zip(data.X, sp_probas)]
    return DataBunch(X, data.Y, data.file_names, data.superpixels)


def eval_on_pixels(data, sp_predictions, print_results=True):
    """Evaluate segmentation performance on pixel level.

    Parameters
    ----------
    data : DataBunch Named tuple
        Contains superpixels, descriptors, superpixel gt and filenames.

    sp_predictions : list of arrays
        For each image, list of labels per superpixel

    print_results : bool, default=True
        Whether to print results to stdout.

    """
    msrc = MSRC21Dataset()
    pixel_predictions = [sp_pred[sp] for sp_pred, sp in zip(sp_predictions,
                                                            data.superpixels)]
    result = msrc.eval_pixel_performance(data.file_names, pixel_predictions,
                                         print_results=print_results)
    return result


def add_edge_features(data):
    X = []
    msrc = MSRC21Dataset()
    for x, superpixels, file_name in zip(data.X, data.superpixels,
                                         data.file_names):
        features = [np.ones((x[1].shape[0], 1))]
        image = msrc.get_image(file_name)
        features.append(get_edge_contrast(x[1], image, superpixels))
        features.append(get_edge_directions(x[1], superpixels))
        X.append((x[0], x[1], np.hstack(features)))
    return DataBunch(X, data.Y, data.file_names, data.superpixels)


def plot_confusion_matrix(dataset, confusion, title=None):
    confusion_normalized = (confusion.astype(np.float) /
                            confusion.sum(axis=1)[:, np.newaxis])
    plt.matshow(confusion_normalized)
    plt.axis("off")
    plt.colorbar()
    for i, c in enumerate(dataset.classes):
        plt.text(i, -1, c, rotation=60, va='bottom')
        plt.text(-1, i, c, ha='right')
    if title:
        plt.title(title)
