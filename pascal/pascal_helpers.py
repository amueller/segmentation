from collections import namedtuple
import os

import numpy as np
from scipy.misc import imread
from scipy import sparse
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


from sklearn.externals.joblib import Memory
from slic_python import slic_n

from latent_crf_experiments.utils import get_edge_contrast, get_edge_directions


memory = Memory(cachedir="/tmp/cache")
pascal_path = "/home/local/datasets/VOC2011/TrainVal/VOCdevkit/VOC2011"

colors = np.loadtxt("pascal_colors.txt")
cmap = ListedColormap(colors)

# stores information that was COMPUTED from the dataset + file names for
# correspondence
DataBunch = namedtuple('DataBunch', 'X, Y, file_names, superpixels')
DataBunchNoSP = namedtuple('DataBunchNoSP', 'X, Y, file_names')

classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
           'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'potted plant', 'sheep', 'sofa', 'train',
           'tvmonitor', 'void']


def load_image(filename):
    return imread(pascal_path + "/JPEGImages/%s.jpg" % filename)


def get_ground_truth(filename):
    return imread(pascal_path + "/SegmentationClass/%s.png" % filename)


def add_edge_features(data):
    X = []
    for x, superpixels, file_name in zip(data.X, data.superpixels,
                                         data.file_names):
        features = [np.ones((x[1].shape[0], 1))]
        image = load_image(file_name)
        features.append(get_edge_contrast(x[1], image, superpixels))
        features.append(get_edge_directions(x[1], superpixels))
        X.append((x[0], x[1], np.hstack(features)))
    return DataBunch(X, data.Y, data.file_names, data.superpixels)


def load_kraehenbuehl(filename):
    path = "/home/user/amueller/local/voc_potentials_kraehenbuehl/unaries/"
    with open(path + filename + ".unary") as f:
        size = np.fromfile(f, dtype=np.uint32, count=3).byteswap()
        data = np.fromfile(f, dtype=np.float32).byteswap()
        img = data.reshape(size[1], size[0], size[2])
    return img


@memory.cache
def load_pascal_pixelwise(which='train', year="2010"):
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

    return DataBunchNoSP(X, Y, files)


def gt_in_sp(filename, superpixels):
    y = get_ground_truth(filename)
    votes = sparse.coo_matrix((np.ones(superpixels.size),
                               (y.ravel(), superpixels.ravel())))
    return np.argmax(votes.toarray(), axis=0)


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
    if which not in ["train", "val", "train1", "train2"]:
        raise ValueError("Expected 'which' to be 'train' or 'val', got %s." %
                         which)
    split_file = pascal_path + "/ImageSets/Segmentation/%s.txt" % which
    files = np.loadtxt(split_file, dtype=np.str)
    files = [f for f in files if f.split("_")[0] <= year]
    X, Y, superpixels = [], [], []
    for f in files:
        image = load_image(f)
        superpixels.append(slic_n(image, n_superpixels=100, compactness=10))
        X.append(get_kraehenbuehl_pot_sp(f, superpixels[-1]))
        Y.append(gt_in_sp(f, superpixels[-1]))

    return DataBunch(X, Y, files, superpixels)


def eval_on_pixels(Y_true, Y_pred, print_results=False):
    tp, tn, fp, fn = np.zeros(21), np.zeros(21), np.zeros(21), np.zeros(21)
    for y_true, y_pred in zip(Y_true, Y_pred):
        mask = y_true != 255  # don't care at borders
        y_true, y_pred = y_true[mask], y_pred[mask]
        for k in range(21):
            tp[k] += np.sum((y_true == k) * (y_pred == k))
            tn[k] += np.sum((y_true != k) * (y_pred != k))
            fp[k] += np.sum((y_true != k) * (y_pred == k))
            fn[k] += np.sum((y_true == k) * (y_pred != k))
    jaccard = tp / (fp + fn + tp) * 100
    hamming = tp / (tp + fn) * 100
    if print_results:
        np.set_printoptions(precision=2)
        print("Jaccard")
        print(jaccard)
        print("Hamming")
        print(hamming)
        print("Mean Jaccard: %.1f   Mean Hamming: %.1f"
              % (np.mean(jaccard), np.mean(hamming)))

    return hamming, jaccard


def eval_on_sp(data, Y_pred, print_results=False):
    Y_pred_pixels = [y_pred[sp] for sp, y_pred in zip(data.superpixels,
                                                      Y_pred)]
    Y_true = [get_ground_truth(f) for f in data.file_names]
    return eval_on_pixels(Y_true, Y_pred_pixels, print_results=print_results)


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


def plot_results(data, Y_pred, folder="figures", use_colors_predict=True):
    if not os.path.exists(folder):
        os.mkdir(folder)
    import matplotlib.colors as cl
    np.random.seed(0)
    random_colormap = cl.ListedColormap(np.random.uniform(size=(100, 3)))
    for image_name, superpixels, y, y_pred in zip(data.file_names,
                                                  data.superpixels, data.Y,
                                                  Y_pred):
        image = load_image(image_name)
        fig, axes = plt.subplots(2, 3, figsize=(12, 6))
        axes[0, 0].imshow(image)
        axes[0, 1].set_title("ground truth")
        axes[0, 1].imshow(image)
        gt = get_ground_truth(image_name)
        axes[0, 1].imshow(colors[gt], alpha=.7)
        axes[1, 0].set_title("sp ground truth")
        axes[1, 0].imshow(image)
        axes[1, 0].imshow(colors[y[superpixels]], vmin=0, vmax=23, alpha=.7)

        axes[1, 1].set_title("prediction")
        axes[1, 1].imshow(image)
        if use_colors_predict:
            axes[1, 1].imshow(colors[y_pred[superpixels]], alpha=.7)
        else:
            vmax = np.max(np.hstack(Y_pred))
            axes[1, 1].imshow(y_pred[superpixels], vmin=0, vmax=vmax, alpha=.9,
                              cmap=random_colormap)
        if use_colors_predict:
            present_y = np.unique(np.hstack([y, y_pred]))
        else:
            present_y = np.unique(y)
        present_y = present_y[present_y != 255]
        axes[0, 2].imshow(colors[present_y, :][:, np.newaxis, :],
                          interpolation='nearest')
        for i, c in enumerate(present_y):
            axes[0, 2].text(1, i, classes[c])
        for ax in axes.ravel():
            ax.set_xticks(())
            ax.set_yticks(())
        axes[1, 2].set_visible(False)
        fig.savefig(folder + "/%s.png" % image_name, bbox_inches="tight")
        plt.close(fig)
