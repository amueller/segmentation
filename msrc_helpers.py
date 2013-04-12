from collections import namedtuple
import os
from glob import glob

import numpy as np
from scipy.misc import imread
from scipy import sparse
import matplotlib.pyplot as plt

from sklearn.externals.joblib import Memory
from sklearn.metrics import confusion_matrix

from pystruct.utils import make_grid_edges

from datasets.msrc import MSRCDataset, colors, classes

DataBunch = namedtuple('DataBunch', 'X, Y, file_names, images, superpixels')

base_path = "/home/user/amueller/datasets/aurelien_msrc_features/msrc/"
memory = Memory(cachedir="/tmp/cache")


@memory.cache
def load_data(dataset="train", independent=False):
    mountain_idx = np.where(classes == "mountain")[0]
    horse_idx = np.where(classes == "horse")[0]
    void_idx = np.where(classes == "void")[0]

    ds_dict = dict(train="Train", val="Validation", test="Test")
    if dataset not in ds_dict.keys():
        raise ValueError("dataset must be one of 'train', 'val', 'test',"
                         " got %s" % dataset)
    ds_path = base_path + ds_dict[dataset]
    file_names, images, all_superpixels = [], [], []
    X, Y = [], []
    for f in glob(ds_path + "/*.dat"):
        name = os.path.basename(f).split('.')[0]
        img = imread("%s/%s.bmp" % (ds_path, name))
        labels = np.loadtxt(base_path + "labels/%s.txt" % name, dtype=np.int)
        images.append(img)
        file_names.append(name)
        # features
        feat = np.hstack([np.loadtxt("%s/%s.local%s" % (ds_path, name, i)) for
                          i in xrange(1, 7)])
        # superpixels
        superpixels = np.fromfile("%s/%s.dat" % (ds_path, name),
                                  dtype=np.int32)
        superpixels = superpixels.reshape(img.shape[:-1][::-1]).T - 1
        all_superpixels.append(superpixels)
        # generate graph
        if independent:
            X.append((feat, np.empty((0, 2), dtype=np.int)))
        else:
            graph = region_graph(superpixels)
            X.append((feat, graph))
        # make horse and mountain to void
        labels[labels == mountain_idx] = void_idx
        labels[labels == horse_idx] = void_idx
        Y.append(labels)
    data = DataBunch(X, Y, file_names, images, all_superpixels)
    return data


def region_graph(regions):
    edges = make_grid_edges(regions)
    n_vertices = regions.size

    crossings = edges[regions.ravel()[edges[:, 0]]
                      != regions.ravel()[edges[:, 1]]]
    crossing_hash = (regions.ravel()[crossings[:, 0]]
                     + n_vertices * regions.ravel()[crossings[:, 1]])
    # find unique connections
    unique_hash = np.unique(crossing_hash)
    # undo hashing
    unique_crossings = np.asarray([[x % n_vertices, x / n_vertices]
                                   for x in unique_hash])
    if False:
        # plotting code
        # compute region centers:
        gridx, gridy = np.mgrid[:regions.shape[0], :regions.shape[1]]
        centers = np.zeros((n_vertices, 2))
        for v in xrange(n_vertices):
            centers[v] = [gridy[regions == v].mean(),
                          gridx[regions == v].mean()]
        # plot labels
        plt.imshow(regions)
        # overlay graph:
        for crossing in unique_crossings:
            plt.plot([centers[crossing[0]][0], centers[crossing[1]][0]],
                     [centers[crossing[0]][1], centers[crossing[1]][1]])
        plt.show()
    return unique_crossings


def plot_results(data, Y_pred, folder="figures", use_colors_predict=True):
    if not os.path.exists(folder):
        os.mkdir(folder)
    msrc = MSRCDataset()
    import matplotlib.colors as cl
    np.random.seed(0)
    random_colormap = cl.ListedColormap(np.random.uniform(size=(100, 3)))
    for image, image_name, superpixels, y, y_pred in zip(data.images,
                                                         data.file_names,
                                                         data.superpixels,
                                                         data.Y, Y_pred):
        fig, axes = plt.subplots(2, 3, figsize=(12, 6))
        axes[0, 0].imshow(image)
        axes[0, 1].set_title("ground truth")
        axes[0, 1].imshow(image)
        gt = msrc.get_ground_truth(image_name)
        gt = gt - 1
        gt[gt == 255] = 21
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


def discard_void(X, Y, void_label=21):
    X_new, Y_new = [], []
    for x, y in zip(X, Y):
        mask = y != void_label
        voids = np.where(~mask)[0]

        if len(x) == 2:
            features, edges = x
        else:
            features, edges, n_hidden = x
            mask = np.hstack([mask, np.ones(n_hidden, dtype=np.bool)])

        edges_new = edges
        if edges_new.shape[0] > 0:
            # if there are no edges, don't need to filter them
            # also, below code would break ;)
            for void_node in voids:
                involves_void_node = np.any(edges_new == void_node, axis=1)
                edges_new = edges_new[~involves_void_node]

        reindex_edges = np.zeros(len(mask), dtype=np.int)
        reindex_edges[mask] = np.arange(np.sum(mask))
        edges_new = reindex_edges[edges_new]
        if len(x) == 2:
            X_new.append((features[mask], edges_new))
            Y_new.append(y[mask])
        else:
            n_hidden_new = np.max(edges_new) - np.sum(mask[:-n_hidden]) + 1
            X_new.append((features[mask[:-n_hidden]], edges_new, n_hidden_new))
            Y_new.append(y[mask[:-n_hidden]])

    return X_new, Y_new


def load_kraehenbuehl(filename):
    path = "/home/user/amueller/datasets/kraehenbuehl_potentials_msrc/out/"
    with open(path + filename + ".unary") as f:
        size = np.fromfile(f, dtype=np.uint32, count=3).byteswap()
        data = np.fromfile(f, dtype=np.float32).byteswap()
        img = data.reshape(size[1], size[0], size[2])
    return img


@memory.cache
def get_kraehenbuehl_pot_sp(data):
    feats = []
    for x, filename, superpixels in zip(data.X, data.file_names,
                                        data.superpixels):
        probs = load_kraehenbuehl(filename)
        # accumulate votes in superpixels
        # interleaved repeat
        class_indices = np.repeat(np.arange(21)[np.newaxis, :],
                                  superpixels.size, axis=0).ravel()
        # non-interleaved repeat
        superpixel_indices = np.repeat(superpixels.ravel(), 21)
        sp_probs = sparse.coo_matrix((probs.ravel(), (superpixel_indices,
                                                      class_indices)))
        feats.append(sp_probs.toarray())
    return feats


def eval_on_pixels(data, sp_predictions, print_results=True):
    """Evaluate segmentation performance on pixel level.

    Parameters
    ----------
    data : DataBunch Named tuple
        Contains images, superpixels, descriptors and filenames.

    sp_predictions : list of arrays
        For each image, list of labels per superpixel

    print_results : bool, default=True
        Whether to print results to stdout.

    """
    msrc = MSRCDataset()
    confusion = np.zeros((22, 22))
    for y, sp, f in zip(sp_predictions, data.superpixels, data.file_names):
        # load ground truth image
        gt = msrc.get_ground_truth(f)
        gt = gt - 1
        gt[gt == 255] = 21
        prediction = y[sp]
        confusion += confusion_matrix(gt.ravel(), prediction.ravel(),
                                      labels=np.arange(0, 22))
    # drop void
    confusion_normalized = (confusion.astype(np.float) /
                            confusion.sum(axis=1)[:, np.newaxis])
    confusion = confusion[:-1, :-1]
    confusion_normalized = confusion_normalized[:-1, :-1]
    per_class_acc = np.diag(confusion_normalized)
    global_acc = np.diag(confusion).sum() / confusion.sum()
    average_acc = np.mean(per_class_acc)
    if print_results:
        print("global: %f, average: %f" % (global_acc, average_acc))
        print(["%s: %.2f" % (c, x) for c, x in zip(classes, per_class_acc)])
    return {'global': global_acc, 'average': average_acc,
            'per_class': per_class_acc, 'confusion': confusion}
