from collections import namedtuple
import os
from glob import glob

import numpy as np
from scipy.misc import imread
from scipy import sparse
import matplotlib.pyplot as plt

from sklearn.externals.joblib import Memory
#from sklearn.metrics import confusion_matrix
from sklearn.kernel_approximation import AdditiveChi2Sampler

from pystruct.utils import make_grid_edges

from datasets.msrc import MSRCDataset, colors, classes

# stores information that was COMPUTED from the dataset + file names for
# correspondence
DataBunch = namedtuple('DataBunch', 'X, Y, file_names, superpixels')

base_path = "/home/user/amueller/datasets/aurelien_msrc_features/msrc/"
memory = Memory(cachedir="/tmp/cache")


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
    file_names, all_superpixels = [], []
    X, Y = [], []
    for f in glob(ds_path + "/*.dat"):
        name = os.path.basename(f).split('.')[0]
        img = imread("%s/%s.bmp" % (ds_path, name))
        labels = np.loadtxt(base_path + "labels/%s.txt" % name, dtype=np.int)
        file_names.append(name)
        # features
        feat = np.hstack([np.loadtxt("%s/%s.local%s" % (ds_path, name, i)) for
                          i in xrange(1, 7)])
        # superpixels
        superpixels = np.fromfile("%s/%s.dat" % (ds_path, name),
                                  dtype=np.int32)
        superpixels = superpixels.reshape(img.shape[:-1][::-1]).T - 1
        all_superpixels.append(superpixels)
        # make horse and mountain to void
        labels[labels == mountain_idx] = void_idx
        labels[labels == horse_idx] = void_idx
        Y.append(labels)
        X.append(feat)
    data = DataBunch(X, Y, file_names, all_superpixels)
    data = add_edges(data, independent=independent)
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
    for image_name, superpixels, y, y_pred in zip(data.file_names,
                                                  data.superpixels, data.Y,
                                                  Y_pred):
        image = msrc.get_image(image_name)
        fig, axes = plt.subplots(2, 3, figsize=(12, 6))
        axes[0, 0].imshow(image)
        axes[0, 1].set_title("ground truth")
        axes[0, 1].imshow(image)
        gt = msrc.get_ground_truth(image_name)
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


def add_edges(data, independent=False):
    # generate graph
    if independent:
        X_new = [(x, np.empty((0, 2), dtype=np.int)) for x in data.X]
    else:
        X_new = [(x, np.sort(region_graph(sp), axis=-1))
                 for x, sp in zip(data.X, data.superpixels)]

    return DataBunch(X_new, data.Y, data.file_names, data.superpixels)


@memory.cache
def transform_chi2(data):
    chi2 = AdditiveChi2Sampler(sample_steps=2)
    if isinstance(data.X[0], np.ndarray):
        X_new = [chi2.fit_transform(x) for x in data.X]
    elif len(data.X[0]) == 2:
        X_new = [(chi2.fit_transform(x[0]), x[1]) for x in data.X]
    elif len(data.X[0]) == 3:
        X_new = [(chi2.fit_transform(x[0]), x[1], x[2]) for x in data.X]
    else:
        raise ValueError("len(x) is weird: %d" % len(data.X[0]))

    return DataBunch(X_new, data.Y, data.file_names, data.superpixels)


@memory.cache
def discard_void(data, void_label=21):
    if isinstance(data.X[0], np.ndarray):
        X_new = [x[y != void_label] for x, y in zip(data.X, data.Y)]
        Y_new = [y[y != void_label] for y in data.Y]
        return DataBunch(X_new, Y_new, data.file_names,
                         data.superpixels)
    X_new, Y_new = [], []
    for x, y in zip(data.X, data.Y):
        mask = y != void_label
        voids = np.where(~mask)[0]

        if len(x) == 2:
            features, edges = x
        elif len(x) == 3:
            features, edges, n_hidden = x
            mask = np.hstack([mask, np.ones(n_hidden, dtype=np.bool)])
        else:
            raise ValueError("len(x) is weird: %d" % len(data.X[0]))

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
            #X_new.append((features[mask], edges_new, n_hidden_new))
            #Y_new.append(y[mask[:-n_hidden]])

    return DataBunch(X_new, Y_new, data.file_names, data.superpixels)


def load_kraehenbuehl(filename):
    #path = "/home/user/amueller/datasets/kraehenbuehl_potentials_msrc/out/"
    path = ("/home/user/amueller/datasets/kraehenbuehl_potentials_msrc/"
            "textonboost_trainval/")
    #path = "/home/local/datasets/MSRC_ObjCategImageDatabase_v2/asdf/"
    with open(path + filename + ".unary") as f:
        size = np.fromfile(f, dtype=np.uint32, count=3).byteswap()
        data = np.fromfile(f, dtype=np.float32).byteswap()
        img = data.reshape(size[1], size[0], size[2])
    return img


#@memory.cache
def get_kraehenbuehl_pot_sp(data):
    feats = []
    for x, filename, superpixels in zip(data.X, data.file_names,
                                        data.superpixels):
        probs = load_kraehenbuehl(filename)
        #if np.min(probs) < 0:
        if True:
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
    msrc = MSRCDataset()
    pixel_predictions = [sp_pred[sp] for sp_pred, sp in zip(sp_predictions,
                                                            data.superpixels)]
    result = msrc.eval_pixel_performance(data.file_names, pixel_predictions,
                                         print_results=print_results)
    return result


def add_edge_features(data):
    X = []
    msrc = MSRCDataset()
    for x, superpixels, file_name in zip(data.X, data.superpixels,
                                         data.file_names):
        features = [np.ones((x[1].shape[0], 1))]
        image = msrc.get_image(file_name)
        features.append(get_edge_contrast(x[1], image, superpixels))
        features.append(get_edge_directions(x[1], superpixels))
        X.append((x[0], x[1], np.hstack(features)))
    return DataBunch(X, data.Y, data.file_names, data.superpixels)


def get_edge_contrast(edges, image, superpixels):
    r = np.bincount(superpixels.ravel(), weights=image[:, :, 0].ravel())
    g = np.bincount(superpixels.ravel(), weights=image[:, :, 1].ravel())
    b = np.bincount(superpixels.ravel(), weights=image[:, :, 2].ravel())
    mean_colors = (np.vstack([r, g, b])
                   / np.bincount(superpixels.ravel())).T / 255.
    contrasts = [np.exp(-10. * np.linalg.norm(mean_colors[e[0]]
                                              - mean_colors[e[1]]))
                 for e in edges]
    return np.vstack(contrasts)


@memory.cache
def get_edge_directions(edges, superpixels):
    n_vertices = np.max(superpixels) + 1
    centers = np.empty((n_vertices, 2))
    gridx, gridy = np.mgrid[:superpixels.shape[0], :superpixels.shape[1]]

    for v in xrange(n_vertices):
        centers[v] = [gridy[superpixels == v].mean(),
                      gridx[superpixels == v].mean()]
    directions = []
    for edge in edges:
        e0, e1 = edge
        diff = centers[e0] - centers[e1]
        diff /= np.linalg.norm(diff)
        directions.append(np.arcsin(diff[1]))
    return np.vstack(directions)
