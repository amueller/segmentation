import os
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread

from sklearn.externals.joblib import Memory
from sklearn.metrics import confusion_matrix
#from sklearn.preprocessing import StandardScaler

#from datasets.msrc import MSRCDataset
from pystruct.utils import make_grid_edges
from pystruct.problems import GraphCRF
#from pystruct.learners import StructuredSVM
from pystruct.learners import SubgradientStructuredSVM


from IPython.core.debugger import Tracer
tracer = Tracer()

memory = Memory(cachedir="cache")

classes = np.array(['building', 'grass', 'tree', 'cow', 'sheep', 'sky',
                    'aeroplane', 'water', 'face', 'car', 'bicycle', 'flower',
                    'sign', 'bird', 'book', 'chair', 'road', 'cat', 'dog',
                    'body', 'boat', 'void', 'mountain', 'horse'])

base_path = "/home/VI/staff/amueller/datasets/aurelien_msrc_features/msrc/"


def load_data(dataset="train"):
    mountain_idx = np.where(classes == "mountain")[0]
    horse_idx = np.where(classes == "horse")[0]
    void_idx = np.where(classes == "void")[0]

    ds_dict = dict(train="Train", val="Validation", test="Test")
    if dataset not in ds_dict.keys():
        raise ValueError("dataset must be one of 'train', 'val', 'test',"
                         " got %s" % dataset)
    ds_path = base_path + ds_dict[dataset]
    image_names, images, all_superpixels = [], [], []
    X, Y = [], []
    for f in glob(ds_path + "/*.dat"):
        name = os.path.basename(f).split('.')[0]
        img = imread("%s/%s.bmp" % (ds_path, name))
        labels = np.loadtxt(base_path + "labels/%s.txt" % name, dtype=np.int)
        images.append(img)
        image_names.append(name)
        # features
        #feat = np.hstack([np.loadtxt("%s/%s.local%s" % (ds_path, name, i)) for
                          #i in xrange(1, 7)])
        feat = np.hstack([np.loadtxt("%s/%s.local%s" % (ds_path, name, i)) for
                          i in xrange(1, 2)])
        # superpixels
        superpixels = np.fromfile("%s/%s.dat" % (ds_path, name),
                                  dtype=np.int32)
        superpixels = superpixels.reshape(img.shape[:-1][::-1]).T - 1
        all_superpixels.append(superpixels)
        # generate graph
        graph = region_graph(superpixels)
        X.append((feat, graph))
        #X.append((feat, np.empty((0, 2), dtype=np.int)))
        # make horse and mountain to void
        labels[labels == mountain_idx] = void_idx
        labels[labels == horse_idx] = void_idx
        Y.append(labels)
    return X, Y, image_names, images, all_superpixels


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


def plot_confusion_matrix(matrix, title=None):
    plt.matshow(matrix)
    plt.axis("off")
    plt.colorbar()
    for i, c in enumerate(classes[:-2]):
        plt.text(i, -1, c, rotation=60, va='bottom')
        plt.text(-1, i, c, ha='right')
    if title:
        plt.title(title)


def main():
    # load training data
    # let's just do images with cars first.
    #car_idx = np.where(classes == "car")[0]
    X, Y, image_names, images, all_superpixels = load_data("train")
    #n_states = len(np.unique(Y))
    n_states = 22
    print("number of samples: %s" % len(X))
    problem = GraphCRF(n_states=n_states, n_features=21,
                       inference_method='qpbo')
    #ssvm = StructuredSVM(problem, verbose=1, check_constraints=True, C=.1,
                         #n_jobs=-1, break_on_bad=False, max_iter=100)
    ssvm = SubgradientStructuredSVM(problem, verbose=1, C=100, n_jobs=-1,
                                    max_iter=100, learning_rate=0.0005,
                                    plot=True)
    ssvm.fit(X, Y)

    # do some evaluation on the training set
    print("score on training set: %f" % ssvm.score(X, Y))
    Y_pred = ssvm.predict(X)
    # compute confusion matrix
    confusion = np.zeros((n_states, n_states))
    for y, y_pred in zip(Y, Y_pred):
        confusion += confusion_matrix(y, y_pred, labels=np.arange(n_states))
    plot_confusion_matrix(confusion, title="confusion")

    # plot pairwise weights
    pairwise_flat = np.asarray(ssvm.w[problem.n_states * problem.n_features:])
    pairwise_params = np.zeros((problem.n_states, problem.n_states))
    pairwise_params[np.tri(problem.n_states, dtype=np.bool)] = pairwise_flat
    plot_confusion_matrix(pairwise_params, title="pairwise_params")

    # make figures with predictions
    for image, image_name, superpixels, y, y_pred in zip(images, image_names,
                                                         all_superpixels, Y,
                                                         Y_pred):
        fig, axes = plt.subplots(1, 3)
        axes[0].imshow(image)
        axes[1].set_title("ground truth")
        axes[1].imshow(image)
        axes[1].imshow(y[superpixels], vmin=0, vmax=23, alpha=.5)
        axes[2].set_title("prediction")
        axes[2].imshow(image)
        axes[2].imshow(y_pred[superpixels], vmin=0, vmax=23, alpha=.5)
        for ax in axes.ravel():
            ax.set_xticks(())
            ax.set_yticks(())
        fig.savefig("figures/%s.png" % image_name, bbox_inches="tight")
        plt.close(fig)
    plt.show()

    tracer()


if __name__ == "__main__":
    main()
