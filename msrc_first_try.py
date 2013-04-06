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
from pystruct import learners
from pystruct.problems.latent_graph_crf import kmeans_init
from pystruct.problems import GraphCRF, LatentGraphCRF
from pystruct.utils import SaveLogger

from datasets.msrc import MSRCDataset
from msrc_helpers import classes, DataBunch, plot_results, discard_void

#from ignore_void_crf import IgnoreVoidCRF

from IPython.core.debugger import Tracer
tracer = Tracer()

memory = Memory(cachedir="/tmp/cache")

base_path = "/home/user/amueller/datasets/aurelien_msrc_features/msrc/"


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


def plot_confusion_matrix(matrix, title=None):
    plt.matshow(matrix)
    plt.axis("off")
    plt.colorbar()
    for i, c in enumerate(classes[:-2]):
        plt.text(i, -1, c, rotation=60, va='bottom')
        plt.text(-1, i, c, ha='right')
    if title:
        plt.title(title)


def plot_parts():
    car_idx = np.where(classes == "car")[0]
    data = load_data("train", independent=False)
    car_images = np.array([i for i, y in enumerate(data.Y)
                           if np.any(y == car_idx)])
    flat_X = [x[0] for x in data.X]
    edges = [[x[1]] for x in data.X]
    n_states_per_label = np.ones(22, dtype=np.int)
    n_states_per_label[car_idx] = 6

    H = kmeans_init(flat_X, data.Y, edges, n_labels=22,
                    n_states_per_label=n_states_per_label, symmetric=True)
    X, Y, file_names, images, all_superpixels, H = zip(*[
        (data.X[i], data.Y[i], data.file_names[i], data.images[i],
         data.all_superpixels[i], H[i])
        for i in car_images])
    plot_results(images, file_names, Y, H, all_superpixels,
                 folder="test_parts", use_colors_predict=False)
    tracer()


def train_car_parts():
    car_idx = np.where(classes == "car")[0]
    data = load_data("train", independent=False)
    car_images = np.array([i for i, y in enumerate(data.Y)
                           if np.any(y == car_idx)])
    n_states_per_label = np.ones(22, dtype=np.int)
    n_states_per_label[car_idx] = 6

    X, Y, file_names, images, all_superpixels = zip(*[
        (data.X[i], data.Y[i], data.file_names[i], data.images[i],
         data.all_superpixels[i]) for i in car_images])
    problem = LatentGraphCRF(n_states_per_label=n_states_per_label,
                             n_labels=22, inference_method='ad3',
                             n_features=21 * 6)
    ssvm = learners.LatentSSVM(
        problem, verbose=2, C=10, max_iter=5000, n_jobs=-1, tol=0.0001,
        show_loss_every=10, base_svm='subgradient', inference_cache=50,
        latent_iter=5, learning_rate=0.001, decay_exponent=0.5)
    ssvm.fit(X, Y)
    plot_results(images, file_names, Y, ssvm.H_init_, all_superpixels,
                 folder="parts_init", use_colors_predict=False)
    H = ssvm.predict_latent(X)
    plot_results(images, file_names, Y, H, all_superpixels,
                 folder="parts_prediction", use_colors_predict=False)
    H_final = [problem.latent(x, y, ssvm.w) for x, y in zip(X, Y)]
    plot_results(images, file_names, Y, H_final, all_superpixels,
                 folder="parts_final", use_colors_predict=False)
    tracer()


def train_car():
    car_idx = np.where(classes == "car")[0]
    data_train = load_data("train", independent=False)
    car_images = np.array([i for i, y in enumerate(data_train.Y)
                           if np.any(y == car_idx)])
    n_states_per_label = np.ones(22, dtype=np.int)
    n_states_per_label[car_idx] = 6

    X, Y, file_names, images, all_superpixels = zip(*[
        (data_train.X[i], data_train.Y[i], data_train.file_names[i],
         data_train.images[i], data_train.superpixels[i])
        for i in car_images])
    problem = GraphCRF(n_states=22, inference_method='ad3', n_features=21 * 6)
    ssvm = learners.SubgradientStructuredSVM(
        problem, verbose=2, C=.001, max_iter=5000, n_jobs=-1,
        show_loss_every=10, learning_rate=0.0001, decay_exponent=0.5)
    ssvm.fit(X, Y)
    Y_pred = ssvm.predict(X)
    plot_results(images, file_names, Y, Y_pred, all_superpixels,
                 folder="cars_only")

    data_val = load_data("val", independent=False)
    car_images_val = np.array([i for i, y in enumerate(data_val.Y)
                               if np.any(y == car_idx)])
    X_val, Y_val, file_names_val, images_val, all_superpixels_val = \
        zip(*[(data_val.X[i], data_val.Y[i], data_val.file_names[i],
               data_val.images[i], data_val.superpixels[i]) for i in
              car_images_val])
    Y_pred_val = ssvm.predict(X_val)
    plot_results(images_val, file_names_val, Y_val, Y_pred_val,
                 all_superpixels_val, folder="cars_only_val")
    # C=10
    ## train:
    #0.92743060939680566V
    #> ssvm.score(X_val, Y_val)
    #0.52921719955898561
    # test 0.61693548387096775
    tracer()


def eval_on_pixels(data, sp_predictions):
    """Evaluate segmentation performance on pixel level.

    Parameters
    ----------
    data : DataBunch Named tuple
        Contains images, superpixels, descriptors and filenames.

    sp_predictions : list of arrays
        For each image, list of labels per superpixel

    """
    msrc = MSRCDataset()
    confusion = np.zeros((22, 22))
    for y, sp, f in zip(sp_predictions, data.superpixels, data.file_names):
        # load ground truth image
        gt = msrc.get_ground_truth(f)
        gt = gt - 1
        gt[gt == 255] == 21
        prediction = y[sp]
        confusion += confusion_matrix(gt.ravel(), prediction.ravel(),
                                      labels=np.arange(0, 22))

    confusion_normalized = (confusion.astype(np.float) /
                            confusion.sum(axis=1)[:, np.newaxis])
    tracer()
    return confusion, confusion_normalized


def train_svm():
    data_train = load_data("train", independent=True)
    X_features = [x[0] for x in data_train.X]
    X_features_flat = np.vstack(X_features)
    y = np.hstack(data_train.Y)
    from sklearn.linear_model import SGDClassifier
    svm = SGDClassifier()
    svm.fit(X_features_flat[y != 21], y[y != 21])

    #eval_on_pixels(data_train, [svm.predict(x) for x in X_features])

    plot_results(data_train, [svm.predict(x) for x in X_features],
                 folder="blub")

    Tracer()()


def main():
    # load training data
    independent = False
    data_train = load_data("train", independent=independent)
    X_, Y_ = discard_void(data_train.X, data_train.Y, 21)
    n_states = 21
    print("number of samples: %s" % len(data_train.X))
    problem = GraphCRF(n_states=n_states, n_features=21 * 6,
                       inference_method='ad3')
    #ssvm = learners.SubgradientStructuredSVM(
        #problem, verbose=2, C=.001, n_jobs=-1, max_iter=100000,
        #learning_rate=0.00015, show_loss_every=10, decay_exponent=.5,
        #momentum=0.98)
    ssvm = learners.OneSlackSSVM(
        problem, verbose=2, C=0.001, max_iter=100000, n_jobs=-1, tol=0.0001,
        show_loss_every=200, inference_cache=50,
        logger=SaveLogger("graph_ad3_0.0001_2.pickle", save_every=100),
        inactive_threshold=1e-5, break_on_bad=False)
    #ssvm = SaveLogger(file_name="graph_qpbo_0.001_3.pickle").load()
    #ssvm.logger = SaveLogger(file_name="graph_qpbo_0.001_3_refit.pickle")
    #ssvm.fit(X_, Y_, warm_start=True)
    ssvm.fit(X_, Y_)
    print("fit finished!")
    tracer()

    # do some evaluation on the training set
    print("score on training set: %f" % ssvm.score(X_, Y_))
    Y_pred = ssvm.predict(data_train.X)
    # compute confusion matrix
    confusion = np.zeros((n_states, n_states))
    for y, y_pred in zip(data_train.Y, Y_pred):
        confusion += confusion_matrix(y, y_pred, labels=np.arange(n_states))
    plot_confusion_matrix(confusion, title="confusion")

    # plot pairwise weights
    pairwise_flat = np.asarray(ssvm.w[problem.n_states * problem.n_features:])
    pairwise_params = np.zeros((problem.n_states, problem.n_states))
    pairwise_params[np.tri(problem.n_states, dtype=np.bool)] = pairwise_flat
    plot_confusion_matrix(pairwise_params, title="pairwise_params")
    plt.figure()
    plt.plot(ssvm.objective_curve_)
    plt.show()

    # make figures with predictions
    #plot_results(data_train.images, data_train.file_names, data_train.Y,
                 #Y_pred, data_train.all_superpixels, folder="figures_train")
    data_val = load_data("val", independent=independent)
    X_val_, Y_val_ = discard_void(data_val.X, data_val.Y, 21)
    print("score on validation set: %f" % ssvm.score(X_val_, Y_val_))
    Y_pred_val = ssvm.predict(data_val.X)
    plot_results(data_val, Y_pred_val, folder="figures_val")
    tracer()


if __name__ == "__main__":
    main()
    #train_svm()
    #plot_parts()
    #train_car_parts()
    #train_car()
