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

#from ignore_void_crf import IgnoreVoidCRF

from IPython.core.debugger import Tracer
tracer = Tracer()

memory = Memory(cachedir="cache")

colors = np.array(
    [[128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
     # [128, 0, 128], horse
     [0, 128, 128], [128, 128, 128],
     # [64, 0, 0], mountain
     [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
     [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0],
     [128, 64, 0], [0, 192, 0], [128, 64, 128], [0, 192, 128],
     [128, 192, 128], [64, 64, 0], [192, 64, 0], [0, 0, 0]])

classes = np.array(['building', 'grass', 'tree', 'cow', 'sheep', 'sky',
                    'aeroplane', 'water', 'face', 'car', 'bicycle', 'flower',
                    'sign', 'bird', 'book', 'chair', 'road', 'cat', 'dog',
                    'body', 'boat', 'void', 'mountain', 'horse'])

base_path = "/home/VI/staff/amueller/datasets/aurelien_msrc_features/msrc/"


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
    image_names, images, all_superpixels = [], [], []
    X, Y = [], []
    for f in glob(ds_path + "/*.dat"):
        name = os.path.basename(f).split('.')[0]
        img = imread("%s/%s.bmp" % (ds_path, name))
        labels = np.loadtxt(base_path + "labels/%s.txt" % name, dtype=np.int)
        images.append(img)
        image_names.append(name)
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
    return X, Y, image_names, images, all_superpixels


def discard_void(X, Y, void_label=21):
    X_new, Y_new = [], []
    for x, y in zip(X, Y):
        features, edges = x
        mask = y != void_label
        voids = np.where(~mask)[0]
        edges_new = edges
        if edges_new.shape[0] > 0:
            # if there are no edges, don't need to filter them
            # also, below code would break ;)
            for void_node in voids:
                involves_void_node = np.any(edges_new == void_node, axis=1)
                edges_new = edges_new[~involves_void_node]
        reindex_edges = 1000 * np.zeros(len(mask), dtype=np.int)
        reindex_edges[mask] = np.arange(len(mask))
        edges_new = reindex_edges[edges_new]

        if np.max(edges_new) > len(features[mask]):
            tracer()
        X_new.append((features[mask], edges_new))
        Y_new.append(y[mask])
    return X_new, Y_new


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


def plot_results(images, image_names, Y, Y_pred, all_superpixels,
                 folder="figures", use_colors_predict=True):
    if not os.path.exists(folder):
        os.mkdir(folder)
    import matplotlib.colors as cl
    np.random.seed(0)
    random_colormap = cl.ListedColormap(np.random.uniform(size=(100, 3)))
    for image, image_name, superpixels, y, y_pred in zip(images, image_names,
                                                         all_superpixels, Y,
                                                         Y_pred):
        fig, axes = plt.subplots(1, 4, figsize=(12, 3))
        axes[0].imshow(image)
        axes[1].set_title("ground truth")
        axes[1].imshow(image)
        axes[1].imshow(colors[y[superpixels]], vmin=0, vmax=23, alpha=.7)
        axes[2].set_title("prediction")
        axes[2].imshow(image)
        if use_colors_predict:
            axes[2].imshow(colors[y_pred[superpixels]], vmin=0, vmax=23,
                           alpha=.7)
        else:
            vmax = np.max(np.hstack(Y_pred))
            axes[2].imshow(y_pred[superpixels], vmin=0, vmax=vmax, alpha=.9,
                           cmap=random_colormap)
        if use_colors_predict:
            present_y = np.unique(np.hstack([y, y_pred]))
        else:
            present_y = np.unique(y)
        axes[3].imshow(colors[present_y, :][:, np.newaxis, :],
                       interpolation='nearest')
        for i, c in enumerate(present_y):
            axes[3].text(1, i, classes[c])
        for ax in axes.ravel():
            ax.set_xticks(())
            ax.set_yticks(())
        fig.savefig(folder + "/%s.png" % image_name, bbox_inches="tight")
        plt.close(fig)


def plot_parts():
    car_idx = np.where(classes == "car")[0]
    X, Y, image_names, images, all_superpixels = load_data(
        "train", independent=False)
    car_images = np.array([i for i, y in enumerate(Y) if np.any(y == car_idx)])
    flat_X = [x[0] for x in X]
    edges = [[x[1]] for x in X]
    n_states_per_label = np.ones(22, dtype=np.int)
    n_states_per_label[car_idx] = 6

    H = kmeans_init(flat_X, Y, edges, n_labels=22,
                    n_states_per_label=n_states_per_label, symmetric=True)
    X, Y, image_names, images, all_superpixels, H = zip(*[
        (X[i], Y[i], image_names[i], images[i], all_superpixels[i], H[i])
        for i in car_images])
    plot_results(images, image_names, Y, H, all_superpixels,
                 folder="test_parts", use_colors_predict=False)
    tracer()


def train_car_parts():
    car_idx = np.where(classes == "car")[0]
    X, Y, image_names, images, all_superpixels = load_data(
        "train", independent=False)
    car_images = np.array([i for i, y in enumerate(Y) if np.any(y == car_idx)])
    n_states_per_label = np.ones(22, dtype=np.int)
    n_states_per_label[car_idx] = 6

    X, Y, image_names, images, all_superpixels = zip(*[
        (X[i], Y[i], image_names[i], images[i], all_superpixels[i])
        for i in car_images])
    problem = LatentGraphCRF(n_states_per_label=n_states_per_label,
                             n_labels=22, inference_method='ad3',
                             n_features=21 * 6)
    ssvm = learners.LatentSSVM(
        problem, verbose=2, C=10, max_iter=5000, n_jobs=-1, tol=0.0001,
        show_loss_every=10, base_svm='subgradient', inference_cache=50,
        latent_iter=5, learning_rate=0.001, decay_exponent=0.5)
    ssvm.fit(X, Y)
    plot_results(images, image_names, Y, ssvm.H_init_, all_superpixels,
                 folder="parts_init", use_colors_predict=False)
    H = ssvm.predict_latent(X)
    plot_results(images, image_names, Y, H, all_superpixels,
                 folder="parts_prediction", use_colors_predict=False)
    H_final = [problem.latent(x, y, ssvm.w) for x, y in zip(X, Y)]
    plot_results(images, image_names, Y, H_final, all_superpixels,
                 folder="parts_final", use_colors_predict=False)
    tracer()


def train_car():
    car_idx = np.where(classes == "car")[0]
    X, Y, image_names, images, all_superpixels = load_data(
        "train", independent=False)
    car_images = np.array([i for i, y in enumerate(Y) if np.any(y == car_idx)])
    n_states_per_label = np.ones(22, dtype=np.int)
    n_states_per_label[car_idx] = 6

    X, Y, image_names, images, all_superpixels = zip(*[
        (X[i], Y[i], image_names[i], images[i], all_superpixels[i])
        for i in car_images])
    problem = GraphCRF(n_states=22, inference_method='ad3', n_features=21 * 6)
    ssvm = learners.SubgradientStructuredSVM(
        problem, verbose=2, C=.001, max_iter=5000, n_jobs=-1,
        show_loss_every=10, learning_rate=0.0001, decay_exponent=0.5)
    ssvm.fit(X, Y)
    Y_pred = ssvm.predict(X)
    plot_results(images, image_names, Y, Y_pred, all_superpixels,
                 folder="cars_only")

    X_val, Y_val, image_names_val, images_val, all_superpixels_val = \
        load_data("val", independent=False)
    car_images_val = np.array([i for i, y in enumerate(Y_val)
                               if np.any(y == car_idx)])
    X_val, Y_val, image_names_val, images_val, all_superpixels_val = \
        zip(*[(X_val[i], Y_val[i], image_names_val[i], images_val[i],
               all_superpixels_val[i]) for i in car_images_val])
    Y_pred_val = ssvm.predict(X_val)
    plot_results(images_val, image_names_val, Y_val, Y_pred_val,
                 all_superpixels_val, folder="cars_only_val")
    # C=10
    ## train:
    #0.92743060939680566V
    #> ssvm.score(X_val, Y_val)
    #0.52921719955898561
    # test 0.61693548387096775
    tracer()


def main():
    # load training data
    independent = False
    X, Y, image_names, images, all_superpixels = load_data(
        "train", independent=independent)
    X_, Y_ = discard_void(X, Y, 21)
    n_states = 21
    print("number of samples: %s" % len(X))
    problem = GraphCRF(n_states=n_states, n_features=21 * 6,
                       inference_method='qpbo')
    # 80% on training set with C=100, lr=0.000001, max_iter=100
    # score on training set: 0.843112 with ignoreVoidCRF, max_iter=100
    # score on training set: 0.850884 with C=100, max_iter=100, pairwise.
    # careful! not really the right measure :-/
    # val:0.78365315031101435
    # OneSlack, C=10, max_iter=200, score on training set: 0.689209
    ssvm = learners.SubgradientStructuredSVM(
        problem, verbose=2, C=.001, n_jobs=-1, max_iter=100000,
        learning_rate=0.00015, show_loss_every=10, decay_exponent=.5,
        momentum=0.98)
    #ssvm = learners.OneSlackSSVM(problem, verbose=2, C=100, max_iter=5000,
                                 #n_jobs=-1, tol=0.0001, show_loss_every=20,
                                 #inference_cache=50)
    #ssvm.w = np.load("pairwise_w.npy")
    ssvm.fit(X_, Y_)
    print("fit finished!")

    # do some evaluation on the training set
    print("score on training set: %f" % ssvm.score(X_, Y_))
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
    plt.figure()
    plt.plot(ssvm.objective_curve_)
    plt.show()

    # make figures with predictions
    plot_results(images, image_names, Y, Y_pred, all_superpixels,
                 folder="figures_train")
    X_val, Y_val, image_names_val, images_val, all_superpixels_val = load_data(
        "val", independent=independent)
    X_val_, Y_val_ = discard_void(X_val, Y_val, 21)
    print("score on validation set: %f" % ssvm.score(X_val_, Y_val_))
    Y_pred_val = ssvm.predict(X_val)
    plot_results(images_val, image_names_val, Y_val, Y_pred_val,
                 all_superpixels_val, folder="figures_val")
    tracer()


if __name__ == "__main__":
    main()
    #plot_parts()
    #train_car_parts()
    #train_car()
