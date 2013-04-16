import cPickle

import numpy as np
import matplotlib.pyplot as plt

#from sklearn.kernel_approximation import AdditiveChi2Sampler

from datasets.msrc import MSRCDataset
from pystruct import learners
from pystruct.problems.latent_graph_crf import kmeans_init
#from pystruct.problems import EdgeFeatureGraphCRF
from pystruct.problems import GraphCRF
from pystruct.utils import SaveLogger

from msrc_helpers import (classes, load_data, plot_results, discard_void,
                          eval_on_pixels, add_edge_features, add_edges,
                          DataBunch)

from kraehenbuehl_potentials import add_kraehenbuehl_features


from IPython.core.debugger import Tracer
tracer = Tracer()


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


def load_stacked_results(ds='train'):
    path = "../superpixel_crf/blub/"
    msrc = MSRCDataset()
    files = msrc.get_split(ds)
    X = []
    for f in files:
        probs = np.load(path + f + "_probs.npy")
        X.append(probs)
    with open("../superpixel_crf/data_probs_%s_cw.pickle"
              % ds) as f:
        data = cPickle.load(f)
    assert(np.all(data.file_names == files))
    return DataBunch(X, data.Y, files, data.superpixels)


def train_svm(test=False, C=0.01):

    data_train = load_data("train", independent=True)
    data_train = add_kraehenbuehl_features(data_train)
    X_features = [x[0] for x in data_train.X]
    X_features_flat = np.vstack(X_features)
    y = np.hstack(data_train.Y)
    if test:
        data_val = load_data("val", independent=True)
        data_val = add_kraehenbuehl_features(data_val)
        X_features_val = [x[0] for x in data_val.X]
        X_features_flat_val = np.vstack(X_features_val)
        y_val = np.hstack(data_val.Y)
        y = np.hstack([y, y_val])
        X_features_flat = np.vstack([X_features_flat, X_features_flat_val])
    from sklearn.svm import LinearSVC
    svm = LinearSVC(C=C, dual=False, class_weight='auto')
    #from sklearn.linear_model import LogisticRegression
    #svm = LogisticRegression(C=.5, dual=False, class_weight='auto')
    #from pystruct.learners import OneSlackSSVM
    #from latent_crf_experiments.magic_svm.svm_definition \
        #import AureliensMagicSVM
    #class_weight = 1. / np.bincount(y)
    #class_weight *= len(class_weight) / np.sum(class_weight)
    #pbl = AureliensMagicSVM(class_weight=class_weight[:-1])
    #svm = OneSlackSSVM(pbl, n_jobs=1, verbose=2, max_iter=100000, C=0.01,
                       #show_loss_every=10, tol=0.00001,
                       #inactive_threshold=1e-4, inactive_window=50,
                       #check_constraints=True, break_on_bad=True)
    svm.fit(X_features_flat[y != 21], y[y != 21])

    eval_on_pixels(data_train, [svm.predict(x) for x in X_features])

    if test:
        data_test = load_data("test", independent=True)
    else:
        data_test = load_data("val", independent=True)

    data_test = add_kraehenbuehl_features(data_test)

    X_features = [x[0] for x in data_test.X]
    y = np.hstack(data_test.Y)
    eval_on_pixels(data_test, [svm.predict(x) for x in X_features])
    plot_results(data_test, [svm.predict(x) for x in X_features],
                 folder="kraehenbuehl_aurelien_linear_svc")

    Tracer()()


def main():
    # load training data
    independent = False
    test = False
    with open("../superpixel_crf/data_probs_train_cw.pickle") as f:
        data_train = cPickle.load(f)

    #with open("../superpixel_crf/data_val_1000_color.pickle") as f:
        #data_val = cPickle.load(f)
    #data_train = load_data("train", independent=independent)
    #data_train = add_kraehenbuehl_features(data_train)
    data_train = add_edges(data_train, independent=independent)
    #data_val = add_edges(data_val)

    data_train = discard_void(data_train, 21)
    #data_train = add_edge_features(data_train)
    X_, Y_ = data_train.X, data_train.Y
    #chi2 = AdditiveChi2Sampler(sample_steps=2)
    #X_ = [(chi2.fit_transform(x[0]), x[1]) for x in X_]

    if test:
        data_val = load_data("val", independent=independent)
        #data_val = add_kraehenbuehl_features(data_val)
        data_val = discard_void(data_val, 21)
        data_val = add_edge_features(data_val)

        X_.extend(data_val.X)
        Y_.extend(data_val.Y)

    n_states = 21
    print("number of samples: %s" % len(data_train.X))
    class_weights = 1. / np.bincount(np.hstack(Y_))
    class_weights *= 21. / np.sum(class_weights)
    #class_weights = np.ones(n_states)
    print(class_weights)
    problem = GraphCRF(n_states=n_states, n_features=X_[0][0].shape[1],
                       inference_method='qpbo', class_weight=class_weights)
    #problem = EdgeFeatureGraphCRF(n_states=n_states, n_features=21 * 6,
                                  #inference_method='qpbo',
                                  #class_weight=class_weights,
                                  #n_edge_features=3,
                                  #symmetric_edge_features=[0, 1],
                                  #antisymmetric_edge_features=[2])
    #ssvm = learners.SubgradientStructuredSVM(
        #problem, verbose=2, C=.001, n_jobs=-1, max_iter=100000,
        #learning_rate=0.00015, show_loss_every=10, decay_exponent=.5,
        #momentum=0.98)
    ssvm = learners.OneSlackSSVM(
        problem, verbose=1, C=0.001, max_iter=100000, n_jobs=-1,
        tol=0.0001, show_loss_every=50, inference_cache=0, cache_tol='auto',
        logger=SaveLogger("probs_pairwise_0.001.pickle",
                          save_every=100),
        inactive_threshold=1e-5, break_on_bad=False)
    #ssvm = SaveLogger("bow_1000_no_edges_chi2_.01.pickle").load()
    #ssvm.logger = SaveLogger(
        #file_name="bow_1000_no_edges_chi2_.01_refit.pickle", save_every=100)
    #ssvm.n_jobs = 5
    #ssvm.problem.class_weight = np.ones(ssvm.problem.n_states)
    #ssvm.problem.inference_method = 'ad3'
    #ssvm.fit(X_, Y_, warm_start=True)
    ssvm.fit(X_, Y_)
    print("fit finished!")
    tracer()

    # do some evaluation on the training set
    print("score on training set: %f" % ssvm.score(X_, Y_))

    # make figures with predictions
    #plot_results(data_train.images, data_train.file_names, data_train.Y,
                 #Y_pred, data_train.all_superpixels, folder="figures_train")

    #data_val = load_data("val", independent=independent)
    #data_val = add_kraehenbuehl_features(data_val)
    X_val_, Y_val_ = discard_void(data_val.X, data_val.Y, 21)
    X_edge_features_val = [(x[0], x[1], np.ones((x[1].shape[0], 1))) for x
                           in data_val.X]

    #print("score on validation set: %f" % ssvm.score(X_val_, Y_val_))
    print("score on validation set: %f" % ssvm.score(X_edge_features_val,
                                                     Y_val_))
    tracer()


if __name__ == "__main__":
    main()
    #train_svm()
    #plot_parts()
