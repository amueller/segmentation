import numpy as np

from sklearn.utils import shuffle
#from sklearn.grid_search import GridSearchCV

from pystruct.models import EdgeFeatureGraphCRF, LatentNodeCRF
from pystruct import learners
from pystruct.utils import SaveLogger
from pystruct.models.latent_node_crf import kmeans_init

from hierarchical_segmentation import plot_results_hierarchy
from hierarchical_helpers import make_hierarchical_data
from msrc_helpers import (discard_void, add_edges, load_data,
                          add_kraehenbuehl_features)  # , add_edge_features)

from IPython.core.debugger import Tracer
tracer = Tracer()


def svm_on_segments(C=.1, learning_rate=.001, subgradient=True):
    # load and prepare data
    lateral = True
    latent = True
    test = False
    data_train = load_data(which="piecewise")
    data_train = add_edges(data_train, independent=False)
    data_train = add_kraehenbuehl_features(data_train, which="train_30px")
    data_train = add_kraehenbuehl_features(data_train, which="train")
    #if lateral:
        #data_train = add_edge_features(data_train)
    X_org_ = data_train.X
    data_train = make_hierarchical_data(data_train, lateral=lateral,
                                        latent=latent)
    data_train = discard_void(data_train, 21)
    X_, Y_ = data_train.X, data_train.Y
    # remove edges
    if not lateral:
        X_org_ = [(x[0], np.zeros((0, 2), dtype=np.int)) for x in X_org_]

    if test:
        data_val = load_data('val', which="piecewise")
        data_val = add_edges(data_val, independent=False)
        data_val = add_kraehenbuehl_features(data_val)
        data_val = make_hierarchical_data(data_val, lateral=lateral,
                                          latent=latent)
        data_val = discard_void(data_val, 21)

        X_.extend(data_val.X)
        Y_.extend(data_val.Y)

    n_states = 21
    class_weights = 1. / np.bincount(np.hstack(Y_))
    class_weights *= 21. / np.sum(class_weights)
    experiment_name = ("latent25_subgradient_hierarchical_C%f_lr%f"
                       % (C, learning_rate))
    logger = SaveLogger(experiment_name + ".pickle", save_every=10)
    if latent:
        model = LatentNodeCRF(n_labels=n_states,
                              n_features=data_train.X[0][0].shape[1],
                              n_hidden_states=25, inference_method='qpbo' if
                              lateral else 'dai', class_weight=class_weights)
        if subgradient:
            ssvm = learners.LatentSubgradientSSVM(
                model, C=C, verbose=1, show_loss_every=10, logger=logger,
                n_jobs=-1, learning_rate=learning_rate, decay_exponent=0,
                momentum=0.9, max_iter=200)
        else:
            latent_logger = SaveLogger("lssvm_" + experiment_name +
                                       "_%d.pickle", save_every=10)
            base_ssvm = learners.OneSlackSSVM(
                model, verbose=2, C=C, max_iter=100000, n_jobs=1,
                tol=0, show_loss_every=200, inference_cache=50, logger=logger,
                cache_tol='auto', inactive_threshold=1e-5, break_on_bad=False)
            ssvm = learners.LatentSSVM(base_ssvm, logger=latent_logger)
        #ssvm = logger.load()
        #ssvm.logger = SaveLogger(experiment_name + "_retrain2.pickle",
                                 #save_every=10)
        #ssvm.learning_rate = 0.001
    else:
        #model = GraphCRF(n_states=n_states,
                         #n_features=data_train.X[0][0].shape[1],
                         #inference_method='qpbo' if lateral else 'dai',
                         #class_weight=class_weights)
        model = EdgeFeatureGraphCRF(n_states=n_states,
                                    n_features=data_train.X[0][0].shape[1],
                                    inference_method='qpbo' if lateral else
                                    'dai', class_weight=class_weights,
                                    n_edge_features=4,
                                    symmetric_edge_features=[0, 1],
                                    antisymmetric_edge_features=[2])
        ssvm = learners.OneSlackSSVM(
            model, verbose=2, C=C, max_iter=100000, n_jobs=-1,
            tol=0.0001, show_loss_every=200, inference_cache=50, logger=logger,
            cache_tol='auto', inactive_threshold=1e-5, break_on_bad=False)

    #ssvm = logger.load()

    X_, Y_ = shuffle(X_, Y_)
    #ssvm.fit(data_train.X, data_train.Y)
    ssvm.fit(X_, Y_)
    print("fit finished!")


def plot_init():
    data = load_data("train", independent=False)
    data = make_hierarchical_data(data, lateral=False, latent=True)
    #X, Y = discard_void(data.X, data.Y, 21)
    #data.X, data.Y = X, Y
    H = kmeans_init(data.X, data.Y, n_labels=22, n_hidden_states=22)
    plot_results_hierarchy(data, H)


def plot_results():
    data = load_data("val", independent=False)
    data = make_hierarchical_data(data, lateral=False, latent=True)
    logger = SaveLogger("test_latent_2.0001.pickle", save_every=100)
    ssvm = logger.load()
    plot_results_hierarchy(data, ssvm.predict(data.X),
                           folder="latent_results_val_50_states_no_lateral")


if __name__ == "__main__":
    #for C in 10. ** np.arange(-5, 2):
    for lr in 10. ** np.arange(-5, -2):
        svm_on_segments(C=.01, learning_rate=lr)
    #svm_on_segments(C=.1)
    #plot_init()
    #plot_results()
