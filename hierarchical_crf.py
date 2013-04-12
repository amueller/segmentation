import numpy as np
from collections import namedtuple
#import matplotlib.pyplot as plt
from scipy import sparse

#from sklearn.utils import shuffle
#from sklearn.grid_search import GridSearchCV

from pystruct.problems import GraphCRF, LatentNodeCRF
from pystruct import learners
from pystruct.utils import SaveLogger
from pystruct.problems.latent_node_crf import kmeans_init

from hierarchical_segmentation import (get_segment_features,
                                       plot_results_hierarchy)
from msrc_first_try import load_data
from msrc_helpers import discard_void

from IPython.core.debugger import Tracer

tracer = Tracer()


def make_hierarchy_edges(segments, superpixels):
    all_edges = []
    for seg, sps in zip(segments, superpixels):
        seg = seg[sps]
        edges = np.vstack([seg.ravel() + sps.max() + 1, sps.ravel()])
        edge_matrix = sparse.coo_matrix((np.ones(edges.shape[1]), edges))
        # make edges unique
        edges = np.vstack(edge_matrix.tocsr().nonzero()).T
        all_edges.append(edges)
    return all_edges


HierarchicalDataBunch = namedtuple('HierarchicalDataBunch', 'X, Y, file_names,'
                                   'images, superpixels, segments')


def make_hierarchical_data(data, lateral=False, latent=False):
    segment_features = [get_segment_features(*stuff)
                        for stuff in zip(data.X, data.Y,
                                         data.images,
                                         data.superpixels)]
    all_segments, all_features, all_labels, segment_edges =\
        zip(*segment_features)

    all_edges = make_hierarchy_edges(all_segments, data.superpixels)

    if latent:
        X_stacked = [(x[0],
                      np.vstack([x[1], edges] if lateral else edges),
                      len(feat))
                     for x, feat, edges in zip(data.X, all_features,
                                               all_edges)]
        Y_stacked = data.Y
    else:
        X_stacked = [(np.vstack([x[0], feat]),
                      np.vstack([x[1], edges] if lateral else edges))
                     for x, feat, edges in zip(data.X, all_features,
                                               all_edges)]
        Y_stacked = [np.hstack([y, y_]) for y, y_ in zip(data.Y, all_labels)]

    return HierarchicalDataBunch(X_stacked, Y_stacked, data.file_names,
                                 data.images, data.superpixels, all_segments)


def svm_on_segments():
    # load and prepare data
    lateral = True
    latent = True
    data_train = load_data("train", independent=False)
    X_org_ = data_train.X
    data_train = make_hierarchical_data(data_train, lateral=lateral,
                                        latent=latent)
    X_, Y_ = discard_void(data_train.X, data_train.Y, 21)
    # remove edges
    if not lateral:
        X_org_ = [(x[0], np.zeros((0, 2), dtype=np.int)) for x in X_org_]
    n_states = 21
    #class_weights = 1. / np.bincount(np.hstack(Y_))
    #class_weights[-1] = 0
    #class_weights *= 21. / np.sum(class_weights)

    logger = SaveLogger("test_latent_lateral.001_2.pickle", save_every=100)
    latent_logger = SaveLogger("lssvm_latent_lateral.001_2_%d.pickle",
                               save_every=1)
    if latent:
        problem = LatentNodeCRF(n_labels=n_states, n_features=21 * 6,
                                n_hidden_states=25,
                                inference_method='qpbo' if lateral else 'dai',)
                                #class_weight=class_weights)
        ssvm = learners.LatentSSVM(learners.OneSlackSSVM(
            problem, verbose=1, C=0.01, max_iter=100000, n_jobs=-1,
            tol=0.0001, show_loss_every=200, inference_cache=50, logger=logger,
            cache_tol='auto', inactive_threshold=1e-5, break_on_bad=False),
            logger=latent_logger)
    else:
        problem = GraphCRF(n_states=n_states, n_features=21 * 6,
                           inference_method='qpbo' if lateral else 'dai',)
                           #class_weight=class_weights)
        ssvm = learners.OneSlackSSVM(
            problem, verbose=1, C=0.0001, max_iter=100000, n_jobs=-1,
            tol=0.0001, show_loss_every=200, inference_cache=50, logger=logger,
            cache_tol='auto', inactive_threshold=1e-5, break_on_bad=False)

    #ssvm = logger.load()

    #X_, Y_ = shuffle(X_, Y_)
    ssvm.fit(X_, Y_)
    print("fit finished!")
    tracer()

    # do some evaluation on the training set
    print("score on augmented training set: %f" % ssvm.score(X_, Y_))
    if not latent:
        print("score on original training set (no void): %f"
              % ssvm.score(X_org_, data_train.Y))

    data_val = load_data("val", independent=False)
    X_val_org = data_val.X
    if not lateral:
        X_val_org = [(x[0], np.empty((0, 2), dtype=np.int)) for x in X_val_org]

    if not latent:
        print("score on original validation set: %f"
              % ssvm.score(X_val_org, data_val.Y))
    # load and prepare data
    data_val = make_hierarchical_data(data_val, lateral=lateral, latent=latent)

    X_val_, Y_val_ = discard_void(data_val.X, data_val.Y, 21)
    print("validation score on augmented validation set: %f"
          % ssvm.score(X_val_, Y_val_))

    #plot_results(images_val, image_names_val, all_labels_val, Y_pred_val,
                 #all_segments_val, folder="figures_segments_val",
                 #use_colors_predict=True)
    tracer()


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
    #svm_on_segments()
    #plot_init()
    plot_results()
