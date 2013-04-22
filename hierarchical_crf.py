import cPickle
import numpy as np
from collections import namedtuple
#import matplotlib.pyplot as plt
from scipy import sparse

from sklearn.utils import shuffle
#from sklearn.grid_search import GridSearchCV

from pystruct.problems import GraphCRF, LatentNodeCRF
from pystruct import learners
from pystruct.utils import SaveLogger
from pystruct.problems.latent_node_crf import kmeans_init

from hierarchical_segmentation import (get_segment_features,
                                       plot_results_hierarchy)
from msrc_first_try import load_data
from msrc_helpers import discard_void, add_edges

from kraehenbuehl_potentials import add_kraehenbuehl_features

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
                                   'superpixels, segments')


def make_hierarchical_data(data, lateral=False, latent=False):
    from datasets.msrc import MSRCDataset
    msrc = MSRCDataset()
    images = [msrc.get_image(f) for f in data.file_names]
    segment_features = [get_segment_features(*stuff)
                        for stuff in zip(data.X, data.Y,
                                         images, data.superpixels)]
    all_segments, all_features, all_labels, segment_edges =\
        zip(*segment_features)

    all_edges = make_hierarchy_edges(all_segments, data.superpixels)

    if latent:
        #X_stacked = [(np.vstack([x[0], feat]),
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
                                 data.superpixels, all_segments)


def svm_on_segments():
    # load and prepare data
    lateral = True
    latent = True
    test = True
    with open("../superpixel_crf/data_probs_train_cw_trainval.pickle") as f:
        data_train = cPickle.load(f)
    data_train = add_edges(data_train, independent=False)
    data_train = add_kraehenbuehl_features(data_train)
    #data_train = load_data("train", independent=False)
    X_org_ = data_train.X
    data_train = make_hierarchical_data(data_train, lateral=lateral,
                                        latent=latent)
    data_train = discard_void(data_train, 21)
    X_, Y_ = data_train.X, data_train.Y
    # remove edges
    if not lateral:
        X_org_ = [(x[0], np.zeros((0, 2), dtype=np.int)) for x in X_org_]

    if test:
        with open("../superpixel_crf/data_probs_val_cw_trainval.pickle") as f:
            data_val = cPickle.load(f)
        #data_val = load_data("val", independent=independent)
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
    experiment_name = "latent_piecewise_texton_subgradient_100_lr.0001_trainval_bla"
    #experiment_name = "latent_piecewise_texton_subgradient_.1_lr.0001_trainval_BAM"
    logger = SaveLogger(experiment_name + ".pickle", save_every=10)
    #latent_logger = SaveLogger("lssvm_" + experiment_name + "_%d.pickle",
                               #save_every=1)
    if latent:
        problem = LatentNodeCRF(n_labels=n_states, n_features=21 * 2,
                                n_hidden_states=25, inference_method='qpbo' if
                                lateral else 'dai', class_weight=class_weights)
        #base_ssvm = learners.OneSlackSSVM(
            #problem, verbose=1, C=.1, max_iter=100000, n_jobs=-1,
            #tol=0, show_loss_every=200, inference_cache=50, logger=logger,
            #cache_tol='auto', inactive_threshold=1e-5, break_on_bad=False)
        #ssvm = learners.LatentSSVM(base_ssvm, logger=latent_logger)
        #ssvm = learners.LatentSubgradientSSVM(problem, C=.1, verbose=1,
                                              #show_loss_every=10,
                                              #logger=logger, n_jobs=1,
                                              #learning_rate=0.001,
                                              #decay_exponent=0, momentum=0.99,
                                              #max_iter=100000)
        ssvm = logger.load()
        ssvm.logger = SaveLogger(experiment_name + "_retrain2.pickle", save_every=10)
        ssvm.learning_rate = 0.001
    else:
        problem = GraphCRF(n_states=n_states, n_features=21,
                           inference_method='qpbo' if lateral else 'dai',)
                           #class_weight=class_weights)
        ssvm = learners.OneSlackSSVM(
            problem, verbose=1, C=0.001, max_iter=100000, n_jobs=-1,
            tol=0.0001, show_loss_every=200, inference_cache=50, logger=logger,
            cache_tol='auto', inactive_threshold=1e-5, break_on_bad=False)

    #ssvm = logger.load()

    X_, Y_ = shuffle(X_, Y_)
    #ssvm.fit(data_train.X, data_train.Y)
    ssvm.fit(X_, Y_)
    print("fit finished!")
    tracer()

    # do some evaluation on the training set
    print("score on augmented training set: %f" % ssvm.score(data_train.X,
                                                             data_train.Y))
    if not latent:
        print("score on original training set (no void): %f"
              % ssvm.score(X_org_, data_train.Y))

    with open("../superpixel_crf/data_probs_val_cw.pickle") as f:
        data_val = cPickle.load(f)
    data_train = add_edges(data_val, independent=False)
    #data_val = load_data("val", independent=False)
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
    svm_on_segments()
    #plot_init()
    #plot_results()
