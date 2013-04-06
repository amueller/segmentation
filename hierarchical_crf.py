import numpy as np
#import matplotlib.pyplot as plt
from scipy import sparse

from sklearn.utils import shuffle

from hierarchical_segmentation import get_segment_features

from msrc_first_try import load_data, plot_results, discard_void

#from sklearn.grid_search import GridSearchCV


from pystruct.problems import GraphCRF
from pystruct import learners
from pystruct.utils import SaveLogger

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


def svm_on_segments():
    X, Y, image_names, images, all_superpixels = load_data(
        "train", independent=False)

    segment_features = [get_segment_features(*stuff)
                        for stuff in zip(X, Y, images, all_superpixels)]
    all_segments, all_features, all_labels, segment_edges =\
        zip(*segment_features)

    all_edges = make_hierarchy_edges(all_segments, all_superpixels)

    X_stacked = [(np.vstack([x[0], feat]), edges)
                 for x, feat, edges in zip(X, all_features, all_edges)]

    Y_stacked = [np.hstack([y, y_]) for y, y_ in zip(Y, all_labels)]

    features, labels = np.vstack(all_features), np.hstack(all_labels)

    X_, Y_ = discard_void(X_stacked, Y_stacked, 21)
    X_org_, Y_org_ = discard_void(X, Y, 21)

    n_states = 21

    problem = GraphCRF(n_states=n_states, n_features=21 * 6,
                       inference_method='dai')
    logger = SaveLogger(save_every=10, file_name="hierarchical.pickle")
    ssvm = learners.SubgradientStructuredSVM(
        problem, verbose=2, C=10, n_jobs=1, max_iter=1000,
        learning_rate=0.0001, show_loss_every=10, decay_exponent=.0,
        momentum=0.0, logger=logger)

    X_, Y_ = shuffle(X_, Y_)
    ssvm.fit(X_, Y_)
    print("fit finished!")

    # do some evaluation on the training set
    print("score on augmented training set: %f" % ssvm.score(X_, Y_))
    # remove edges
    X_org_ = [(x[0], np.zeros((0, 2), dtype=np.int)) for x in X_org_]
    print("score on original training set: %f" % ssvm.score(X_org_, Y_org_))

    X_val, Y_val, image_names_val, images_val, all_superpixels_val = load_data(
        "val", independent=True)

    X_val_, Y_val_ = discard_void(X_val, Y_val, 21)
    print("score on validation set: %f" % ssvm.score(X_val_, Y_val_))
    return
    tracer()
    #all_segments = [segments[sps]
                    #for segments, sps in zip(all_segments, all_superpixels)]

    #plot_results(images, image_names, all_labels, Y_pred, all_segments,
                 #folder="figures_segments", use_colors_predict=True)

    segment_features_val = [get_segment_features(*stuff) for stuff in
                            zip(X_val, Y_val, images_val, all_superpixels_val)]
    all_segments_val, all_features_val, all_labels_val, segment_edges_val =\
        zip(*segment_features_val)
    features_val, labels_val = (np.vstack(all_features_val),
                                np.hstack(all_labels_val))

    print("validation score: %f"
          % ssvm.score(features_val[labels_val != 21],
                       labels_val[labels_val != 21]))
    Y_pred_val = [ssvm.predict(feats) for feats in all_features_val]
    plot_results(images_val, image_names_val, all_labels_val, Y_pred_val,
                 all_segments_val, folder="figures_segments_val",
                 use_colors_predict=True)
    tracer()

if __name__ == "__main__":
    svm_on_segments()
