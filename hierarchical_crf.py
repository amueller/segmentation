import numpy as np
#import matplotlib.pyplot as plt
from scipy import sparse

#from sklearn.utils import shuffle

from hierarchical_segmentation import get_segment_features

from msrc_first_try import load_data, discard_void

#from sklearn.grid_search import GridSearchCV


from pystruct.problems import GraphCRF, LatentNodeCRF
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


def make_hierarchical_data(data, lateral=False, latent=False):
    segment_features = [get_segment_features(*stuff)
                        for stuff in zip(data.X, data.Y,
                                         data.images,
                                         data.superpixels)]
    all_segments, all_features, all_labels, segment_edges =\
        zip(*segment_features)

    all_edges = make_hierarchy_edges(all_segments, data.superpixels)

    if latent:
        X_stacked = [(np.vstack([x[0], feat]),
                      np.vstack([x[1], edges] if lateral else edges))
                     for x, feat, edges in zip(data.X, all_features,
                                               all_edges)]
    else:
        X_stacked = [(np.vstack([x[0], feat]),
                      np.vstack([x[1], edges] if lateral else edges),
                      feat.shape[0])
                     for x, feat, edges in zip(data.X, all_features,
                                               all_edges)]

    Y_stacked = [np.hstack([y, y_]) for y, y_ in zip(data.Y, all_labels)]

    return X_stacked, Y_stacked


def svm_on_segments():
    # load and prepare data
    lateral = False
    latent = True
    data_train = load_data("train", independent=False)
    X_stacked, Y_stacked = make_hierarchical_data(data_train, lateral=lateral,
                                                  latent=latent)
    X_, Y_ = discard_void(X_stacked, Y_stacked, 21)
    X_org_, Y_org_ = discard_void(data_train.X, data_train.Y, 21)
    if lateral:
        X_org_ = [(x[0], np.empty((0, 2), dtype=np.int)) for x in X_org_]

    n_states = 21

    if latent:
        problem = LatentNodeCRF(n_states=n_states, n_features=21 * 6,
                                inference_method='qpbo' if lateral else 'dai')
        logger = SaveLogger("hierarchy_lateral_0.0001.pickle", save_every=100)
        ssvm = learners.LatentSSVM(
            problem, verbose=2, C=0.0001, max_iter=100000, n_jobs=-1,
            tol=0.0001, show_loss_every=200, inference_cache=50, logger=logger,
            cache_tol='auto', inactive_threshold=1e-5, break_on_bad=False,
            base_svm='1-slack')
    else:
        problem = GraphCRF(n_states=n_states, n_features=21 * 6,
                           inference_method='qpbo' if lateral else 'dai')
        logger = SaveLogger("hierarchy_lateral_0.0001.pickle", save_every=100)
        ssvm = learners.OneSlackSSVM(
            problem, verbose=2, C=0.0001, max_iter=100000, n_jobs=-1,
            tol=0.0001, show_loss_every=200, inference_cache=50, logger=logger,
            cache_tol='auto', inactive_threshold=1e-5, break_on_bad=False)

    #ssvm = logger.load()

    #X_, Y_ = shuffle(X_, Y_)
    ssvm.fit(X_, Y_)
    print("fit finished!")
    #tracer()

    # do some evaluation on the training set
    print("score on augmented training set: %f" % ssvm.score(X_, Y_))
    # remove edges
    X_org_ = [(x[0], np.zeros((0, 2), dtype=np.int)) for x in X_org_]
    print("score on original training set: %f" % ssvm.score(X_org_, Y_org_))

    data_val = load_data("val", independent=False)

    X_val_org, Y_val_org = discard_void(data_val.X, data_val.Y, 21)

    if lateral:
        X_val_org = [(x[0], np.empty((0, 2), dtype=np.int)) for x in X_val_org]
    print("score on original validation set: %f"
          % ssvm.score(X_val_org, Y_val_org))
    # load and prepare data
    X_stacked_val, Y_stacked_val = make_hierarchical_data(data_val,
                                                          lateral=lateral,
                                                          latent=latent)
    X_val_, Y_val_ = discard_void(X_stacked_val, Y_stacked_val, 21)

    print("validation score on augmented validation set: %f"
          % ssvm.score(X_val_, Y_val_))

    #plot_results(images_val, image_names_val, all_labels_val, Y_pred_val,
                 #all_segments_val, folder="figures_segments_val",
                 #use_colors_predict=True)
    tracer()

if __name__ == "__main__":
    svm_on_segments()
