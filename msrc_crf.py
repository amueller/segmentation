import cPickle

import numpy as np

from pystruct import learners
#from pystruct.problems import EdgeFeatureGraphCRF
#from pystruct.problems import GraphCRF
import pystruct.problems as crfs
from pystruct.utils import SaveLogger

from msrc_helpers import discard_void, add_edge_features, add_edges
#from msrc_helpers import SimpleSplitCV, concatenate_datasets

from kraehenbuehl_potentials import add_kraehenbuehl_features


from IPython.core.debugger import Tracer
tracer = Tracer()


def main():
    # load training data
    #independent = True
    independent = False
    test = False
    with open("/home/user/amueller/checkout/superpixel_crf/"
              "data_probs_train_cw_trainval.pickle") as f:
    #with open("/home/user/amueller/checkout/superpixel_crf/"
              #"data_train_1000_color.pickle") as f:
        data_train = cPickle.load(f)
    #data_train = load_stacked_results()

    #with open("../superpixel_crf/data_val_1000_color.pickle") as f:
        #data_val = cPickle.load(f)
    #data_train = load_data("train", independent=independent)
    data_train = add_edges(data_train, independent=independent)
    data_train = add_kraehenbuehl_features(data_train)
    data_train = discard_void(data_train, 21)

    if not independent:
        data_train = add_edge_features(data_train)

    if test:
        pass
        #with open("../superpixel_crf/data_probs_val_cw_trainval.pickle") as f:
        #with open("../superpixel_crf/data_val_1000_color.pickle") as f:
            #data_val = cPickle.load(f)
        ##data_val = load_data("val", independent=independent)
        #data_val = add_edges(data_val, independent=independent)
        ##data_val = add_kraehenbuehl_features(data_val)
        #data_val = discard_void(data_val, 21)
        #if not independent:
            #data_val = add_edge_features(data_val)

        #X_.extend(data_val.X)
        #Y_.extend(data_val.Y)

    n_states = 21
    print("number of samples: %s" % len(data_train.X))
    class_weights = 1. / np.bincount(np.hstack(data_train.Y))
    #class_weights[21] = 0
    class_weights *= 21. / np.sum(class_weights)
    #class_weights = np.ones(n_states)
    print(class_weights)
    #problem = crfs.GraphCRF(n_states=n_states, n_features=X_[0][0].shape[1],
                            #inference_method='qpbo',
                            #class_weight=class_weights, rescale_C=True)
    problem = crfs.EdgeFeatureGraphCRF(n_states=n_states,
                                       n_features=data_train.X[0][0].shape[1],
                                       inference_method='qpbo',
                                       class_weight=class_weights,
                                       n_edge_features=3,
                                       symmetric_edge_features=[0, 1],
                                       antisymmetric_edge_features=[2])
    experiment_name = "redo_edge_features_0.05"
    warm_start = True
    #warm_start = False
    #ssvm = learners.SubgradientSSVM(
        #problem, verbose=2, C=0.1, n_jobs=-1, max_iter=100000,
        #learning_rate=0.001, show_loss_every=10, decay_exponent=0.5,
        #momentum=0.0,
        #logger=SaveLogger(experiment_name + ".pickle", save_every=10))
    ssvm = learners.OneSlackSSVM(
        problem, verbose=3, C=.01, max_iter=100000, n_jobs=-1,
        tol=0.001, show_loss_every=50, inference_cache=10, cache_tol='auto',
        logger=SaveLogger(experiment_name + ".pickle", save_every=100),
        inactive_threshold=1e-5, break_on_bad=False, inactive_window=50)

    if warm_start:
        ssvm = SaveLogger(experiment_name + ".pickle").load()
        ssvm.logger = SaveLogger(
            file_name=experiment_name + "_refit.pickle",
            save_every=100)
        ssvm.problem.inference_method = 'ad3'
        ssvm.n_jobs = 1

    ssvm.fit(data_train.X, data_train.Y, warm_start=warm_start)
    print("fit finished!")
    return
    tracer()

    # do some evaluation on the training set
    #print("score on training set: %f" % ssvm.score(X_, Y_))

    # make figures with predictions
    #plot_results(data_train.images, data_train.file_names, data_train.Y,
                 #Y_pred, data_train.all_superpixels, folder="figures_train")

    #data_val = load_data("val", independent=independent)
    #data_val = add_kraehenbuehl_features(data_val)
    #X_val_, Y_val_ = discard_void(data_val.X, data_val.Y, 21)
    #X_edge_features_val = [(x[0], x[1], np.ones((x[1].shape[0], 1))) for x
                           #in data_val.X]

    #print("score on validation set: %f" % ssvm.score(X_val_, Y_val_))
    #print("score on validation set: %f" % ssvm.score(X_edge_features_val,
                                                     #Y_val_))
    #tracer()


if __name__ == "__main__":
    main()
    #train_svm()
    #plot_parts()
