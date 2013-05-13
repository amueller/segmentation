import numpy as np

from pystruct import learners
import pystruct.models as crfs
from pystruct.utils import SaveLogger

from msrc_helpers import (discard_void, add_edge_features, add_edges,
                          load_data, add_kraehenbuehl_features)
#from msrc_helpers import SimpleSplitCV, concatenate_datasets


from IPython.core.debugger import Tracer
tracer = Tracer()


def main(C=1):
    # load training data
    #independent = True
    independent = False
    test = False
    data_train = load_data(which="piecewise")

    data_train = add_edges(data_train, independent=independent)
    data_train = add_kraehenbuehl_features(data_train, which="train_30px")
    data_train = add_kraehenbuehl_features(data_train, which="train")

    if not independent:
        data_train = add_edge_features(data_train)

    data_train = discard_void(data_train, 21)

    if test:
        raise ValueError("grrr")
        ##data_val = load_data("val")
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
    #model = crfs.GraphCRF(n_states=n_states,
                          #n_features=data_train.X[0][0].shape[1],
                          #inference_method='qpbo', class_weight=class_weights)
    model = crfs.EdgeFeatureGraphCRF(n_states=n_states,
                                     n_features=data_train.X[0][0].shape[1],
                                     inference_method='qpbo',
                                     class_weight=class_weights,
                                     n_edge_features=3,
                                     symmetric_edge_features=[0, 1],
                                     antisymmetric_edge_features=[2])
    experiment_name = "all_new_edge_features_C%f" % C
    #warm_start = True
    warm_start = False
    #ssvm = learners.SubgradientSSVM(
        #problem, verbose=2, C=0.1, n_jobs=-1, max_iter=100000,
        #learning_rate=0.001, show_loss_every=10, decay_exponent=0.5,
        #momentum=0.0,
        #logger=SaveLogger(experiment_name + ".pickle", save_every=10))
    ssvm = learners.OneSlackSSVM(
        model, verbose=3, C=C, max_iter=100000, n_jobs=-1,
        tol=0.001, show_loss_every=50, inference_cache=50, cache_tol='auto',
        logger=SaveLogger(experiment_name + ".pickle", save_every=100),
        inactive_threshold=1e-5, break_on_bad=False, inactive_window=50)

    if warm_start:
        ssvm = SaveLogger(experiment_name + ".pickle").load()
        ssvm.logger = SaveLogger(
            file_name=experiment_name + "_refit.pickle",
            save_every=100)
        ssvm.model.inference_method = 'ad3'
        ssvm.n_jobs = 1

    ssvm.fit(data_train.X, data_train.Y, warm_start=warm_start)
    print("fit finished!")
    return
    tracer()


if __name__ == "__main__":
    for C in 10. ** np.arange(-3, 3):
        main(C)
