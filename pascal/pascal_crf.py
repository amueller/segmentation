import numpy as np

from pystruct import learners
import pystruct.models as crfs
from pystruct.utils import SaveLogger

from datasets.pascal import PascalSegmentation
from pascal_helpers import load_pascal
from latent_crf_experiments.utils import (discard_void, add_edges,
                                          add_edge_features, eval_on_sp)


from IPython.core.debugger import Tracer
tracer = Tracer()


def main(C=1, test=False):
    ds = PascalSegmentation()
    # load training data
    edge_type = "extended"
    if test:
        data_train = load_pascal("train")
    else:
        data_train = load_pascal("kTrain")

    data_train = add_edges(data_train, edge_type)
    data_train = add_edge_features(ds, data_train, more_colors=True)
    data_train = discard_void(ds, data_train, ds.void_label)

    n_states = 21
    print("number of samples: %s" % len(data_train.X))
    class_weights = 1. / np.bincount(np.hstack(data_train.Y))
    class_weights *= 21. / np.sum(class_weights)
    print(class_weights)
    #model = crfs.GraphCRF(n_states=n_states,
                          #n_features=data_train.X[0][0].shape[1],
                          #inference_method='qpbo', class_weight=class_weights)
    model = crfs.EdgeFeatureGraphCRF(n_states=n_states,
                                     n_features=data_train.X[0][0].shape[1],
                                     inference_method='qpbo',
                                     class_weight=class_weights,
                                     n_edge_features=7,
                                     symmetric_edge_features=[0, 1, 2, 3, 4,
                                                              5],
                                     antisymmetric_edge_features=[6])
    experiment_name = "extended_edges_val_more_colors_distance%f" % C
    #warm_start = True
    warm_start = False
    ssvm = learners.OneSlackSSVM(
        model, verbose=2, C=C, max_iter=1000000, n_jobs=-1,
        tol=0.0001, show_loss_every=50, inference_cache=50, cache_tol='auto',
        logger=SaveLogger(experiment_name + ".pickle", save_every=100),
        inactive_threshold=1e-5, break_on_bad=False, inactive_window=50,
        switch_to=None)
    #ssvm = learners.SubgradientSSVM(
        #model, verbose=3, C=C, max_iter=10000, n_jobs=-1, show_loss_every=10,
        #logger=SaveLogger(experiment_name + ".pickle", save_every=10),
        #momentum=0, learning_rate=0.1, decay_exponent=1, decay_t0=100)

    if warm_start:
        ssvm = SaveLogger(experiment_name + ".pickle").load()
        ssvm.logger = SaveLogger(
            file_name=experiment_name + "_refit.pickle",
            save_every=10)
        #ssvm.learning_rate = 0.000001

        ssvm.model.inference_method = 'ad3bb'
        #ssvm.n_jobs = 1

    ssvm.fit(data_train.X, data_train.Y, warm_start=warm_start)

    print("fit finished!")
    if test:
        data_val = load_pascal('val')
    else:
        data_val = load_pascal('kVal')

    data_val = add_edges(data_val, edge_type)
    data_val = add_edge_features(ds, data_val, more_colors=True)
    eval_on_sp(ds, data_val, ssvm.predict(data_val.X), print_results=True)

if __name__ == "__main__":
    #for C in 10. ** np.arange(-4, 2):
        #main(C)
    main(0.01, test=True)
