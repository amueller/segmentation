import numpy as np

from pystruct import learners
import pystruct.models as crfs
from pystruct.utils import SaveLogger

from datasets.pascal import PascalSegmentation
from pascal_helpers import load_pascal
from latent_crf_experiments.utils import (discard_void, add_edges, eval_on_sp,
                                          add_edge_features)


from IPython.core.debugger import Tracer
tracer = Tracer()


def main(C=1, test=False):
    ds = PascalSegmentation()
    # load training data
    independent = False
    #independent = True
    if test:
        data_train = load_pascal("trainval")
    else:
        data_train = load_pascal("kTrain")

    data_train = add_edges(data_train, 'independent' if independent else
                           "pairwise")
    data_train = discard_void(ds, data_train, ds.void_label)

    #data_train = load_data_global_probs()

    if not independent:
        data_train = add_edge_features(ds, data_train)

    n_states = 21
    print("number of samples: %s" % len(data_train.X))
    class_weights = 1. / np.bincount(np.hstack(data_train.Y))
    #class_weights = class_weights[:n_states]
    #class_weights[-1] = 0
    class_weights *= 21. / np.sum(class_weights)
    #class_weights = np.ones(n_states)
    print(class_weights)
    #model = crfs.CrammerSingerSVMModel(n_features=X.shape[1],
                                       #n_classes=n_states,
                                       #class_weight=class_weights)
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
    experiment_name = "edge_features_ksplit_trainval_%f" % C
    #warm_start = True
    warm_start = False
    ssvm = learners.OneSlackSSVM(
        model, verbose=2, C=C, max_iter=1000000, n_jobs=-1,
        tol=0.0001, show_loss_every=50, inference_cache=50, cache_tol='auto',
        logger=SaveLogger(experiment_name + ".pickle", save_every=100),
        inactive_threshold=1e-5, break_on_bad=False, inactive_window=50,
        switch_to_ad3=False)
    #ssvm = learners.SubgradientSSVM(
        #model, verbose=3, C=C, max_iter=10000, n_jobs=-1, show_loss_every=10,
        #logger=SaveLogger(experiment_name + ".pickle", save_every=10),
        #momentum=0, learning_rate=0.001, decay_exponent=1)

    if warm_start:
        ssvm = SaveLogger(experiment_name + ".pickle").load()
        ssvm.logger = SaveLogger(
            file_name=experiment_name + "_refit.pickle",
            save_every=10)
        #ssvm.learning_rate = 0.000001
        ssvm.cache_tol = 0.1
        ssvm.cache_tol_ = 0.1

        #ssvm.model.inference_method = 'ad3'
        #ssvm.n_jobs = 1

    ssvm.fit(data_train.X, data_train.Y, warm_start=warm_start)

    print("fit finished!")
    if test:
        data_val = load_pascal('val')
    else:
        data_val = load_pascal('kVal')
    data_val = add_edges(data_val, 'independent' if independent else
                         "pairwise")
    data_val = add_edge_features(data_val)
    eval_on_sp(data_val, ssvm.predict(data_val.X), print_results=True)

if __name__ == "__main__":
    #for C in 10. ** np.arange(-4, 2):
        #main(C)
    main(1, test=True)
