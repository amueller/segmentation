import numpy as np

from pystruct import learners
import pystruct.models as crfs
from pystruct.utils import SaveLogger

from datasets.nyu import NYUSegmentation
from nyu_helpers import load_nyu
from latent_crf_experiments.utils import discard_void, add_edge_features, add_edges
#from hierarchical_helpers import load_data_global_probs
#from msrc_helpers import SimpleSplitCV, concatenate_datasets


from IPython.core.debugger import Tracer
tracer = Tracer()


def main(C=1, test=False):
    dataset = NYUSegmentation()
    # load training data
    data_train = load_nyu()
    data_train = add_edges(data_train)
    data_train = add_edge_features(dataset, data_train, depth_diff=True)

    data_train = discard_void(dataset, data_train)

    n_states = 4.
    print("number of samples: %s" % len(data_train.X))
    class_weights = 1. / np.bincount(np.hstack(data_train.Y))
    class_weights *= n_states / np.sum(class_weights)
    #class_weights = np.ones(n_states)
    print(class_weights)
    #model = crfs.GraphCRF(n_states=n_states,
                          #n_features=data_train.X[0][0].shape[1],
                          #inference_method='qpbo', class_weight=class_weights)
    model = crfs.EdgeFeatureGraphCRF(inference_method='qpbo',
                                     class_weight=class_weights,
                                     n_edge_features=4,
                                     symmetric_edge_features=[0, 1, 3],
                                     antisymmetric_edge_features=[2])
    experiment_name = "edge_features_depth_ad3_%f" % C
    ssvm = learners.OneSlackSSVM(
        model, verbose=2, C=C, max_iter=100000, n_jobs=-1,
        tol=0.001, show_loss_every=10, inference_cache=50, cache_tol='auto',
        logger=SaveLogger(experiment_name + ".pickle", save_every=100),
        inactive_threshold=1e-5, break_on_bad=False, inactive_window=50,
        switch_to=("ad3", {'branch_and_bound':True}))

    ssvm.fit(data_train.X, data_train.Y)
    print("fit finished!")
    return


if __name__ == "__main__":
    #for C in 10. ** np.arange(-4, 2):
        #main(C)
    main(.1, test=False)
