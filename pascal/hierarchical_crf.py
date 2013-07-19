import numpy as np

from sklearn.utils import shuffle
#from sklearn.grid_search import GridSearchCV

from pystruct import learners
from pystruct.utils import SaveLogger
from pystruct.models import LatentNodeCRF

from datasets.pascal import PascalSegmentation
from pascal_helpers import load_pascal, make_cpmc_hierarchy
from latent_crf_experiments.utils import discard_void
from IPython.core.debugger import Tracer
tracer = Tracer()


def svm_on_segments(C=.1, learning_rate=.001, subgradient=False):
    ds = PascalSegmentation()
    # load and prepare data
    data_train = load_pascal("kTrain", sp_type="cpmc")
    data_train = make_cpmc_hierarchy(ds, data_train)
    data_train = discard_void(ds, data_train)
    X_, Y_ = data_train.X, data_train.Y

    n_states = 21
    class_weights = 1. / np.bincount(np.hstack(Y_))
    class_weights *= 21. / np.sum(class_weights)
    experiment_name = ("latent_10_cpmc_%f_qpbo_n_slack" % C)
    logger = SaveLogger(experiment_name + ".pickle", save_every=10)
    model = LatentNodeCRF(n_labels=n_states,
                          n_features=data_train.X[0][0].shape[1],
                          n_hidden_states=10,
                          inference_method='qpbo',
                          class_weight=class_weights,
                          latent_node_features=False)
    if subgradient:
        ssvm = learners.LatentSubgradientSSVM(
            model, C=C, verbose=1, show_loss_every=10, logger=logger,
            n_jobs=-1, learning_rate=learning_rate, decay_exponent=1,
            momentum=0., max_iter=100000, decay_t0=100)
    else:
        latent_logger = SaveLogger("lssvm_" + experiment_name +
                                   "_%d.pickle", save_every=1)
        #base_ssvm = learners.OneSlackSSVM(
            #model, verbose=2, C=C, max_iter=100, n_jobs=-1, tol=0.001,
            #show_loss_every=200, inference_cache=50, logger=logger,
            #cache_tol='auto', inactive_threshold=1e-5, break_on_bad=False,
            #switch_to=('ogm', {'alg': 'dd'}))
        base_ssvm = learners.NSlackSSVM(
            model, verbose=2, C=C, max_iter=10000, n_jobs=10, tol=0.001,
            show_loss_every=20, logger=logger, inactive_threshold=1e-5,
            break_on_bad=False, batch_size=10, inactive_window=10)
        ssvm = learners.LatentSSVM(base_ssvm, logger=latent_logger,
                                   latent_iter=3)
    #warm_start = True
    warm_start = False
    if warm_start:
        ssvm = logger.load()
        ssvm.logger = SaveLogger(experiment_name + "_retrain.pickle",
                                 save_every=10)
        ssvm.max_iter = 10000
        ssvm.decay_exponent = 1
        #ssvm.decay_t0 = 1000
        #ssvm.learning_rate = 0.00001
        #ssvm.momentum = 0

    X_, Y_ = shuffle(X_, Y_)
    #ssvm.fit(data_train.X, data_train.Y)
    ssvm.fit(X_, Y_)
    #H_init = [np.hstack([y, np.random.randint(21, 26)]) for y in Y_]
    #ssvm.fit(X_, Y_, H_init=H_init)
    print("fit finished!")


if __name__ == "__main__":
    #for C in 10. ** np.arange(-5, 2):
    #for lr in 10. ** np.arange(-3, 2)[::-1]:
        #svm_on_segments(C=.01, learning_rate=lr)
    #svm_on_segments(C=.01, learning_rate=0.1)
    svm_on_segments(C=.01, subgradient=False)
    #plot_init()
    #plot_results()
