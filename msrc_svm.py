import cPickle

import numpy as np

from msrc_helpers import (discard_void, eval_on_pixels,
                          transform_chi2, concatenate_datasets)
from msrc_helpers import SimpleSplitCV

#from kraehenbuehl_potentials import add_kraehenbuehl_features


from IPython.core.debugger import Tracer
tracer = Tracer()


def train_svm(test=False, C=0.01, gamma=.1):
    #data_train = load_stacked_results()
    with open("/home/user/amueller/checkout/superpixel_crf/"
              "data_train_1000_color.pickle") as f:
        data_train = cPickle.load(f)

    cv = 3
    if test:
        with open("/home/user/amueller/checkout/superpixel_crf/"
                  "data_val_1000_color.pickle") as f:
            data_val = cPickle.load(f)
        n_samples_train = len(np.hstack(data_train.Y))
        n_samples_val = len(np.hstack(data_val.Y))
        cv = SimpleSplitCV(n_samples_train, n_samples_val)
        data_train = concatenate_datasets(data_train, data_val)

    data_train = transform_chi2(data_train)
    data_train_novoid = discard_void(data_train, 21)
    #data_train = load_data("train", independent=True)
    #data_train = add_kraehenbuehl_features(data_train)
    #X_features = [x[0] for x in data_train.X]
    #from sklearn.svm import LinearSVC
    #svm = LinearSVC(C=C, dual=False,
                    #multi_class='crammer_singer', fit_intercept=False,
                    #verbose=10, class_weight='auto')
    from sklearn.svm import SVC
    svm = SVC(kernel='rbf', class_weight='auto', C=C, gamma=gamma,
              shrinking=False, cache_size=5000)
    print(svm)
    svm.fit(np.vstack(data_train_novoid.X), np.hstack(data_train_novoid.Y))
    #from sklearn.grid_search import GridSearchCV
    #grid = GridSearchCV(svm, param_grid={'C': 10. ** np.arange(1, 4), 'gamma':
                                         #10. ** np.arange(-3, 1)}, verbose=10,
                        #n_jobs=1, cv=cv)
    #grid.fit(X_features_flat, y)

    eval_on_pixels(data_train, [svm.predict(x) for x in
                                data_train.X])

    if test:
        #data_test = load_data("test", independent=True)
        with open("/home/user/amueller/checkout/superpixel_crf/"
                  "data_test_1000_color.pickle") as f:
            data_test = cPickle.load(f)
    else:
        #data_test = load_data("val", independent=True)
        #data_test = load_stacked_results('val')
        with open("/home/user/amueller/checkout/superpixel_crf/"
                  "data_val_1000_color.pickle") as f:
            data_test = cPickle.load(f)

    #data_test = add_kraehenbuehl_features(data_test)

    #X_features = [x[0] for x in data_test.X]
    data_test = transform_chi2(data_test)
    eval_on_pixels(data_test, [svm.predict(x) for x in
                               data_test.X])
    #plot_results(data_test, [svm.predict(x) for x in X_features],
                 #folder="probs_100_linear_svc_0.1")
    return svm


if __name__ == "__main__":
    train_svm()
