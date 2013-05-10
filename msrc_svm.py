import numpy as np

from msrc_helpers import (discard_void, eval_on_pixels,
                          transform_chi2, concatenate_datasets)
from msrc_helpers import SimpleSplitCV, load_data
#from kraehenbuehl_potentials import add_kraehenbuehl_features

from IPython.core.debugger import Tracer
tracer = Tracer()


def train_svm(test=False, C=0.01, gamma=.1, grid=False):
    data_train = load_data()

    cv = 3
    if test:
        data_val = load_data('val')
        n_samples_train = len(np.hstack(data_train.Y))
        n_samples_val = len(np.hstack(data_val.Y))
        cv = SimpleSplitCV(n_samples_train, n_samples_val)
        data_train = concatenate_datasets(data_train, data_val)

    data_train = transform_chi2(data_train)
    data_train_novoid = discard_void(data_train, 21)

    from sklearn.svm import SVC
    svm = SVC(kernel='rbf', class_weight='auto', C=C, gamma=gamma,
              shrinking=False, cache_size=5000)

    if grid:
        from sklearn.grid_search import GridSearchCV
        grid = GridSearchCV(svm, param_grid={'C': 10. ** np.arange(1, 4),
                                             'gamma': 10. ** np.arange(-3, 1)},
                            verbose=10, n_jobs=1, cv=cv)
        grid.fit(np.vstack(data_train_novoid.X),
                 np.hstack(data_train_novoid.Y))
    else:
        print(svm)
        svm.fit(np.vstack(data_train_novoid.X), np.hstack(data_train_novoid.Y))

        eval_on_pixels(data_train, [svm.predict(x) for x in data_train.X])

    if test:
        data_test = load_data("test")
    else:
        data_test = load_data("val")

    data_test = transform_chi2(data_test)
    eval_on_pixels(data_test, [svm.predict(x) for x in
                               data_test.X])
    return svm


if __name__ == "__main__":
    train_svm()
