import numpy as np

#from sklearn.svm import SVC
from sklearn.svm import LinearSVC

from msrc_helpers import (discard_void, PixelwiseScorer, concatenate_datasets)
from msrc_helpers import SimpleSplitCV, load_data
from kraehenbuehl_potentials import add_kraehenbuehl_features

from IPython.core.debugger import Tracer
tracer = Tracer()


def train_svm(test=False, C=0.01, gamma=.1, grid=False):
    which = "piecewise"

    data_train = load_data(which=which)
    data_train = add_kraehenbuehl_features(data_train)
    data_train_novoid = discard_void(data_train, 21)

    if grid and test:
        raise ValueError("Don't you dare grid-search on the test-set!")

    svm = LinearSVC(C=C, class_weight='auto')
    #svm = SVC(kernel='rbf', class_weight='auto', C=C, gamma=gamma,
              #shrinking=False, cache_size=5000)

    if grid:
        data_val = load_data('val', which=which)
        data_val = add_kraehenbuehl_features(data_val)
        data_val_novoid = discard_void(data_train, 21)
        n_samples_train = len(np.hstack(data_train_novoid.Y))
        n_samples_val = len(np.hstack(data_val_novoid.Y))
        cv = SimpleSplitCV(n_samples_train, n_samples_val)
        data_trainval = concatenate_datasets(data_train_novoid,
                                             data_val_novoid)
        scorer = PixelwiseScorer(data=data_val)

        from sklearn.grid_search import GridSearchCV
        #param_grid = {'C': 10. ** np.arange(1, 4), 'gamma': 10. **
                      #np.arange(-3, 1)}
        param_grid = {'C': 10. ** np.arange(-3, 3)}
        grid = GridSearchCV(svm, param_grid=param_grid, verbose=10, n_jobs=-1,
                            cv=cv, scoring=scorer)
        grid.fit(np.vstack(data_trainval.X),
                 np.hstack(data_trainval.Y))
        print(grid.best_params_)
        print(grid.best_score_)
    else:
        print(svm)
        svm.fit(np.vstack(data_train_novoid.X), np.hstack(data_train_novoid.Y))
        scorer(svm, data_train.X, data_train.Y)
        if test:
            data_test = load_data("test", which=which)
        else:
            data_test = load_data("val", which=which)
        scorer(svm, data_test.X, data_test.Y)

    return svm


if __name__ == "__main__":
    train_svm(grid=True)
