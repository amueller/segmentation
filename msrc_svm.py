import numpy as np

from sklearn.svm import LinearSVC
#from sklearn.linear_model import LogisticRegression

from msrc_helpers import (discard_void, PixelwiseScorer, concatenate_datasets)
from msrc_helpers import SimpleSplitCV, load_data
#from msrc_helpers import add_kraehenbuehl_features

from IPython.core.debugger import Tracer
tracer = Tracer()


def train_svm(test=False, C=0.01, gamma=.1, grid=False):
    which = "bow"

    data_train = load_data(which=which)
    #data_train = add_kraehenbuehl_features(data_train, which="trainval")
    data_train_novoid = discard_void(data_train, 21)

    if grid and test:
        raise ValueError("Don't you dare grid-search on the test-set!")

    svm = LinearSVC(C=C, class_weight='auto', multi_class='crammer_singer',
                    dual=False, loss='l1')
    #svm = LogisticRegression(C=C, class_weight='auto')

    if grid:
        #data_val = add_kraehenbuehl_features(data_val, which="trainval")
        data_val = load_data('val', which=which)

        data_val_novoid = discard_void(data_train, 21)
        n_samples_train = len(np.hstack(data_train_novoid.Y))
        n_samples_val = len(np.hstack(data_val_novoid.Y))
        cv = SimpleSplitCV(n_samples_train, n_samples_val)
        data_trainval = concatenate_datasets(data_train_novoid,
                                             data_val_novoid)

        from sklearn.grid_search import GridSearchCV
        #param_grid = {'C': 10. ** np.arange(1, 4), 'gamma': 10. **
                      #np.arange(-3, 1)}
        param_grid = {'C': 10. ** np.arange(-2, 2)}
        scorer = PixelwiseScorer(data=data_val)
        grid = GridSearchCV(svm, param_grid=param_grid, verbose=10, n_jobs=1,
                            cv=cv, scoring=scorer, pre_dispatch=1,
                            refit=False)
        grid.fit(np.vstack(data_trainval.X),
                 np.hstack(data_trainval.Y))
        print(grid.best_params_)
        print(grid.best_score_)
    else:
        print(svm)
        print(np.vstack(data_train_novoid.X).shape)
        svm.fit(np.vstack(data_train_novoid.X), np.hstack(data_train_novoid.Y))
        if test:
            data_test = load_data("test", which=which)
        else:
            data_test = load_data("val", which=which)
        scorer = PixelwiseScorer(data=data_test)
        scorer(svm, None, None)

    return svm


if __name__ == "__main__":
    train_svm(grid=True, C=.5)
