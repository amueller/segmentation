import numpy as np

from numpy.testing.utils import assert_array_equal
from nose.tools import assert_almost_equal, assert_true

from pystruct.learners import SubgradientStructuredSVM
from pystruct.utils import find_constraint, exhaustive_inference

from ..ignore_void_crf import IgnoreVoidCRF

from IPython.core.debugger import Tracer
tracer = Tracer()

# make simple binary problem
X = np.random.uniform(size=(20, 3, 2))
Y = (X[:, :, 0] > .5).astype(np.int)
# add void around decision boundary
Y[np.abs(X[:, :, 0] - 0.5) < 0.1] = 2
X = [(x, np.empty(shape=(0, 2), dtype=np.int)) for x in X]


def test_inference():
    crf = IgnoreVoidCRF(n_states=3, n_features=2, void_label=2,
                        inference_method='lp')
    # set some random weights, do inference and check that everything is ok
    np.random.seed(0)
    for x, y in zip(X, Y):
        w = np.random.normal(size=crf.size_psi)
        y_hat, energy = crf.inference(x, w, relaxed=True,
                                      return_energy=True)
        energy_svm = np.dot(w, crf.psi(x, y_hat))
        assert_almost_equal(energy_svm, -energy)
        y_hat_exhaustive = exhaustive_inference(crf, x, w)

        y_hat = crf.inference(x, w)
        assert_array_equal(y_hat, y_hat_exhaustive)


def test_loss_augmented_inference():
    crf = IgnoreVoidCRF(n_states=3, n_features=2, void_label=2,
                        inference_method='lp')
    np.random.seed(0)
    for x, y in zip(X, Y):
        w = np.random.normal(size=crf.size_psi)
        y_hat, energy = crf.loss_augmented_inference(x, y, w, relaxed=True,
                                                     return_energy=True)
        assert_almost_equal(np.dot(crf.psi(x, y_hat), w) +
                            crf.continuous_loss(y, y_hat[0]), -energy)


def test_learning():
    crf = IgnoreVoidCRF(n_states=3, n_features=2, void_label=2,
                        inference_method='lp')
    ssvm = SubgradientStructuredSVM(crf, verbose=10, C=100, n_jobs=1,
                                    max_iter=50, learning_rate=0.01)
    ssvm.fit(X, Y)

    for x in X:
        y_hat_exhaustive = exhaustive_inference(crf, x, ssvm.w)
        y_hat = crf.inference(x, ssvm.w)
        assert_array_equal(y_hat, y_hat_exhaustive)

    constr = [find_constraint(crf, x, y, ssvm.w, y_hat=y_hat)
              for x, y, y_hat in zip(X, Y, ssvm.predict(X))]
    losses = [c[3] for c in constr]
    slacks = [c[2] for c in constr]
    assert_true(np.all(np.array(slacks) >= np.array(losses)))
