import numpy as np
from nose.tools import assert_equal

from ..ignore_void_crf import IgnoreVoidCRF

# make simple binary problem
X = np.random.uniform(size=(20, 1, 2))
Y = (X[:, :, 0] > .5).astype(np.int).reshape(20, -1)
# add void around decision boundary
Y[np.abs(X[:, :, 0] - 0.5) < 0.1] = 2
X = [(x, np.empty(shape=(0, 2), dtype=np.int)) for x in X]


def test_energy():
    crf = IgnoreVoidCRF(n_states=2, n_features=2, void_label=2,
                        inference_method='lp')
    # set some random weights, do inference and check that everything is ok
    np.random.seed(0)
    for x, y in zip(X, Y):
        w = np.random.normal(size=crf.size_psi)
        y_hat, energy = crf.inference(x, w, relaxed=True,
                                      return_energy=True)
        energy_svm = np.dot(w, crf.psi(x, y_hat))
        assert_equal(energy_svm, -energy)


def test_loss_augmented_inference():
    crf = IgnoreVoidCRF(n_states=2, n_features=2, void_label=2,
                        inference_method='lp')
    np.random.seed(0)
    for x, y in zip(X, Y):
        w = np.random.normal(size=crf.size_psi)
        y_hat, energy = crf.loss_augmented_inference(x, y, w, relaxed=True,
                                                     return_energy=True)
        assert_equal(np.dot(crf.psi(x, y_hat), w)
                     + crf.continuous_loss(y, y_hat[0]), -energy)
