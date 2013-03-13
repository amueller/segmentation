import numpy as np

from pystruct.problems import GraphCRF
from pystruct.inference import inference_dispatch


from IPython.core.debugger import Tracer

tracer = Tracer()


class IgnoreVoidCRF(GraphCRF):
    """GraphCRF that ignores nodes with void label in ground truth.
    """
    def __init__(self, n_states=2, n_features=None, inference_method='qpbo',
                 void_label=21):
        if void_label >= n_states:
            raise ValueError("void_label must be one of the states!")
        GraphCRF.__init__(self, n_states, n_features, inference_method)
        self.void_label = void_label

    def max_loss(self, y):
        return np.sum(y != self.void_label)

    def loss(self, y, y_hat):
        # hamming loss:
        return np.sum((y != y_hat)[y != self.void_label])

    def loss_augmented_inference(self, x, y, w, relaxed=False,
                                 return_energy=False):
        self.inference_calls += 1
        self._check_size_w(w)
        unary_potentials = self.get_unary_potentials(x, w)
        pairwise_potentials = self.get_pairwise_potentials(x, w)
        edges = self.get_edges(x)
        # do loss-augmentation
        for l in np.arange(self.n_states):
            # for each class, decrement features
            # for loss-agumention
            unary_potentials[(y != l) * (y != self.void_label), l] += 1.
            unary_potentials[:, l] += 1. / y.size
        return inference_dispatch(unary_potentials, pairwise_potentials, edges,
                                  self.inference_method, relaxed=relaxed,
                                  return_energy=return_energy)

    def continuous_loss(self, y, y_hat):
        # continuous version of the loss
        # y is the result of linear programming
        mask = y != self.void_label
        return (GraphCRF.continuous_loss(self, y[mask], y_hat[mask])
                + np.sum(y_hat == self.void_label) / y.size)
