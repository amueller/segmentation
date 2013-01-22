import numpy as np

from pystruct.problems import GraphCRF


from IPython.core.debugger import Tracer

tracer = Tracer()


class IgnoreVoidCRF(GraphCRF):
    """GraphCRF that ignores nodes with void label in ground truth.
    """
    def __init__(self, n_states=2, n_features=None, inference_method='qpbo',
                 void_label=21):
        GraphCRF.__init__(self, n_states, n_features, inference_method)
        self.void_label = void_label

    def psi(self, x, y):
        """Feature vector associated with instance (x, y).

        Feature representation psi, such that the energy of the configuration
        (x, y) and a weight vector w is given by np.dot(w, psi(x, y)).

        Parameters
        ----------
        x : tuple
            Unary evidence.

        y : ndarray or tuple
            Either y is an integral ndarray, giving
            a complete labeling for x.
            Or it is the result of a linear programming relaxation. In this
            case, ``y=(unary_marginals, pariwise_marginals)``.

        Returns
        -------
        p : ndarray, shape (size_psi,)
            Feature vector associated with state (x, y).

        """
        self._check_size_x(x)
        features, edges = self.get_features(x), self.get_edges(x)
        n_nodes = features.shape[0]

        if isinstance(y, tuple):
            # y is result of relaxation, tuple of unary and pairwise marginals
            unary_marginals, pw = y
            unary_marginals = unary_marginals.reshape(n_nodes, self.n_states)
            # accumulate pairwise
            pw = pw.reshape(-1, self.n_states, self.n_states).sum(axis=0)
        else:
            y = y.reshape(n_nodes)
            gx = np.ogrid[:n_nodes]

            #make one hot encoding
            unary_marginals = np.zeros((n_nodes, self.n_states), dtype=np.int)
            gx = np.ogrid[:n_nodes]
            # nodes with void labels are "not assigned" - don't actually exist
            mask = y != self.void_label
            gx_masked = gx[mask]
            y_masked = y[mask]
            unary_marginals[gx_masked, y_masked] = 1

            ##accumulated pairwise
            pw = np.dot(unary_marginals[edges[:, 0]].T,
                        unary_marginals[edges[:, 1]])

        unaries_acc = np.dot(unary_marginals.T, features)
        pw = pw + pw.T - np.diag(np.diag(pw))  # make symmetric

        psi_vector = np.hstack([unaries_acc.ravel(),
                                pw[np.tri(self.n_states, dtype=np.bool)]])
        return psi_vector

    def loss(self, y, y_hat):
        # hamming loss:
        return np.sum((y != y_hat)[y != self.void_label])

    def continuous_loss(self, y, y_hat):
        # continuous version of the loss
        # y is the result of linear programming
        y_one_hot = np.zeros_like(y_hat)
        mask = y != self.void_label
        gx = np.ogrid[:y.shape[0]]
        gx_masked = gx[mask]
        y_masked = y[mask]
        y_one_hot[gx_masked, y_masked] = 1

        # all entries minus correct ones
        return np.prod(y.shape) - np.sum(y_one_hot * y_hat)
