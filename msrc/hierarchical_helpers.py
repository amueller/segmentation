
import numpy as np

from msrc_helpers import (DataBunch, load_data, sigm, add_edges,
                          add_kraehenbuehl_features)


def add_top_node(data):
    X_stacked, Y_stacked = [], []
    for x, y in zip(data.X, data.Y):
        new_node = np.max(x[1]) + 1
        n_nodes = len(x[0])
        edges = np.c_[np.arange(n_nodes), np.repeat(new_node, n_nodes)]
        edges_stacked = np.vstack([x[1], edges])

        y_stacked = y
        x_stacked = (x[0], edges_stacked, 1)
        X_stacked.append(x_stacked)
        Y_stacked.append(y_stacked)
    return DataBunch(X_stacked, Y_stacked, data.file_names, data.superpixels)


def load_data_global_probs(dataset="train", latent=False):
    def padded_vstack(blub):
        a, b = blub
        if a.shape[0] > b.shape[0]:
            b = np.hstack([b, np.zeros((b.shape[0], a.shape[1] - b.shape[1]))])
        return np.vstack([a, b])

    data = load_data(dataset=dataset, which="piecewise")
    data = add_kraehenbuehl_features(data, which="train_30px")
    data = add_kraehenbuehl_features(data, which="train")
    data = add_edges(data)
    if latent:
        data = add_top_node(data)
    descs = np.load("/home/user/amueller/checkout/superpixel_crf/"
                    "global_probs_%s.npy" % dataset)
    X = []
    for x, glob_desc in zip(data.X, descs):
        if latent:
            x_ = padded_vstack([x[0],
                                np.repeat(sigm(glob_desc)[np.newaxis, 1:],
                                          x[2], axis=0)])
        else:
            x_ = np.hstack([x[0], np.repeat(sigm(glob_desc)[np.newaxis, :],
                                            x[0].shape[0], axis=0)])
            # add features for latent node
        if len(x) == 3:
            X.append((x_, x[1], x[2]))
        else:
            X.append((x_, x[1]))

    return DataBunch(X, data.Y, data.file_names, data.superpixels)
