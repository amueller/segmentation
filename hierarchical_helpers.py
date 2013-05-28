import numpy as np
from collections import namedtuple
from scipy import sparse

from msrc_helpers import DataBunch, load_data, sigm, add_edges
from hierarchical_segmentation import get_segment_features

HierarchicalDataBunch = namedtuple('HierarchicalDataBunch', 'X, Y, file_names,'
                                   'superpixels, segments')


def make_hierarchy_edges(segments, superpixels):
    all_edges = []
    for seg, sps in zip(segments, superpixels):
        seg = seg[sps]
        edges = np.vstack([seg.ravel() + sps.max() + 1, sps.ravel()])
        edge_matrix = sparse.coo_matrix((np.ones(edges.shape[1]), edges))
        # make edges unique
        edges = np.vstack(edge_matrix.tocsr().nonzero()).T
        all_edges.append(edges)
    return all_edges


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
    data = load_data(dataset=dataset, which="piecewise")
    data = add_edges(data)
    if latent:
        data = add_top_node(data)
    descs = np.load("/home/user/amueller/checkout/superpixel_crf/"
                    "global_probs_%s.npy" % dataset)
    X = []
    for x, glob_desc in zip(data.X, descs):
        if latent:
            x_ = np.vstack([x[0], np.repeat(sigm(glob_desc)[np.newaxis, :],
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


def make_hierarchical_data(data, lateral=False, latent=False,
                           latent_lateral=False):
    from datasets.msrc import MSRCDataset
    msrc = MSRCDataset()
    images = [msrc.get_image(f) for f in data.file_names]
    segment_features = [get_segment_features(*stuff)
                        for stuff in zip(data.X, data.Y,
                                         images, data.superpixels)]
    all_segments, all_features, all_labels, segment_edges =\
        zip(*segment_features)

    all_edges = make_hierarchy_edges(all_segments, data.superpixels)

    X_stacked, Y_stacked = [], []
    for x, y, feat, edges, labels in zip(data.X, data.Y, all_features,
                                         all_edges, all_labels):
        edges_stacked = np.vstack([x[1], edges] if lateral else edges)

        if latent:
            y_stacked = y
            n_nodes = len(x[0])
            if latent_lateral:
                hierarchy = sparse.csr_matrix(
                    (np.ones(len(edges)), edges.T), shape=(n_nodes + len(feat),
                                                           n_nodes))
                visible_lateral = sparse.csr_matrix(
                    (np.ones(len(x[1])), x[1].T), shape=(n_nodes, n_nodes))
                graph_latent_lateral = (hierarchy * visible_lateral *
                                        hierarchy.T)
                # make symmetric
                graph_latent_lateral = (graph_latent_lateral +
                                        graph_latent_lateral.T)
                edges_latent_lateral = np.c_[graph_latent_lateral.nonzero()]
                # remove self-edges and make sorted
                edges_latent_lateral = \
                    edges_latent_lateral[edges_latent_lateral[:, 0] <
                                         edges_latent_lateral[:, 1]]
                edges_stacked = np.vstack([edges_stacked,
                                           edges_latent_lateral])
            x_stacked = (x[0], edges_stacked, len(feat))
        else:
            if latent_lateral:
                raise ValueError("wut?")
            feat = np.vstack(x[0], feat)
            y_stacked = np.hstack([y, labels])
            x_stacked = (feat, edges_stacked)

        X_stacked.append(x_stacked)
        Y_stacked.append(y_stacked)

    return HierarchicalDataBunch(X_stacked, Y_stacked, data.file_names,
                                 data.superpixels, all_segments)
