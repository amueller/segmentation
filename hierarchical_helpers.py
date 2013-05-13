import numpy as np
from collections import namedtuple
from scipy import sparse

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


def make_hierarchical_data(data, lateral=False, latent=False):
    from datasets.msrc import MSRCDataset
    msrc = MSRCDataset()
    images = [msrc.get_image(f) for f in data.file_names]
    segment_features = [get_segment_features(*stuff)
                        for stuff in zip(data.X, data.Y,
                                         images, data.superpixels)]
    all_segments, all_features, all_labels, segment_edges =\
        zip(*segment_features)

    all_edges = make_hierarchy_edges(all_segments, data.superpixels)

    if latent:
        #X_stacked = [(np.vstack([x[0], feat]),
        X_stacked = [(x[0],
                      np.vstack([x[1], edges] if lateral else edges),
                      len(feat))
                     for x, feat, edges in zip(data.X, all_features,
                                               all_edges)]
        Y_stacked = data.Y
    else:
        #X_stacked = [(np.vstack([x[0], feat]),
                      #np.vstack([x[1], edges] if lateral else edges))
                     #for x, feat, edges in zip(data.X, all_features,
                                               #all_edges)]
        # edge features:
        X_stacked = [(np.vstack([x[0], feat]),
                      np.vstack([x[1], edges] if lateral else edges),
                      np.vstack([np.hstack([x[2], np.zeros((len(x[1]), 1))]),
                                 np.hstack([np.zeros((len(edges),
                                                      x[2].shape[1])),
                                            np.ones((len(edges), 1))])]))
                     for x, feat, edges in zip(data.X, all_features,
                                               all_edges)]
        Y_stacked = [np.hstack([y, y_]) for y, y_ in zip(data.Y, all_labels)]

    return HierarchicalDataBunch(X_stacked, Y_stacked, data.file_names,
                                 data.superpixels, all_segments)
