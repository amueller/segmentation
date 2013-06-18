from collections import namedtuple
import numbers
import itertools

import numpy as np

from sklearn.externals.joblib import Memory
from pystruct.utils import make_grid_edges

DataBunch = namedtuple('DataBunch', 'X, Y, file_names, superpixels')

memory = Memory(cachedir="/tmp/cache")


@memory.cache
def discard_void(data, void_label=21, latent_features=False):
    if isinstance(data.X[0], np.ndarray):
        X_new = [x[y != void_label] for x, y in zip(data.X, data.Y)]
        Y_new = [y[y != void_label] for y in data.Y]
        return DataBunch(X_new, Y_new, data.file_names,
                         data.superpixels)
    X_new, Y_new = [], []
    for x, y in zip(data.X, data.Y):
        mask = y != void_label
        voids = np.where(~mask)[0]

        if len(x) == 2:
            features, edges = x
        elif len(x) == 3:
            if isinstance(x[2], numbers.Integral):
                features, edges, n_hidden = x
                mask = np.hstack([mask, np.ones(n_hidden, dtype=np.bool)])
            else:
                features, edges, edge_features = x
                edge_features_new = edge_features
        else:
            raise ValueError("len(x) is weird: %d" % len(data.X[0]))

        edges_new = edges
        if edges_new.shape[0] > 0:
            # if there are no edges, don't need to filter them
            # also, below code would break ;)
            for void_node in voids:
                involves_void_node = np.any(edges_new == void_node, axis=1)
                edges_new = edges_new[~involves_void_node]
                if len(x) == 3 and not isinstance(x[2], numbers.Integral):
                    edge_features_new = edge_features_new[~involves_void_node]

        reindex_edges = np.zeros(len(mask), dtype=np.int)
        reindex_edges[mask] = np.arange(np.sum(mask))
        edges_new = reindex_edges[edges_new]
        if len(x) == 2:
            X_new.append((features[mask], edges_new))
            Y_new.append(y[mask])
        else:
            if isinstance(x[2], numbers.Integral):
                n_hidden_new = np.max(edges_new) - np.sum(mask[:-n_hidden]) + 1
                if latent_features:
                    X_new.append((features[mask], edges_new, n_hidden_new))
                else:
                    X_new.append((features[mask[:-n_hidden]], edges_new,
                                  n_hidden_new))
                Y_new.append(y[mask[:-n_hidden]])
                #X_new.append((features[mask], edges_new, n_hidden_new))
                #Y_new.append(y[mask[:-n_hidden]])
            else:
                X_new.append((features[mask], edges_new, edge_features_new))
                Y_new.append(y[mask])

    return DataBunch(X_new, Y_new, data.file_names, data.superpixels)


@memory.cache
def add_edges(data, independent=False, fully_connected=False):
    # generate graph
    if independent:
        X_new = [(x, np.empty((0, 2), dtype=np.int)) for x in data.X]
    else:
        if fully_connected:
            X_new = [(x, np.vstack([e for e in
                                    itertools.combinations(np.arange(len(x)),
                                                           2)]))
                     for x in data.X]
        else:
            X_new = [(x, region_graph(sp))
                     for x, sp in zip(data.X, data.superpixels)]

    return DataBunch(X_new, data.Y, data.file_names, data.superpixels)


def region_graph(regions):
    edges = make_grid_edges(regions)
    n_vertices = np.max(regions) + 1

    crossings = edges[regions.ravel()[edges[:, 0]]
                      != regions.ravel()[edges[:, 1]]]
    edges = regions.ravel()[crossings]
    edges = np.sort(edges, axis=1)
    crossing_hash = (edges[:, 0] + n_vertices * edges[:, 1])
    # find unique connections
    unique_hash = np.unique(crossing_hash)
    # undo hashing
    unique_crossings = np.c_[unique_hash % n_vertices,
                             unique_hash // n_vertices]
    return unique_crossings


def get_edge_contrast(edges, image, superpixels):
    r = np.bincount(superpixels.ravel(), weights=image[:, :, 0].ravel())
    g = np.bincount(superpixels.ravel(), weights=image[:, :, 1].ravel())
    b = np.bincount(superpixels.ravel(), weights=image[:, :, 2].ravel())
    mean_colors = (np.vstack([r, g, b])
                   / np.bincount(superpixels.ravel())).T / 255.
    contrasts = [np.exp(-10. * np.linalg.norm(mean_colors[e[0]]
                                              - mean_colors[e[1]]))
                 for e in edges]
    return np.vstack(contrasts)


@memory.cache
def get_edge_directions(edges, superpixels):
    n_vertices = np.max(superpixels) + 1
    centers = np.empty((n_vertices, 2))
    gridx, gridy = np.mgrid[:superpixels.shape[0], :superpixels.shape[1]]

    for v in xrange(n_vertices):
        centers[v] = [gridy[superpixels == v].mean(),
                      gridx[superpixels == v].mean()]
    directions = []
    for edge in edges:
        e0, e1 = edge
        diff = centers[e0] - centers[e1]
        diff /= np.linalg.norm(diff)
        directions.append(np.arcsin(diff[1]))
    return np.vstack(directions)
