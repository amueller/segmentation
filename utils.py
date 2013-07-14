from collections import namedtuple
import numbers
import itertools

import numpy as np
from scipy import sparse

from sklearn.externals.joblib import Memory
from pystruct.utils import make_grid_edges

DataBunch = namedtuple('DataBunch', 'X, Y, file_names, superpixels')

memory = Memory(cachedir="/tmp/cache", verbose=1)


@memory.cache
def discard_void(dataset, data, latent_features=False):
    if isinstance(data.X[0], np.ndarray):
        X_new = [x[y != dataset.void_label] for x, y in zip(data.X, data.Y)]
        Y_new = [y[y != dataset.void_label] for y in data.Y]
        return DataBunch(X_new, Y_new, data.file_names,
                         data.superpixels)
    X_new, Y_new = [], []
    for x, y in zip(data.X, data.Y):
        mask = y != dataset.void_label
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
        elif len(x) == 4:
                features, edges, edge_features, n_hidden = x
                edge_features_new = edge_features
                mask = np.hstack([mask, np.ones(n_hidden, dtype=np.bool)])
        else:
            raise ValueError("len(x) is weird: %d" % len(data.X[0]))

        edges_new = edges
        if edges_new.shape[0] > 0:
            # if there are no edges, don't need to filter them
            # also, below code would break ;)
            for void_node in voids:
                involves_void_node = np.any(edges_new == void_node, axis=1)
                edges_new = edges_new[~involves_void_node]
                if (len(x) == 3 and not isinstance(x[2], numbers.Integral) or
                        len(x) == 4):
                    edge_features_new = edge_features_new[~involves_void_node]

        reindex_edges = np.zeros(len(mask), dtype=np.int)
        reindex_edges[mask] = np.arange(np.sum(mask))
        edges_new = reindex_edges[edges_new]
        if len(x) == 2:
            X_new.append((features[mask], edges_new))
            Y_new.append(y[mask])
        elif len(x) == 3:
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
        else:
            n_hidden_new = np.max(edges_new) - np.sum(mask[:-n_hidden]) + 1
            X_new.append((features[mask[:-n_hidden]], edges_new,
                          edge_features_new, n_hidden_new))
            Y_new.append(y[mask[:-n_hidden]])
    return DataBunch(X_new, Y_new, data.file_names, data.superpixels)


@memory.cache
def add_edges(data, kind="pairwise"):
    # generate graph
    if kind == "independent":
        X_new = [(x, np.empty((0, 2), dtype=np.int)) for x in data.X]

    elif kind == "extended":
        X_new = [(x, extend_edges(region_graph(sp), length=3))
                 for x, sp in zip(data.X, data.superpixels)]

    elif kind == "fully_connected":
        X_new = [(x, np.vstack([e for e in
                                itertools.combinations(np.arange(len(x)), 2)]))
                 for x in data.X]
    elif kind == "pairwise":
        X_new = [(x, region_graph(sp))
                 for x, sp in zip(data.X, data.superpixels)]
    else:
        raise ValueError("Parameter 'kind' should be one of 'independent'"
                         ",'fully_connected' or 'pairwise', got %s"
                         % kind)

    return DataBunch(X_new, data.Y, data.file_names, data.superpixels)


def radius_graph(superpixels, eps=40):
    n_vertices = np.max(superpixels) + 1
    centers = np.empty((n_vertices, 2))
    gridx, gridy = np.mgrid[:superpixels.shape[0], :superpixels.shape[1]]

    for v in xrange(n_vertices):
        centers[v] = [gridy[superpixels == v].mean(),
                      gridx[superpixels == v].mean()]
    edges = []
    for e in itertools.combinations(np.arange(n_vertices), 2):
        if np.linalg.norm(centers[e[0]] - centers[e[1]]) < eps:
            edges.append(e)
    return np.vstack(edges)


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


def extend_edges(edges, length=2):
    # returns all paths of length one or two in the graph given by edges
    n_vertices = np.max(edges) + 1
    graph = sparse.coo_matrix((np.ones(len(edges)), edges.T),
                              shape=(n_vertices, n_vertices))
    neighborhood = graph
    for i in range(length - 1):
        graph = graph + neighborhood * graph
    return np.c_[graph.nonzero()]


def get_mean_colors(image, superpixels):
    r = np.bincount(superpixels.ravel(), weights=image[:, :, 0].ravel())
    g = np.bincount(superpixels.ravel(), weights=image[:, :, 1].ravel())
    b = np.bincount(superpixels.ravel(), weights=image[:, :, 2].ravel())
    mean_colors = (np.vstack([r, g, b])
                   / np.bincount(superpixels.ravel())).T / 255.
    return mean_colors


def get_edge_contrast(edges, image, superpixels, gamma=10):
    mean_colors = get_mean_colors(image, superpixels)
    contrasts = [np.exp(-gamma * np.linalg.norm(mean_colors[e[0]]
                                                - mean_colors[e[1]]))
                 for e in edges]
    return np.vstack(contrasts)


def get_superpixel_centers(superpixels):
    n_vertices = np.max(superpixels) + 1
    centers = np.empty((n_vertices, 2))
    gridx, gridy = np.mgrid[:superpixels.shape[0], :superpixels.shape[1]]
    for v in xrange(n_vertices):
        centers[v] = [gridy[superpixels == v].mean(),
                      gridx[superpixels == v].mean()]
    return centers


def get_center_distances(edges, superpixels):
    centers = get_superpixel_centers(superpixels)
    distances = np.sum((centers[edges[:, 0]] - centers[edges[:, 1]]) ** 2,
                       axis=1)
    distances -= distances.min()
    distances /= distances.max()
    return np.exp(-distances[:, np.newaxis] * 2.)


@memory.cache
def get_edge_directions(edges, superpixels):
    centers = get_superpixel_centers(superpixels)

    directions = []
    for edge in edges:
        e0, e1 = edge
        diff = centers[e0] - centers[e1]
        diff /= np.linalg.norm(diff)
        directions.append(np.arcsin(diff[1]))
    return np.vstack(directions)


def add_edge_features(dataset, data, more_colors=False,
                      center_distances=False):
    X = []
    for x, superpixels, file_name in zip(data.X, data.superpixels,
                                         data.file_names):
        features = [np.ones((x[1].shape[0], 1))]
        image = dataset.get_image(file_name)
        if more_colors:
            features.append(get_edge_contrast(x[1], image, superpixels,
                                              gamma=5))
            features.append(get_edge_contrast(x[1], image, superpixels,
                                              gamma=10))
            features.append(get_edge_contrast(x[1], image, superpixels,
                                              gamma=20))
            features.append(get_edge_contrast(x[1], image, superpixels,
                                              gamma=100))
        else:
            features.append(get_edge_contrast(x[1], image, superpixels,
                                              gamma=10))
        if center_distances:
            features.append(get_center_distances(x[1], superpixels))
        features.append(get_edge_directions(x[1], superpixels))
        X.append((x[0], x[1], np.hstack(features)))
    return DataBunch(X, data.Y, data.file_names, data.superpixels)


def gt_in_sp(dataset, filename, superpixels):
    y = dataset.get_ground_truth(filename)
    votes = sparse.coo_matrix((np.ones(superpixels.size),
                               (y.ravel(), superpixels.ravel())))
    return np.argmax(votes.toarray(), axis=0)


def eval_on_pixels(dataset, Y_true, Y_pred, print_results=False):
    tp, tn, fp, fn = np.zeros(21), np.zeros(21), np.zeros(21), np.zeros(21)
    for y_true, y_pred in zip(Y_true, Y_pred):
        mask = y_true != dataset.void_label
        y_true, y_pred = y_true[mask], y_pred[mask]
        for k in range(21):
            tp[k] += np.sum((y_true == k) * (y_pred == k))
            tn[k] += np.sum((y_true != k) * (y_pred != k))
            fp[k] += np.sum((y_true != k) * (y_pred == k))
            fn[k] += np.sum((y_true == k) * (y_pred != k))
    jaccard = tp / (fp + fn + tp) * 100
    hamming = tp / (tp + fn) * 100
    if print_results:
        np.set_printoptions(precision=2)
        print("Jaccard")
        print(jaccard)
        print("Hamming")
        print(hamming)
        print("Mean Jaccard: %.1f   Mean Hamming: %.1f"
              % (np.mean(jaccard), np.mean(hamming)))

    return hamming, jaccard


def eval_on_sp(dataset, data, Y_pred, print_results=False):
    Y_pred_pixels = [y_pred[sp] for sp, y_pred in zip(data.superpixels,
                                                      Y_pred)]
    Y_true = [dataset.get_ground_truth(f) for f in data.file_names]
    return eval_on_pixels(dataset, Y_true, Y_pred_pixels,
                          print_results=print_results)


@memory.cache
def add_global_descriptor(data):
    # adds to each superpixel a feature consisting of the average over the
    # image
    global_descs = map(lambda x: x.sum(axis=0) / x.shape[0], data.X)
    X_new = [np.hstack([x, np.repeat(d[np.newaxis, :], x.shape[0], axis=0)])
             for d, x in zip(global_descs, data.X)]
    return DataBunch(X_new, data.Y, data.file_names, data.superpixels)
