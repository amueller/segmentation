import numpy as np
from scipy import sparse

#from sklearn.cluster import Ward
from sklearn.cluster import KMeans
from sklearn.externals.joblib import Memory


memory = Memory(cachedir="/tmp/cache", verbose=0)


def get_colors(img, sps):
    reds = np.bincount(sps.ravel(), weights=img[:, :, 0].ravel())
    greens = np.bincount(sps.ravel(), weights=img[:, :, 1].ravel())
    blues = np.bincount(sps.ravel(), weights=img[:, :, 2].ravel())
    counts = np.bincount(sps.ravel())
    reds /= counts
    greens /= counts
    blues /= counts
    return np.vstack([reds, greens, blues]).T


def get_centers(sps):
    gridx, gridy = np.mgrid[:sps.shape[0], :sps.shape[1]]
    n_vertices = len(np.unique(sps))
    centers = np.zeros((n_vertices, 2))
    for v in xrange(n_vertices):
        centers[v] = [gridy[sps == v].mean(), gridx[sps == v].mean()]
    return centers


def get_km_segments(x, image, sps, n_segments=25):
    if len(x) == 2:
        feats, edges = x
    else:
        feats, edges, _ = x
    colors_ = get_colors(image, sps)
    centers = get_centers(sps)
    #graph = sparse.coo_matrix((np.ones(edges.shape[0]), edges.T))
    #ward = Ward(n_clusters=25, connectivity=graph)
    km = KMeans(n_clusters=n_segments)
    color_feats = np.hstack([colors_, centers * 2])
    return km.fit_predict(color_feats)


@memory.cache
def get_segment_features(x, y, image, sps):
    segments = get_km_segments(x, image, sps)
    if len(x) == 2:
        feats, edges = x
    else:
        feats, edges, _ = x
    segment_edges = segments[edges]
    # make direction of edges unique
    segment_edges = np.sort(segment_edges, axis=1)
    # to get rid of duplicate edges, self edges, become sparse matrix
    graph = sparse.coo_matrix((np.ones(segment_edges.shape[0]),
                               segment_edges.T))
    # conversion removes duplicates
    graph = graph.tocsr()
    # remove self edges at diag
    graph.setdiag(np.zeros(graph.shape[0]))
    segment_edges = np.vstack(graph.nonzero()).T

    features = [np.mean(feats[segments == i], axis=0) for i in
                np.unique(segments)]
    labels = [np.argmax(np.bincount(y[segments == i])) for i in
              np.unique(segments)]
    return segments, features, np.array(labels), edges
