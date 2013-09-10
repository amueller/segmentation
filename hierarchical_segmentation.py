from collections import namedtuple
import numpy as np
from scipy import sparse

from sklearn.cluster import Ward
#from sklearn.cluster import KMeans
from sklearn.externals.joblib import Memory


memory = Memory(cachedir="/tmp/cache", verbose=0)

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
        all_edges.append(np.sort(edges, axis=1))
    return all_edges


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
    n_spixel = len(feats)
    graph = sparse.coo_matrix((np.ones(edges.shape[0]), edges.T),
                              shape=(n_spixel, n_spixel))
    ward = Ward(n_clusters=n_segments, connectivity=graph + graph.T)
    #km = KMeans(n_clusters=n_segments)
    color_feats = np.hstack([colors_, centers * .5])
    #return km.fit_predict(color_feats)
    return ward.fit_predict(color_feats)


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


@memory.cache
def make_hierarchical_data(dataset, data, lateral=False, latent=False,
                           latent_lateral=False, add_edge_features=False):
    images = [dataset.get_image(f) for f in data.file_names]
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
            if add_edge_features:
                edge_features = x[2]
                # we assume that thie fist edge feature is symmetric, I guess..
                n_edge_features = x[2].shape[1]
                edge_features_new = np.zeros(n_edge_features)
                edge_features_new[0] = 1
                edge_features_new = np.repeat(edge_features_new[np.newaxis, :],
                                              len(edges), axis=0)
                edge_features_stacked = np.vstack([edge_features,
                                                   edge_features_new])
                x_stacked = (x[0], edges_stacked, edge_features_stacked,
                             len(feat))
            else:
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


def main():
    from pascal.pascal_helpers import load_pascal
    from datasets.pascal import PascalSegmentation
    from utils import add_edges
    from scipy.misc import imsave
    from skimage.segmentation import mark_boundaries

    ds = PascalSegmentation()
    data = load_pascal("train1")

    data = add_edges(data, independent=False)
    #X, Y, image_names, images, all_superpixels = load_data(
        #"train", independent=False)
    for x, name, sps in zip(data.X, data.file_names, data.superpixels):
        segments = get_km_segments(x, ds.get_image(name), sps, n_segments=25)
        boundary_image = mark_boundaries(mark_boundaries(ds.get_image(name),
                                                         sps), segments[sps],
                                         color=[1, 0, 0])
        imsave("hierarchy_sp_own_25/%s.png" % name, boundary_image)


def plot_results_hierarchy(dataset, data, Y_pred, folder="figures"):
    import os
    import matplotlib.pyplot as plt
    from skimage.segmentation import mark_boundaries
    if not os.path.exists(folder):
        os.mkdir(folder)
    import matplotlib.colors as cl
    np.random.seed(0)
    random_colormap = cl.ListedColormap(np.random.uniform(size=(100, 3)))
    for stuff in zip(data.file_names, data.superpixels,
                     data.segments, data.Y, Y_pred):
        image_name, superpixels, segments, y, y_pred = stuff
        image = dataset.get_image(image_name)
        h = y_pred[len(y):]
        y_pred = y_pred[:len(y)]

        fig, axes = plt.subplots(2, 3, figsize=(12, 6))

        axes[0, 0].imshow(image)
        axes[0, 1].set_title("ground truth")
        axes[0, 1].imshow(image)
        gt = dataset.get_ground_truth(image_name)
        axes[0, 1].imshow(gt, alpha=.7, cmap=dataset.cmap)
        axes[1, 0].set_title("sp ground truth")
        axes[1, 0].imshow(image)
        axes[1, 0].imshow(y[superpixels], vmin=0, vmax=23, alpha=.7,
                          cmap=dataset.cmap)

        axes[1, 1].set_title("prediction")
        axes[1, 1].imshow(image)
        axes[1, 1].imshow(y_pred[superpixels], vmin=0, vmax=23,
                          alpha=.7, cmap=dataset.cmap)
        present_y = np.unique(np.hstack([y, y_pred]))

        vmax = np.max(np.hstack(Y_pred))
        vmin = np.min(np.hstack(Y_pred))
        axes[1, 2].imshow(mark_boundaries(image, segments[superpixels]))
        axes[1, 2].imshow(h[segments[superpixels]], vmin=vmin, vmax=vmax,
                          alpha=.7, cmap=random_colormap)

        axes[0, 2].imshow(present_y[np.newaxis, :], interpolation='nearest',
                          alpha=.7, cmap=dataset.cmap)
        for i, c in enumerate(present_y):
            axes[0, 2].text(1, i, dataset.classes[c])
        for ax in axes.ravel():
            ax.set_xticks(())
            ax.set_yticks(())
        fig.savefig(folder + "/%s.png" % image_name, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
