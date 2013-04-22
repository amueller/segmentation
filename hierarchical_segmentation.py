import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imsave
from scipy import sparse

#from sklearn.cluster import Ward
from sklearn.cluster import KMeans
from sklearn.externals.joblib import Memory
from skimage.segmentation import mark_boundaries

#from msrc_first_try import load_data
from msrc_helpers import MSRCDataset, colors, classes, add_edges


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
    feats, edges = x
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
    feats, edges = x
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


def plot_results_hierarchy(data, Y_pred, folder="figures"):
    if not os.path.exists(folder):
        os.mkdir(folder)
    msrc = MSRCDataset()
    import matplotlib.colors as cl
    np.random.seed(0)
    random_colormap = cl.ListedColormap(np.random.uniform(size=(100, 3)))
    for stuff in zip(data.file_names, data.superpixels,
                     data.segments, data.Y, Y_pred):
        image_name, superpixels, segments, y, y_pred = stuff
        image = msrc.get_image(image_name)
        h = y_pred[len(y):]
        y_pred = y_pred[:len(y)]

        fig, axes = plt.subplots(2, 3, figsize=(12, 6))

        axes[0, 0].imshow(image)
        axes[0, 1].set_title("ground truth")
        axes[0, 1].imshow(image)
        gt = msrc.get_ground_truth(image_name)
        axes[0, 1].imshow(colors[gt], alpha=.7)
        axes[1, 0].set_title("sp ground truth")
        axes[1, 0].imshow(image)
        axes[1, 0].imshow(colors[y[superpixels]], vmin=0, vmax=23, alpha=.7)

        axes[1, 1].set_title("prediction")
        axes[1, 1].imshow(image)
        axes[1, 1].imshow(colors[y_pred[superpixels]], vmin=0, vmax=23,
                          alpha=.7)
        present_y = np.unique(np.hstack([y, y_pred]))

        vmax = np.max(np.hstack(Y_pred))
        vmin = np.min(np.hstack(Y_pred))
        axes[1, 2].imshow(mark_boundaries(image, segments[superpixels]))
        axes[1, 2].imshow(h[segments[superpixels]], vmin=vmin, vmax=vmax,
                          alpha=.7, cmap=random_colormap)

        axes[0, 2].imshow(colors[present_y, :][:, np.newaxis, :],
                          interpolation='nearest', alpha=.7)
        for i, c in enumerate(present_y):
            axes[0, 2].text(1, i, classes[c])
        for ax in axes.ravel():
            ax.set_xticks(())
            ax.set_yticks(())
        fig.savefig(folder + "/%s.png" % image_name, bbox_inches="tight")
        plt.close(fig)


def main():
    import cPickle
    from datasets.msrc import MSRCDataset
    with open("../superpixel_crf/data_probs_train_cw.pickle") as f:
        data = cPickle.load(f)
    data = add_edges(data, independent=False)
    msrc = MSRCDataset()
    #X, Y, image_names, images, all_superpixels = load_data(
        #"train", independent=False)
    for x, name, sps in zip(data.X, data.file_names, data.superpixels):
        segments = get_km_segments(x, msrc.get_image(name), sps, n_segments=10)
        boundary_image = mark_boundaries(mark_boundaries(msrc.get_image(name),
                                                         sps), segments[sps],
                                         color=[1, 0, 0])
        imsave("hierarchy_sp_own_10/%s.png" % name, boundary_image)


if __name__ == "__main__":
    main()
