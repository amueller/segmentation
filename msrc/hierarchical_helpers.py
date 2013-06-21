import os
import matplotlib.pyplot as plt
from scipy.misc import imsave
from skimage.segmentation import mark_boundaries

import numpy as np
from collections import namedtuple
from scipy import sparse

from msrc_helpers import (DataBunch, load_data, sigm, add_edges,
                          add_kraehenbuehl_features)
from msrc_helpers import MSRCDataset, colors, classes
from latent_crf_experiments.hierarchical_segmentation import \
    (get_segment_features, get_km_segments)


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


def make_hierarchical_data(data, lateral=False, latent=False,
                           latent_lateral=False):
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
