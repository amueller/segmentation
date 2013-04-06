from collections import namedtuple

import os
import numpy as np
import matplotlib.pyplot as plt

from datasets.msrc import MSRCDataset

colors = np.array(
    [[128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
     # [128, 0, 128], horse
     [0, 128, 128], [128, 128, 128],
     # [64, 0, 0], mountain
     [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
     [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0],
     [128, 64, 0], [0, 192, 0], [128, 64, 128], [0, 192, 128],
     [128, 192, 128], [64, 64, 0], [192, 64, 0], [0, 0, 0]])

classes = np.array(['building', 'grass', 'tree', 'cow', 'sheep', 'sky',
                    'aeroplane', 'water', 'face', 'car', 'bicycle', 'flower',
                    'sign', 'bird', 'book', 'chair', 'road', 'cat', 'dog',
                    'body', 'boat', 'void', 'mountain', 'horse'])

DataBunch = namedtuple('DataBunch', 'X, Y, file_names, images, superpixels')


def plot_results(data, Y_pred, folder="figures", use_colors_predict=True):
    if not os.path.exists(folder):
        os.mkdir(folder)
    import matplotlib.colors as cl
    np.random.seed(0)
    msrc = MSRCDataset()
    random_colormap = cl.ListedColormap(np.random.uniform(size=(100, 3)))
    for image, image_name, superpixels, y, y_pred in zip(data.images,
                                                         data.file_names,
                                                         data.superpixels,
                                                         data.Y, Y_pred):
        fig, axes = plt.subplots(2, 3, figsize=(12, 6))
        axes[0, 0].imshow(image)
        axes[0, 1].set_title("ground truth")
        axes[0, 1].imshow(image)
        gt = msrc.get_ground_truth(image_name)
        gt = gt - 1
        gt[gt == 255] = 21
        axes[0, 1].imshow(colors[gt], alpha=.7)
        axes[1, 0].set_title("sp ground truth")
        axes[1, 0].imshow(image)
        axes[1, 0].imshow(colors[y[superpixels]], vmin=0, vmax=23, alpha=.7)

        axes[1, 1].set_title("prediction")
        axes[1, 1].imshow(image)
        if use_colors_predict:
            axes[1, 1].imshow(colors[y_pred[superpixels]], alpha=.7)
        else:
            vmax = np.max(np.hstack(Y_pred))
            axes[1, 1].imshow(y_pred[superpixels], vmin=0, vmax=vmax, alpha=.9,
                              cmap=random_colormap)
        if use_colors_predict:
            present_y = np.unique(np.hstack([y, y_pred]))
        else:
            present_y = np.unique(y)
        axes[0, 2].imshow(colors[present_y, :][:, np.newaxis, :],
                          interpolation='nearest')
        for i, c in enumerate(present_y):
            axes[0, 2].text(1, i, classes[c])
        for ax in axes.ravel():
            ax.set_xticks(())
            ax.set_yticks(())
        axes[1, 2].set_visible(False)
        fig.savefig(folder + "/%s.png" % image_name, bbox_inches="tight")
        plt.close(fig)


def discard_void(X, Y, void_label=21):
    X_new, Y_new = [], []
    for x, y in zip(X, Y):
        features, edges = x
        mask = y != void_label
        voids = np.where(~mask)[0]
        edges_new = edges
        if edges_new.shape[0] > 0:
            # if there are no edges, don't need to filter them
            # also, below code would break ;)
            for void_node in voids:
                involves_void_node = np.any(edges_new == void_node, axis=1)
                edges_new = edges_new[~involves_void_node]
        reindex_edges = 1000 * np.zeros(len(mask), dtype=np.int)
        reindex_edges[mask] = np.arange(len(mask))
        edges_new = reindex_edges[edges_new]

        X_new.append((features[mask], edges_new))
        Y_new.append(y[mask])
    return X_new, Y_new
