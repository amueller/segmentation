import os
import matplotlib.pyplot as plt
import numpy as np

from skimage.segmentation import mark_boundaries

from utils import get_superpixel_centers


def plot_results(dataset, data, Y_pred, folder="figures",
                 use_colors_predict=True):
    if not os.path.exists(folder):
        os.mkdir(folder)
    import matplotlib.colors as cl
    np.random.seed(0)
    random_colormap = cl.ListedColormap(np.random.uniform(size=(100, 3)))
    for image_name, superpixels, y, y_pred in zip(data.file_names,
                                                  data.superpixels, data.Y,
                                                  Y_pred):
        image = dataset.get_image(image_name)
        fig, axes = plt.subplots(2, 3, figsize=(12, 6))
        axes[0, 0].imshow(image)
        axes[0, 1].set_title("ground truth")
        axes[0, 1].imshow(image)
        gt = dataset.get_ground_truth(image_name)
        axes[0, 1].imshow(gt, alpha=.7, cmap=dataset.cmap)
        axes[1, 0].set_title("sp ground truth")
        axes[1, 0].imshow(image)
        axes[1, 0].imshow(y[superpixels], vmin=0, vmax=dataset.cmap.N,
                          alpha=.7, cmap=dataset.cmap)

        axes[1, 1].set_title("prediction")
        axes[1, 1].imshow(image)
        if use_colors_predict:
            axes[1, 1].imshow(y_pred[superpixels], alpha=.7, vmin=0,
                              vmax=dataset.cmap.N, cmap=dataset.cmap)
        else:
            vmax = np.max(np.hstack(Y_pred))
            axes[1, 1].imshow(y_pred[superpixels], vmin=0, vmax=vmax, alpha=.9,
                              cmap=random_colormap)
        if use_colors_predict:
            present_y = np.unique(np.hstack([y, y_pred]))
        else:
            present_y = np.unique(y)
        present_y = present_y[present_y != dataset.void_label]
        axes[0, 2].imshow(present_y[:, np.newaxis], interpolation='nearest',
                          cmap=dataset.cmap, vmin=0, vmax=dataset.cmap.N)
        for i, c in enumerate(present_y):
            axes[0, 2].text(1, i, dataset.classes[c])
        for ax in axes.ravel():
            ax.set_xticks(())
            ax.set_yticks(())
        axes[1, 2].set_visible(False)
        fig.savefig(folder + "/%s.png" % image_name, bbox_inches="tight")
        plt.close(fig)


def plot_sp_graph(image, superpixels, edges, colors='black'):
    plt.figure(figsize=(10, 10))
    bounary_image = mark_boundaries(image, superpixels)
    plt.imshow(bounary_image)
    centers = get_superpixel_centers(superpixels)

    for i, edge in enumerate(edges):
        e0, e1 = edge
        if len(colors) == len(edges):
            color = colors[i]
        else:
            color = colors
        plt.plot([centers[e0][0], centers[e1][0]],
                 [centers[e0][1], centers[e1][1]],
                 c=color)
    plt.scatter(centers[:, 0], centers[:, 1], s=100)
    plt.tight_layout()
    plt.xlim(0, superpixels.shape[1])
    plt.ylim(superpixels.shape[0], 0)
    plt.axis("off")
