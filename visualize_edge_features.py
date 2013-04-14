import numpy as np
import matplotlib.pyplot as plt

from skimage.segmentation import mark_boundaries

from datasets.msrc import MSRCDataset

from msrc_helpers import load_data

from IPython.core.debugger import Tracer
tracer = Tracer()


def crazy_visual():
    msrc = MSRCDataset()
    data = load_data("train", independent=False)

    for x, image, image_name, superpixels, y in zip(data.X, data.images,
                                                    data.file_names,
                                                    data.superpixels, data.Y):
        plt.figure(figsize=(20, 20))
        bounary_image = mark_boundaries(image, superpixels)
        plt.imshow(bounary_image)
        gridx, gridy = np.mgrid[:superpixels.shape[0], :superpixels.shape[1]]

        n_vertices = np.max(superpixels) + 1
        centers = np.empty((n_vertices, 2))
        edges = x[1]
        for v in xrange(n_vertices):
            centers[v] = [gridy[superpixels == v].mean(),
                          gridx[superpixels == v].mean()]
        for edge in edges:
            e0, e1 = edge
            color = (msrc.colors[y[e0]] + msrc.colors[y[e1]]) / (2. * 255.)
            plt.plot([centers[e0][0], centers[e1][0]],
                     [centers[e0][1], centers[e1][1]],
                     #c=['red', 'blue'][np.arcsin(diff[1]) > 0],
                     c=color,
                     #linewidth=2 * np.abs(np.arcsin(diff[1])))
                     #linewidth=contrast
                     )
        plt.scatter(centers[:, 0], centers[:, 1],
                    c=msrc.colors[y] / 255., s=100)
        plt.tight_layout()
        plt.xlim(0, superpixels.shape[1])
        plt.ylim(superpixels.shape[0], 0)
        plt.axis("off")
        plt.savefig("figures/crazy_edge_strength_color/%s.png" % image_name,
                    bbox_inches="tight")
        plt.close()

if __name__ == "__main__":
    crazy_visual()
