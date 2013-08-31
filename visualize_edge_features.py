import numpy as np
import matplotlib.pyplot as plt

from skimage.segmentation import mark_boundaries

from datasets.nyu import NYUSegmentation
from nyu.nyu_helpers import load_nyu

from utils import get_edge_depth_diff, add_edges

from IPython.core.debugger import Tracer
tracer = Tracer()


def crazy_visual():
    dataset = NYUSegmentation()
    # load training data
    data = load_nyu()
    data = add_edges(data)

    for x, image_name, superpixels, y in zip(data.X, data.file_names,
                                             data.superpixels, data.Y):
        image = dataset.get_image(image_name)
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
        depth = dataset.get_depth(image_name)
        edge_features = get_edge_depth_diff(edges, depth, superpixels, gamma=5)
        #from IPython.core.debugger import Tracer
        #Tracer()()
        for i, edge in enumerate(edges):
            e0, e1 = edge
            #color = (dataset.colors[y[e0]] + dataset.colors[y[e1]]) / (2. * 255.)
            color = "black"
            plt.plot([centers[e0][0], centers[e1][0]],
                     [centers[e0][1], centers[e1][1]],
                     c=color,
                     linewidth=edge_features[i] * 5
                     )
        plt.scatter(centers[:, 0], centers[:, 1], s=100)
        plt.tight_layout()
        plt.xlim(0, superpixels.shape[1])
        plt.ylim(superpixels.shape[0], 0)
        plt.axis("off")
        plt.savefig("figures/depth_diff/%s.png" % image_name,
                    bbox_inches="tight")
        plt.close()

if __name__ == "__main__":
    crazy_visual()
