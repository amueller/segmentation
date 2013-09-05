import matplotlib
matplotlib.use('WxAgg')
matplotlib.interactive(False)

import mayavi.mlab as mv
import numpy as np
import matplotlib.pyplot as plt

from skimage.segmentation import mark_boundaries

from datasets.nyu import NYUSegmentation
from nyu.nyu_helpers import load_nyu

#from utils import get_edge_depth_diff, add_edges
from utils import get_sp_normals, add_edges, get_superpixel_centers

#from IPython.core.debugger import Tracer
#tracer = Tracer()


def crazy_visual():
    dataset = NYUSegmentation()
    # load training data
    data = load_nyu(n_sp=500)
    data = add_edges(data)

    for x, image_name, superpixels, y in zip(data.X, data.file_names,
                                             data.superpixels, data.Y):
        print(image_name)
        if int(image_name) != 11:
            continue
        image = dataset.get_image(image_name)
        plt.figure(figsize=(20, 20))
        bounary_image = mark_boundaries(image, superpixels)
        plt.imshow(bounary_image)
        gridx, gridy = np.mgrid[:superpixels.shape[0], :superpixels.shape[1]]

        edges = x[1]
        points_normals = dataset.get_pointcloud_normals(image_name)
        centers2d = get_superpixel_centers(superpixels)
        centers3d = [np.bincount(superpixels.ravel(), weights=c.ravel())
                   for c in points_normals[:, :, :3].reshape(-1, 3).T]
        centers3d = (np.vstack(centers3d) / np.bincount(superpixels.ravel())).T
        sp_normals = get_sp_normals(points_normals[:, :, 3:], superpixels)
        offset = centers3d[edges[:, 0]] - centers3d[edges[:, 1]]
        offset = offset / np.sqrt(np.sum(offset ** 2, axis=1))[:, np.newaxis]
        #mean_normal = (sp_normals[edges[:, 0]] + sp_normals[edges[:, 1]]) / 2.
        mean_normal = sp_normals[edges[:, 0]]
        #edge_features = np.arccos(np.abs((offset * mean_normal).sum(axis=1))) * 2. / np.pi
        edge_features = 1 - np.abs((offset * mean_normal).sum(axis=1))
        no_normals = (np.all(sp_normals[edges[:, 0]] == 0, axis=1)
                      + np.all(sp_normals[edges[:, 1]] == 0, axis=1))
        edge_features[no_normals] = 0  # nan normals

        if True:
            coords = points_normals[:, :, :3].reshape(-1, 3)
            perm = np.random.permutation(superpixels.max()+1)
            mv.points3d(coords[:,0], coords[:, 1], coords[:, 2], perm[superpixels.ravel()], mode='point')
            #mv.points3d(centers3d[:, 0], centers3d[:, 1], centers3d[:, 2], scale_factor=.04)
            mv.quiver3d(centers3d[:, 0], centers3d[:, 1], centers3d[:, 2], sp_normals[:, 0], sp_normals[:, 1], sp_normals[:, 2])
            mv.show()
        from IPython.core.debugger import Tracer
        Tracer()()


        for i, edge in enumerate(edges):
            e0, e1 = edge
            #color = (dataset.colors[y[e0]] + dataset.colors[y[e1]]) / (2. * 255.)
            #f = edge_features[i]
            #if f < 0:
                #e0, e1 = e1, e0
                #f = -f

            #plt.arrow(centers[e0][0], centers[e0][1],
                     #centers[e1][0] - centers[e0][0], centers[e1][1] - centers[e0][1],
                     #width=f * 5
                     #)
            color = "black"
            plt.plot([centers2d[e0][0], centers2d[e1][0]],
                     [centers2d[e0][1], centers2d[e1][1]],
                     c=color,
                     linewidth=edge_features[i] * 5
                     )
        plt.scatter(centers2d[:, 0], centers2d[:, 1], s=100)
        plt.tight_layout()
        plt.xlim(0, superpixels.shape[1])
        plt.ylim(superpixels.shape[0], 0)
        plt.axis("off")
        plt.savefig("figures/normal_relative/%s.png" % image_name,
                    bbox_inches="tight")
        plt.close()

if __name__ == "__main__":
    crazy_visual()
