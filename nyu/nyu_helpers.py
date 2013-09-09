import numpy as np
#import os
#from glob import glob

from sklearn.externals.joblib import Memory, Parallel, delayed
from scipy.misc import imread
#import matplotlib.pyplot as plt

from datasets.nyu import NYUSegmentation

from slic_python import slic_n
from latent_crf_experiments.utils import (DataBunchNoSP, DataBunch, gt_in_sp,
                                          probabilities_on_sp, get_sp_normals)
from latent_crf_experiments.pascal.pascal_helpers import merge_small_sp
from latent_crf_experiments.hierarchical_segmentation import \
    make_hierarchy_edges, HierarchicalDataBunch
from skimage.segmentation import slic
from skimage import morphology

from information_theoretic_mst import ITM

memory = Memory(cachedir="/home/data/amueller/cache")


def get_probabilities(file_name, path):
    probabilities = []
    for label in xrange(1, 5):
        f = ("%s/prediction_all/%s_lab_image_label_%d.png"
             % (path, file_name, label))
        probabilities.append(imread(f)[:, :, 0])
    probabilities = np.dstack(probabilities).astype(np.float)
    return probabilities / 255.


def load_single_file(dataset, file_name, n_sp=300, sp='rgb', reorder=None):
    print(file_name)
    image = dataset.get_image(file_name)
    if sp == 'rgb':
        sps = slic_n(image, n_superpixels=n_sp, compactness=10)
    elif sp == 'rgb-skimage':
        sps = slic(image, n_segments=n_sp, compactness=10, multichannel=True, sigma=0.1)
        sps = merge_small_sp(image, morphology.label(sps))[0]
    elif sp == 'rgbd':
        depth = dataset.get_depth(file_name)
        #depth -= depth.min()
        #depth /= depth.max()
        rgbd = np.dstack([image / 255., depth])
        sps = slic(rgbd, n_segments=n_sp, compactness=.1, convert2lab=False, multichannel=True, sigma=0)
        sps = merge_small_sp(image, morphology.label(sps))[0]
    else:
        raise ValueError("Expected sp to be 'rgb' or 'rgbd' got %d" % sp)

    gt = gt_in_sp(dataset, file_name, sps)
    probs = get_probabilities(file_name, dataset.directory)
    if reorder is not None:
        probs = probs[:, :, reorder]
    probs_sp = probabilities_on_sp(dataset, probs, sps)
    return probs_sp, gt, sps


@memory.cache
def load_nyu(ds='train', n_sp=300, sp='rgb'):
    # trigger cache.....
    dataset = NYUSegmentation()
    file_names = dataset.get_split(ds)
    if ds == "test":
        reorder = np.array([2, 0, 3, 1])
    else:
        reorder = None
    # load image to generate superpixels
    result = Parallel(n_jobs=-1)(delayed(load_single_file)(dataset, f, n_sp, sp, reorder=reorder)
                                 for f in file_names)
    X, Y, superpixels = zip(*result)

    return DataBunch(X, Y, file_names, superpixels)


@memory.cache
def load_nyu_pixelwise(ds='train'):
    if ds == "test":
        reorder = np.array([2, 0, 3, 1])
    else:
        reorder = np.arange(4)
    # trigger cache.
    dataset = NYUSegmentation()
    file_names, X, Y = [], [], []
    for file_name in dataset.get_split(ds):
        print(file_name)
        file_names.append(file_name)
        gt = dataset.get_ground_truth(file_name)
        prediction = get_probabilities(file_name, dataset.directory)
        Y.append(gt)
        X.append(prediction[:, :, reorder])
    return DataBunchNoSP(X, Y, file_names)


def compute_xyz_segments(dataset, data):
    segments = []
    for image_name, superpixels in zip(data.file_names, data.superpixels):
        points_normals = dataset.get_pointcloud_normals(image_name)
        centers3d = [np.bincount(superpixels.ravel(), weights=c.ravel())
                   for c in points_normals[:, :, :3].reshape(-1, 3).T]
        centers3d = (np.vstack(centers3d) / np.bincount(superpixels.ravel())).T
        sp_normals = get_sp_normals(points_normals[:, :, 3:], superpixels)
        km = ITM(n_clusters=30)
        km.fit(np.hstack([centers3d, sp_normals]))
        segments.append(km.labels_)
    return segments

def make_hierarchical_data(dataset, data):
    segments = compute_xyz_segments(dataset, data)
    hierarchy_edges = make_hierarchy_edges(segments, data.superpixels)
    X_stacked = []
    for x, y, edges in zip(data.X, data.Y, hierarchy_edges):
        edges_stacked = np.vstack([x[1], edges])
        # only add a single constant edge feature
        hierarchy_edge_features = np.zeros((edges.shape[0], x[2].shape[1]))
        hierarchy_edge_features[:, 0] = 1
        edge_features_stacked = np.vstack([x[2], hierarchy_edge_features])
        n_hidden = np.max(edges) + 1
        x_stacked = (x[0], edges_stacked, edge_features_stacked, n_hidden)
        X_stacked.append(x_stacked)
    return HierarchicalDataBunch(X_stacked, data.Y, data.file_names,
                                 data.superpixels, segments)
