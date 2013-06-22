import numpy as np
from scipy import sparse

#from quickshift import quickshift
from vlfeat import vl_dsift
from skimage.color import rgb2gray
from sklearn.utils import shuffle

from sklearn.cluster import MiniBatchKMeans
#from sklearn.cluster import KMeans
#import os

from joblib import Memory
#from joblib import Parallel, delayed

from utils import DataBunch, gt_in_sp

memory = Memory(cachedir="/tmp/cache", verbose=0)

from IPython.core.debugger import Tracer
tracer = Tracer()


from sklearn.metrics.pairwise import chi2_kernel


class Chi2Kernel(object):
    def __init__(self, gamma=1):
        self.gamma = gamma

    def __call__(self, x, y):
        return chi2_kernel(x, y, gamma=self.gamma)

    def __repr__(self):
        return "Chi2Kernel(gamma=%f)" % self.gamma


@memory.cache
def color_descriptors(images, spixel, dataset, vq):
    n_words = 300
    some_colors = []
    for f in images:
        image = dataset.get_image(f)
        some_colors.append(image.reshape(-1, 3)[::10, :])
    if vq is None:
        vq = MiniBatchKMeans(n_clusters=n_words, verbose=1, init='random',
                             batch_size=2 * n_words, random_state=1)
        vq.fit(shuffle(np.vstack(some_colors)))

    bows = []
    for f, sp in zip(images, spixel):
        image = dataset.get_image(f)
        words = vq.predict(image.reshape(-1, 3).astype(np.float))
        bins = [np.arange(np.max(sp) + 2), np.arange(n_words + 1)]
        bow = np.histogram2d(sp.ravel(), words, bins=bins)[0]
        bows.append(bow)
    return vq, bows


def rgsift(image):
    from skimage import img_as_float
    shaped_image = img_as_float(image)
    gray = rgb2gray(image)
    s = shaped_image.sum(axis=2)
    red = shaped_image[:, :, 0] / (s + 1e-5)
    green = shaped_image[:, :, 1] / (s + 1e-5)
    descs = []
    for channel in [gray, red, green]:
        loc, desc = vl_dsift(channel, step=4, size=6)
        descs.append(desc.T.copy())
    return loc, np.hstack(descs)


def global_descriptors(images, dataset, vq=None, n_words=1000):

    descriptors, _ = sift_descriptors(images, dataset)
    vq, bow = global_bow(descriptors, vq=vq, n_words=n_words)
    return vq, bow


@memory.cache
def global_bow(descs, vq=None, n_words=1000):
    """Compute bag of words from sift descriptors and superpixels.

    Parameters
    ----------
    descs : list of ndarray
        For each image, array of sift descriptors.

    vq : Clustering Object or None.
        Fitted clustering object or None if clustering should be performed.

    n_words : int
        Number of words, i.e. clusters to find. Default=1000.
    """

    if vq is None:
        vq = MiniBatchKMeans(n_clusters=n_words, verbose=1, init='random',
                             batch_size=2 * n_words, compute_labels=False,
                             reassignment_ratio=0.0, random_state=1, n_init=3)
        #vq = KMeans(n_clusters=n_words, verbose=10, init='random')
        vq.fit(shuffle(np.vstack(descs)))
    else:
        n_words = vq.n_clusters

    bows = [np.bincount(vq.predict(desc), minlength=n_words).astype(np.float32)
            for desc in descs]
    return vq, bows


@memory.cache
def extract_spatial_pyramid(images, dataset, vq=None, n_words=1000):
    descriptors, locations = sift_descriptors(images, dataset)
    if vq is None:
        vq = MiniBatchKMeans(n_clusters=n_words, verbose=1, init='random',
                             batch_size=2 * n_words, compute_labels=False,
                             reassignment_ratio=0.0, random_state=1, n_init=3)
        #vq = KMeans(n_clusters=n_words, verbose=10, init='random')
        vq.fit(shuffle(np.vstack(descriptors)))
    else:
        n_words = vq.n_clusters

    pyramids = []
    for descr, locs in zip(descriptors, locations):
        words = vq.predict(descr)
        global_ = np.bincount(words, minlength=n_words).astype(np.float)
        global_ /= max(global_.sum(), 1)
        third_of_image = locs[1].max() // 3 + 1
        stripe_indicator = locs[1] // third_of_image
        inds = np.vstack([stripe_indicator, words])
        stripe_hists = sparse.coo_matrix((np.ones(len(words)), inds),
                                         shape=(3, n_words)).toarray()

        stripe_hists = [x / max(x.sum(), 1) for x in stripe_hists]
        pyramids.append(np.hstack([np.hstack(stripe_hists), global_]))

    return vq, np.vstack(pyramids)


@memory.cache
def sift_descriptors(images, dataset):
    descs = []
    coordinates = []
    print("computing sift descriptors")
    for f in images:
        print("processing image %s" % f)
        image = dataset.get_image(f)
        #coords, sift = rgsift(image)
        #tracer()
        gray_image = rgb2gray(image)
        coords, sift = vl_dsift(gray_image, step=3, size=4)
        #coords2, sift2 = vl_dsift(gray_image, step=3, size=8)
        #coords3, sift3 = vl_dsift(gray_image, step=3, size=16)
        #tracer()
        #sift = np.hstack([sift, sift2, sift3])
        #coords = np.hstack([coords, coords2, coords3])
        descs.append(sift.T)
        coordinates.append(coords)
    return descs, coordinates


@memory.cache
def bag_of_words(descs, spixel, coordinates, vq=None, n_words=1000):
    """Compute bag of words from sift descriptors and superpixels.

    Parameters
    ----------
    descs : list of ndarray
        For each image, array of sift descriptors.

    spixel : list of ndarray
        For each image, superpixel index for each pixel.

    coordinates : list of ndarray
        For each image, coordinate positions of sift descriptors.

    vq : Clustering Object or None.
        Fitted clustering object or None if clustering should be performed.

    n_words : int
        Number of words, i.e. clusters to find. Default=1000.
    """

    if vq is None:
        vq = MiniBatchKMeans(n_clusters=n_words, verbose=1, init='random',
                             batch_size=2 * n_words, compute_labels=False,
                             reassignment_ratio=0.0, random_state=1, n_init=3)
        #vq = KMeans(n_clusters=n_words, verbose=10, init='random')
        vq.fit(shuffle(np.vstack(descs)))
    else:
        n_words = vq.n_clusters

    bows = []
    for desc, sp, coords in zip(descs, spixel, coordinates):
        coords = coords.astype(np.int)
        desc_in_sp = sp[coords[1], coords[0]]
        bins = [np.arange(np.max(sp) + 2), np.arange(n_words + 1)]
        bow = np.histogram2d(desc_in_sp, vq.predict(desc), bins=bins)[0]
        bows.append(bow)
    return vq, bows


class SiftBOW(object):
    def __init__(self, dataset, n_words=300):
        self.dataset = dataset
        self.n_words = n_words

    def fit_transform(self, image_names, superpixels):
        descriptors, coordinates = sift_descriptors(image_names, self.dataset)
        print("end sift descriptors")
        vq, X = bag_of_words(descriptors, superpixels, coordinates)
        self.vq_ = vq
        Y = gt_in_sp(self.dataset, image_names, superpixels)
        return DataBunch(X, Y, image_names, superpixels)

    def fit(self, image_names, spixel):
        self.fit_predict(image_names, spixel)
        return self

    def transform(self, image_names, superpixels):
        descriptors, coordinates = sift_descriptors(image_names, self.dataset)
        _, X = bag_of_words(descriptors, superpixels, coordinates, vq=self.vq_)
        Y = gt_in_sp(self.dataset, image_names, superpixels)
        return DataBunch(X, Y, image_names, superpixels)
