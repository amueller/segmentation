from skimage.segmentation import mark_boundaries
from scipy.misc import imsave

from pascal_helpers import load_pascal, load_image
from latent_crf_experiments.utils import add_edges
from latent_crf_experiments.hierarchical_segmentation import get_km_segments


def main():
    data = load_pascal("train1")

    data = add_edges(data, independent=False)
    #X, Y, image_names, images, all_superpixels = load_data(
        #"train", independent=False)
    for x, name, sps in zip(data.X, data.file_names, data.superpixels):
        segments = get_km_segments(x, load_image(name), sps, n_segments=25)
        boundary_image = mark_boundaries(mark_boundaries(load_image(name),
                                                         sps), segments[sps],
                                         color=[1, 0, 0])
        imsave("hierarchy_sp_own_25/%s.png" % name, boundary_image)


if __name__ == "__main__":
    main()
