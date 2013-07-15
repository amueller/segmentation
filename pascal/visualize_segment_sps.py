import numpy as np
import matplotlib.pyplot as plt


from scipy.misc import imsave
from datasets.pascal import PascalSegmentation
from latent_crf_experiments.utils import get_superpixel_centers
from pascal_helpers import (superpixels_segments, merge_small_sp,
                            morphological_clean_sp, create_segment_sp_graph)

from skimage.segmentation import mark_boundaries
from skimage.measure import regionprops


def visualize_sps():
    pascal = PascalSegmentation()
    train_files = pascal.get_split()

    for image_file in train_files:
        print(image_file)
        image = pascal.get_image(image_file)
        segments, superpixels = superpixels_segments(image_file)
        new_regions, correspondences = merge_small_sp(image, superpixels)
        clean_regions = morphological_clean_sp(image, new_regions, 4)
        imsave("segment_sp_fixed/%s.png"
               % image_file, mark_boundaries(image, clean_regions))


def visualize_segments():
    pascal = PascalSegmentation()
    train_files = pascal.get_split()

    for image_file in train_files:
        print(image_file)
        image = pascal.get_image(image_file)
        segments, superpixels = superpixels_segments(image_file)
        new_regions, correspondences = merge_small_sp(image, superpixels)
        clean_regions = morphological_clean_sp(image, new_regions, 4)
        new_regions, correspondences = merge_small_sp(image, superpixels)
        clean_regions = morphological_clean_sp(image, new_regions, 4)
        marked = mark_boundaries(image, clean_regions)
        edges = create_segment_sp_graph(segments, clean_regions)
        edges = np.array(edges)
        n_segments = segments.shape[2]
        segment_centers = [regionprops(segments.astype(np.int)[:, :, i],
                                       ['Centroid'])[0]['Centroid'] for i in
                           range(n_segments)]
        segment_centers = np.vstack(segment_centers)[:, ::-1]
        superpixel_centers = get_superpixel_centers(clean_regions)
        grr = min(n_segments, 10)
        fig, axes = plt.subplots(3, grr // 3, figsize=(30, 30))

        for i, ax in enumerate(axes.ravel()):
            ax.imshow(mark_boundaries(marked, segments[:, :, i], (1, 0, 0)))
            ax.scatter(segment_centers[:, 0], segment_centers[:, 1],
                       color='red')
            ax.scatter(superpixel_centers[:, 0], superpixel_centers[:, 1],
                       color='blue')
            this_edges = edges[edges[:, 1] == i]
            for edge in this_edges:
                ax.plot([superpixel_centers[edge[0]][0],
                         segment_centers[edge[1]][0]],
                        [superpixel_centers[edge[0]][1],
                         segment_centers[edge[1]][1]], c='black')
            ax.set_xlim(0, superpixels.shape[1])
            ax.set_ylim(superpixels.shape[0], 0)
            ax.set_xticks(())
            ax.set_yticks(())
        #plt.show()
        #imsave("segments_test/%s.png" % image_file, marked)
        plt.savefig("segments_test/%s.png" % image_file)
        plt.close()

if __name__ == "__main__":
    visualize_sps()
    #visualize_segments()
