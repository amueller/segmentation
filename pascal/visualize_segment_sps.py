from scipy.misc import imsave
from datasets.pascal import PascalSegmentation
from pascal_helpers import (superpixels_segments, merge_small_sp,
                            morphological_clean_sp)

from skimage.segmentation import mark_boundaries

pascal = PascalSegmentation()
train_files = pascal.get_split()

for image_file in train_files:
    print(image_file)
    image = pascal.get_image(image_file)
    segments, superpixels = superpixels_segments(image_file)
    new_regions, correspondences = merge_small_sp(image, superpixels)
    clean_regions = morphological_clean_sp(image, new_regions, 4)
    imsave("segment_sp_morph/%s.png"
           % image_file, mark_boundaries(image, clean_regions))
