import numpy as np

from datasets.pascal import PascalSegmentation
from pascal_helpers import superpixels_segments, merge_small_sp


def test_remove_small_segments():
    pascal = PascalSegmentation()
    train_files = pascal.get_split()

    idx = 10
    image = pascal.get_image(train_files[idx])
    segments, superpixels = superpixels_segments(train_files[idx])
    new_regions, correspondences = merge_small_sp(image, superpixels)
    new_counts = np.bincount(new_regions.ravel())
    if np.any(new_counts < 50):
        raise ValueError("Stupid thing!")
