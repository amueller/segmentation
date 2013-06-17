#!/usr/bin/python
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

from pystruct.utils import SaveLogger

from msrc_helpers import (add_edge_features, add_edges, load_data)
from msrc_helpers import add_kraehenbuehl_features

from datasets.msrc import MSRCDataset, colors, classes


def main():
    argv = sys.argv
    print("loading %s ..." % argv[1])
    ssvm1 = SaveLogger(file_name=argv[1]).load()
    ssvm2 = SaveLogger(file_name=argv[2]).load()

    data_str = 'val'
    if len(argv) <= 3:
        raise ValueError("Need a folder name for plotting.")
    data = load_data(data_str, which="piecewise")
    data1 = add_edges(data, independent=False)
    data2 = add_edges(data, independent=False, fully_connected=True)
    data1 = add_kraehenbuehl_features(data1, which="train_30px")
    data1 = add_kraehenbuehl_features(data1, which="train")
    data2 = add_kraehenbuehl_features(data2, which="train_30px")
    data2 = add_kraehenbuehl_features(data2, which="train")
    data1 = add_edge_features(data1)
    data2 = add_edge_features(data2)
    Y_pred1 = ssvm1.predict(data1.X)
    Y_pred2 = ssvm2.predict(data2.X)
    folder = argv[3]

    if not os.path.exists(folder):
        os.mkdir(folder)

    msrc = MSRCDataset()
    np.random.seed(0)
    for image_name, superpixels, y_pred1, y_pred2 in zip(data.file_names,
                                                         data.superpixels,
                                                         Y_pred1, Y_pred2):
        if np.all(y_pred1 == y_pred2):
            continue
        image = msrc.get_image(image_name)
        fig, axes = plt.subplots(2, 3, figsize=(12, 6))
        axes[0, 0].imshow(image)
        axes[0, 0].imshow((y_pred1 != y_pred2)[superpixels], vmin=0, vmax=1,
                          alpha=.7)

        axes[0, 1].set_title("ground truth")
        axes[0, 1].imshow(image)
        gt = msrc.get_ground_truth(image_name)
        axes[0, 1].imshow(colors[gt], alpha=.7)
        perf = msrc.eval_pixel_performance([image_name],
                                           [y_pred1[superpixels]],
                                           print_results=False)['global']
        axes[1, 0].set_title("%.2f" % (100 * perf))
        axes[1, 0].imshow(image)
        axes[1, 0].imshow(colors[y_pred1[superpixels]], vmin=0, vmax=23,
                          alpha=.7)

        perf = msrc.eval_pixel_performance([image_name],
                                           [y_pred2[superpixels]],
                                           print_results=False)['global']
        axes[1, 1].set_title("%.2f" % (100 * perf))
        axes[1, 1].imshow(image)
        axes[1, 1].imshow(colors[y_pred2[superpixels]], alpha=.7)
        present_y = np.unique(np.hstack([y_pred1, y_pred2]))
        axes[0, 2].imshow(colors[present_y, :][:, np.newaxis, :],
                          interpolation='nearest')
        for i, c in enumerate(present_y):
            axes[0, 2].text(1, i, classes[c])
        for ax in axes.ravel():
            ax.set_xticks(())
            ax.set_yticks(())
        axes[1, 2].set_visible(False)
        fig.savefig(folder + "/%s.png" % image_name, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
