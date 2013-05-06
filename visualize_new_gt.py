
import numpy as np
import matplotlib.pyplot as plt

from datasets.msrc import MSRCDataset, colors, classes

from IPython.core.debugger import Tracer
tracer = Tracer()


def main():
    msrc = MSRCDataset()
    images = msrc.get_split()
    for image_name in images:
        image = msrc.get_image(image_name)
        fig, axes = plt.subplots(1, 4, figsize=(12, 6))
        axes[0].imshow(image)
        axes[1].set_title("ground truth")
        axes[1].imshow(image)
        gt = msrc.get_ground_truth(image_name)
        axes[1].imshow(colors[gt], alpha=.7)
        axes[2].set_title("new ground_truth")
        gt_new = msrc.get_ground_truth(image_name, ds="new")
        axes[2].imshow(image)
        axes[2].imshow(colors[gt_new], vmin=0, vmax=23, alpha=.7)
        present_y = np.unique(np.hstack([gt.ravel(), gt_new.ravel()]))
        axes[3].imshow(colors[present_y, :][:, np.newaxis, :],
                       interpolation='nearest')
        for i, c in enumerate(present_y):
            axes[3].text(1, i, classes[c])
        for ax in axes.ravel():
            ax.set_xticks(())
            ax.set_yticks(())
        fig.savefig("new_gt/%s.png" % image_name, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
