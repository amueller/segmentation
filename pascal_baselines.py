import numpy as np
import matplotlib.pyplot as plt

from skimage.segmentation import mark_boundaries
from sklearn.svm import LinearSVC

from pascal_helpers import (load_pascal, load_image, cmap, eval_on_pixels,
                            load_kraehenbuehl, eval_on_sp, gt_in_sp,
                            get_ground_truth, load_pascal_pixelwise)


from IPython.core.debugger import Tracer
tracer = Tracer()
np.set_printoptions(precision=2)


def train_svm(C=0.1):
    data_train = load_pascal()
    svm = LinearSVC(C=C, dual=False, class_weight='auto')
    X = np.vstack(data_train.X)
    y = np.hstack(data_train.y)
    tracer()

    svm.fit(X, y)
    print(svm.score(X, y))
    eval_on_sp(data_train, [svm.predict(x) for x in data_train.X],
               print_results=True)


def visualize_pascal(plot_probabilities=False):
    data = load_pascal('val')
    for x, y, f, sps in zip(data.X, data.Y, data.file_names, data.superpixels):
        fig, ax = plt.subplots(2, 3)
        ax = ax.ravel()
        image = load_image(f)
        y_pixel = get_ground_truth(f)
        x_raw = load_kraehenbuehl(f)

        boundary_image = mark_boundaries(image, sps)

        ax[0].imshow(image)
        ax[1].imshow(y_pixel, cmap=cmap)
        ax[2].imshow(boundary_image)
        ax[3].imshow(np.argmax(x_raw, axis=-1), cmap=cmap, vmin=0, vmax=256)
        ax[4].imshow(y[sps], cmap=cmap, vmin=0, vmax=256)
        ax[5].imshow(np.argmax(x, axis=-1)[sps], cmap=cmap, vmin=0, vmax=256)
        for a in ax:
            a.set_xticks(())
            a.set_yticks(())
        plt.savefig("figures_pascal_val/%s.png" % f, bbox_inches='tight')
        plt.close()
        if plot_probabilities:
            fig, ax = plt.subplots(3, 7)
            for k in range(21):
                ax.ravel()[k].matshow(x[:, :, k], vmin=0, vmax=1)
            for a in ax.ravel():
                a.set_xticks(())
                a.set_yticks(())
            plt.savefig("figures_pascal_val/%s_prob.png" % f,
                        bbox_inches='tight')
            plt.close()
    tracer()


def eval_pixel_prediction():
    data = load_pascal_pixelwise('val')
    predictions = [np.argmax(x, axis=-1) for x in data.X]
    hamming, jaccard = eval_on_pixels(data.Y, predictions, print_results=True)
    tracer()


def eval_sp_prediction():
    data = load_pascal('val')
    predictions = [np.argmax(x, axis=-1) for x in data.X]
    hamming, jaccard = eval_on_sp(data, predictions, print_results=True)
    tracer()


def eval_pixel_best_possible():
    data = load_pascal('val')
    Y_on_sp = [gt_in_sp(f, sp) for f, sp in zip(data.file_names,
                                                data.superpixels)]
    hamming, jaccard = eval_on_sp(data, Y_on_sp, print_results=True)
    tracer()

if __name__ == "__main__":
    #for C in 10. ** np.arange(-4, 2):
        #main(C)
    #main(.01, test=False)
    #visualize_pascal()
    #eval_pixel_best_possible()
    #eval_pixel_prediction()
    eval_sp_prediction()
