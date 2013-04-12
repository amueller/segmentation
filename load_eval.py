import sys

import matplotlib.pyplot as plt
import numpy as np

#from sklearn.metrics import confusion_matrix

from pystruct.utils import SaveLogger
from pystruct.problems import LatentNodeCRF

from msrc_first_try import eval_on_pixels, load_data
from msrc_helpers import classes, plot_results
from hierarchical_crf import make_hierarchical_data
from hierarchical_segmentation import plot_results_hierarchy


def main():
    argv = sys.argv
    print("loading %s ..." % argv[1])
    ssvm = SaveLogger(file_name=argv[1]).load()
    print(ssvm)
    try:
        inference_run = ~np.array(ssvm.cached_constraint_)
        print("Iterations: %d" % len(ssvm.objective_curve_))
        print("Dual objective: %f" % ssvm.objective_curve_[-1])
        print("Gap: %f" %
              (np.array(ssvm.primal_objective_curve_)[inference_run][-1] -
               ssvm.objective_curve_[-1]))
    except:
        pass
    if not hasattr(ssvm.problem, 'class_weight'):
        ssvm.problem.class_weight = np.ones(21)

    print("class weights: %s" % ssvm.problem.class_weight)
    if len(argv) <= 2:
        return

    if argv[2] == 'acc':

        #for data_str, title in zip(["train", "val", "test"],
                   #["TRAINING SET", "VALIDATION SET", "TEST SET"]):
        for data_str, title in zip(["train", "val"],
                                   ["TRAINING SET", "VALIDATION SET"]):
            print(title)
            data = load_data(data_str)
            if isinstance(ssvm.problem, LatentNodeCRF):
                data = make_hierarchical_data(data, lateral=True, latent=True)
            Y_pred = ssvm.predict(data.X)
            if isinstance(ssvm.problem, LatentNodeCRF):
                Y_pred = [ssvm.problem.label_from_latent(h) for h in Y_pred]
            #print("Predicted classes")
            #print(["%s: %.2f" % (c, x)
                   #for c, x in zip(classes, np.bincount(np.hstack(Y_pred)))])
            Y_flat = np.hstack(data.Y)
            print("superpixel accuracy: %s"
                  % np.mean((np.hstack(Y_pred) == Y_flat)[Y_flat != 21]))
            results = eval_on_pixels(data, Y_pred)
            print("global: %f, average: %f"
                  % (results['global'], results['average']))
            print(["%s: %.2f" % (c, x)
                   for c, x in zip(classes, results['per_class'])])

    elif argv[2] == 'curves':
        fig, axes = plt.subplots(1, 3)
        axes[0].plot(ssvm.objective_curve_)
        axes[0].plot(ssvm.primal_objective_curve_)
        # if we pressed ctrl+c in a bad moment
        inference_run = inference_run[:len(ssvm.objective_curve_)]
        axes[1].plot(np.array(ssvm.objective_curve_)[inference_run])
        axes[1].plot(np.array(ssvm.primal_objective_curve_)[inference_run])
        axes[2].plot(ssvm.loss_curve_)
        plt.show()

    elif argv[2] == 'plot':
        if len(argv) <= 3:
            raise ValueError("Need a folder name for plotting.")
        data = load_data("val")
        if isinstance(ssvm.problem, LatentNodeCRF):
            data = make_hierarchical_data(data, lateral=True, latent=True)
        Y_pred = ssvm.predict(data.X)
        if isinstance(ssvm.problem, LatentNodeCRF):
            plot_results_hierarchy(data, Y_pred, argv[3])
        else:
            plot_results(data, Y_pred, argv[3])


if __name__ == "__main__":
    main()
