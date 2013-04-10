import sys

import matplotlib.pyplot as plt
import numpy as np

#from sklearn.metrics import confusion_matrix

from pystruct.utils import SaveLogger
from pystruct.problems import LatentNodeCRF

from msrc_first_try import eval_on_pixels, load_data
from msrc_helpers import classes
from hierarchical_crf import make_hierarchical_data


def main():
    argv = sys.argv
    print("loading %s ..." % argv[1])
    ssvm = SaveLogger(file_name=argv[1]).load()
    inference_run = ~np.array(ssvm.cached_constraint_)
    print(ssvm)
    print("Iterations: %d" % len(ssvm.objective_curve_))
    print("Dual objective: %f" % ssvm.objective_curve_[-1])
    print("Gap: %f" %
          (np.array(ssvm.primal_objective_curve_)[inference_run][-1] -
           ssvm.objective_curve_[-1]))
    if not hasattr(ssvm.problem, 'class_weight'):
        ssvm.problem.class_weight = np.ones(21)

    print("class weights: %s" % ssvm.problem.class_weight)
    if len(argv) <= 2:
        return

    if argv[2] == 'acc':

        for data_str, title in zip(["train", "val"],
                                   ["TRAINING SET", "VALIDATION SET"]):
            print(title)
            data = load_data(data_str)
            if isinstance(ssvm.problem, LatentNodeCRF):
                X, Y = make_hierarchical_data(data, lateral=False, latent=True)
            else:
                X, Y = data.X, data.Y
            Y_pred = ssvm.predict(X)
            if isinstance(ssvm.problem, LatentNodeCRF):
                Y_pred = [ssvm.problem.label_from_latent(h) for h in Y_pred]
            #print("Predicted classes")
            #print(["%s: %.2f" % (c, x)
                   #for c, x in zip(classes, np.bincount(np.hstack(Y_pred)))])
            Y_flat = np.hstack(Y)
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


if __name__ == "__main__":
    main()