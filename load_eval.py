import sys

import matplotlib.pyplot as plt
import numpy as np

from pystruct.utils import SaveLogger

from msrc_first_try import eval_on_pixels, load_data
from msrc_helpers import classes


def main():
    argv = sys.argv
    print("loading %s ..." % argv[1])
    ssvm = SaveLogger(file_name=argv[1]).load()
    print(ssvm)
    print("Iterations: %d" % len(ssvm.objective_curve_))

    if len(argv) <= 2 or argv[2] == 'acc':

        for data_str, title in zip(["train", "val"],
                                   ["TRAINING SET", "VALIDATION SET"]):
            print(title)
            data = load_data(data_str)
            results = eval_on_pixels(data, ssvm.predict(data.X))
            print("global: %f, average: %f"
                  % (results['global'], results['average']))
            print(["%s: %.2f" % (c, x)
                   for c, x in zip(classes, results['per_class'])])

    elif argv[2] == 'curves':
        fig, axes = plt.subplots(1, 2)
        axes[0].plot(ssvm.objective_curve_)
        axes[0].plot(ssvm.primal_objective_curve_)
        inference_run = ~np.array(ssvm.cached_constraint_)
        axes[1].plot(np.array(ssvm.objective_curve_)[inference_run])
        axes[1].plot(np.array(ssvm.primal_objective_curve_)[inference_run])
        plt.show()


if __name__ == "__main__":
    main()
