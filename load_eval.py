#!/usr/bin/python
import sys

import matplotlib.pyplot as plt
import numpy as np

#from sklearn.metrics import confusion_matrix

from pystruct.utils import SaveLogger
from pystruct.models import LatentNodeCRF, EdgeFeatureGraphCRF

from msrc.msrc_helpers import (plot_results, add_edge_features, add_edges,
                               eval_on_pixels, load_data,
                               plot_confusion_matrix)
from msrc.hierarchical_crf import make_hierarchical_data
from msrc.hierarchical_segmentation import plot_results_hierarchy
#from hierarchical_helpers import load_data_global_probs
from msrc.msrc_helpers import add_kraehenbuehl_features


def main():
    argv = sys.argv
    print("loading %s ..." % argv[1])
    ssvm = SaveLogger(file_name=argv[1]).load()
    if hasattr(ssvm, 'problem'):
        ssvm.model = ssvm.problem
    print(ssvm)
    if hasattr(ssvm, 'base_ssvm'):
        ssvm = ssvm.base_ssvm
    print("Iterations: %d" % len(ssvm.objective_curve_))
    print("Objective: %f" % ssvm.objective_curve_[-1])
    inference_run = None
    if hasattr(ssvm, 'cached_constraint_'):
        inference_run = ~np.array(ssvm.cached_constraint_)
        print("Gap: %f" %
              (np.array(ssvm.primal_objective_curve_)[inference_run][-1] -
               ssvm.objective_curve_[-1]))

    if len(argv) <= 2:
        return

    if argv[2] == 'acc':

        ssvm.n_jobs = -1
        #for data_str, title in zip(["train", "val", "test"],
                                   #["TRAINING SET", "VALIDATION SET",
                                    #"TEST SET"]):
        for data_str, title in zip(["train", "val"],
                                   ["TRAINING SET", "VALIDATION SET"]):
            print(title)
            independent = True
            #independent = False
            if isinstance(ssvm.model, EdgeFeatureGraphCRF):
                independent = False

            if ssvm.model.inference_method == 'dai':
                independent = True
                print("DAI DAI DAI")
            #data = load_data_global_probs(data_str, latent=True)
            data = load_data(data_str, which="piecewise")
            data = add_edges(data, independent=independent)
            #data = add_kraehenbuehl_features(data, which="train_30px")
            #data = add_kraehenbuehl_features(data, which="train")
            # may Guido have mercy on my soul
            #(I renamed the module after pickling)
            if type(ssvm.model).__name__ == 'EdgeFeatureGraphCRF':
                data = add_edge_features(data)

            #if isinstance(ssvm.model, LatentNodeCRF):
                #data = make_hierarchical_data(data, lateral=True, latent=True,
                                              #latent_lateral=True)
            ssvm.model.inference_method = "qpbo"
            Y_pred = ssvm.predict(data.X)

            if isinstance(ssvm.model, LatentNodeCRF):
                Y_pred = [ssvm.model.label_from_latent(h) for h in Y_pred]
            Y_flat = np.hstack(data.Y)
            print("superpixel accuracy: %.2f"
                  % (np.mean((np.hstack(Y_pred) == Y_flat)[Y_flat != 21]) *
                     100))
            res = eval_on_pixels(data, Y_pred, print_results=True)
            print("global: %.2f, average: %.2f" % (res['global'] * 100,
                                                   res['average'] * 100))
            plot_confusion_matrix(res['confusion'])
        plt.show()

    elif argv[2] == 'plot':
        data_str = 'val'
        if len(argv) <= 3:
            raise ValueError("Need a folder name for plotting.")
        data = load_data(data_str, which="piecewise")
        data = add_edges(data, independent=False)
        data = add_kraehenbuehl_features(data, which="train_30px")
        data = add_kraehenbuehl_features(data, which="train")
        if type(ssvm.model).__name__ == 'EdgeFeatureGraphCRF':
            data = add_edge_features(data)
        #ssvm.model.inference_method = 'qpbo'
        if isinstance(ssvm.model, LatentNodeCRF):
            data = make_hierarchical_data(data, lateral=True, latent=True)
            try:
                Y_pred = ssvm.predict_latent(data.X)
            except AttributeError:
                Y_pred = ssvm.predict(data.X)

            plot_results_hierarchy(data, Y_pred, argv[3])
        else:
            Y_pred = ssvm.predict(data.X)
            plot_results(data, Y_pred, argv[3])


if __name__ == "__main__":
    main()
