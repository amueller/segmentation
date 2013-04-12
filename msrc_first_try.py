import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
#from sklearn.preprocessing import StandardScaler

#from datasets.msrc import MSRCDataset
from pystruct import learners
from pystruct.problems.latent_graph_crf import kmeans_init
from pystruct.problems import GraphCRF, LatentGraphCRF
from pystruct.utils import SaveLogger

from msrc_helpers import (classes, load_data, plot_results, discard_void,
                          eval_on_pixels)

#from ignore_void_crf import IgnoreVoidCRF

from IPython.core.debugger import Tracer
tracer = Tracer()


def plot_confusion_matrix(matrix, title=None):
    plt.matshow(matrix)
    plt.axis("off")
    plt.colorbar()
    for i, c in enumerate(classes[:-2]):
        plt.text(i, -1, c, rotation=60, va='bottom')
        plt.text(-1, i, c, ha='right')
    if title:
        plt.title(title)


def plot_parts():
    car_idx = np.where(classes == "car")[0]
    data = load_data("train", independent=False)
    car_images = np.array([i for i, y in enumerate(data.Y)
                           if np.any(y == car_idx)])
    flat_X = [x[0] for x in data.X]
    edges = [[x[1]] for x in data.X]
    n_states_per_label = np.ones(22, dtype=np.int)
    n_states_per_label[car_idx] = 6

    H = kmeans_init(flat_X, data.Y, edges, n_labels=22,
                    n_states_per_label=n_states_per_label, symmetric=True)
    X, Y, file_names, images, all_superpixels, H = zip(*[
        (data.X[i], data.Y[i], data.file_names[i], data.images[i],
         data.all_superpixels[i], H[i])
        for i in car_images])
    plot_results(images, file_names, Y, H, all_superpixels,
                 folder="test_parts", use_colors_predict=False)
    tracer()


def train_car_parts():
    car_idx = np.where(classes == "car")[0]
    data = load_data("train", independent=False)
    car_images = np.array([i for i, y in enumerate(data.Y)
                           if np.any(y == car_idx)])
    n_states_per_label = np.ones(22, dtype=np.int)
    n_states_per_label[car_idx] = 6

    X, Y, file_names, images, all_superpixels = zip(*[
        (data.X[i], data.Y[i], data.file_names[i], data.images[i],
         data.all_superpixels[i]) for i in car_images])
    problem = LatentGraphCRF(n_states_per_label=n_states_per_label,
                             n_labels=22, inference_method='ad3',
                             n_features=21 * 6)
    ssvm = learners.LatentSSVM(
        problem, verbose=2, C=10, max_iter=5000, n_jobs=-1, tol=0.0001,
        show_loss_every=10, base_svm='subgradient', inference_cache=50,
        latent_iter=5, learning_rate=0.001, decay_exponent=0.5)
    ssvm.fit(X, Y)
    plot_results(images, file_names, Y, ssvm.H_init_, all_superpixels,
                 folder="parts_init", use_colors_predict=False)
    H = ssvm.predict_latent(X)
    plot_results(images, file_names, Y, H, all_superpixels,
                 folder="parts_prediction", use_colors_predict=False)
    H_final = [problem.latent(x, y, ssvm.w) for x, y in zip(X, Y)]
    plot_results(images, file_names, Y, H_final, all_superpixels,
                 folder="parts_final", use_colors_predict=False)
    tracer()


def train_car():
    car_idx = np.where(classes == "car")[0]
    data_train = load_data("train", independent=False)
    car_images = np.array([i for i, y in enumerate(data_train.Y)
                           if np.any(y == car_idx)])
    n_states_per_label = np.ones(22, dtype=np.int)
    n_states_per_label[car_idx] = 6

    X, Y, file_names, images, all_superpixels = zip(*[
        (data_train.X[i], data_train.Y[i], data_train.file_names[i],
         data_train.images[i], data_train.superpixels[i])
        for i in car_images])
    problem = GraphCRF(n_states=22, inference_method='ad3', n_features=21 * 6)
    ssvm = learners.SubgradientStructuredSVM(
        problem, verbose=2, C=.001, max_iter=5000, n_jobs=-1,
        show_loss_every=10, learning_rate=0.0001, decay_exponent=0.5)
    ssvm.fit(X, Y)
    Y_pred = ssvm.predict(X)
    plot_results(images, file_names, Y, Y_pred, all_superpixels,
                 folder="cars_only")

    data_val = load_data("val", independent=False)
    car_images_val = np.array([i for i, y in enumerate(data_val.Y)
                               if np.any(y == car_idx)])
    X_val, Y_val, file_names_val, images_val, all_superpixels_val = \
        zip(*[(data_val.X[i], data_val.Y[i], data_val.file_names[i],
               data_val.images[i], data_val.superpixels[i]) for i in
              car_images_val])
    Y_pred_val = ssvm.predict(X_val)
    plot_results(images_val, file_names_val, Y_val, Y_pred_val,
                 all_superpixels_val, folder="cars_only_val")
    # C=10
    ## train:
    #0.92743060939680566V
    #> ssvm.score(X_val, Y_val)
    #0.52921719955898561
    # test 0.61693548387096775
    tracer()


def train_svm(test=False):
    data_train = load_data("train", independent=True)
    X_features = [x[0] for x in data_train.X]
    X_features_flat = np.vstack(X_features)
    y = np.hstack(data_train.Y)
    if test:
        data_val = load_data("val", independent=True)
        X_features_val = [x[0] for x in data_val.X]
        X_features_flat_val = np.vstack(X_features_val)
        y_val = np.hstack(data_val.Y)
        y = np.hstack([y, y_val])
        X_features_flat = np.vstack([X_features_flat, X_features_flat_val])
    #from sklearn.svm import LinearSVC
    #svm = LinearSVC(C=10000, dual=False, class_weight='auto', loss='l1')
    from sklearn.linear_model import LogisticRegression
    svm = LogisticRegression(C=.001, dual=False, class_weight='auto')
    #from pystruct.learners import OneSlackSSVM
    #from latent_crf_experiments.magic_svm.svm_definition \
        #import AureliensMagicSVM
    #class_weight = 1. / np.bincount(y)
    #class_weight *= len(class_weight) / np.sum(class_weight)
    #pbl = AureliensMagicSVM(class_weight=class_weight[:-1])
    #svm = OneSlackSSVM(pbl, n_jobs=1, verbose=2, max_iter=100000, C=0.01,
                       #show_loss_every=10, tol=0.00001,
                       #inactive_threshold=1e-4, inactive_window=50,
                       #check_constraints=True, break_on_bad=True)
    svm.fit(X_features_flat[y != 21], y[y != 21])

    results = eval_on_pixels(data_train, [svm.predict(x) for x in X_features])

    if test:
        data_test = load_data("test", independent=True)
    else:
        data_test = load_data("val", independent=True)

    X_features = [x[0][:, :5 * 12] for x in data_test.X]
    X_features_flat = np.vstack(X_features)
    y = np.hstack(data_test.Y)
    results = eval_on_pixels(data_test, [svm.predict(x) for x in X_features])
    results
    #plot_results(data_test, [svm.predict(x) for x in X_features],
                 #folder="linear_svc_no_global_001_val_class_weight")

    Tracer()()


def main():
    # load training data
    independent = False
    test = True
    data_train = load_data("train", independent=independent)
    X_, Y_ = discard_void(data_train.X, data_train.Y, 21)
    if test:
        data_val = load_data("val", independent=independent)
        X_val, Y_val = discard_void(data_val.X, data_val.Y, 21)

    X_.extend(X_val)
    Y_.extend(Y_val)

    n_states = 21
    print("number of samples: %s" % len(data_train.X))
    #class_weights = 1. / np.bincount(np.hstack(Y_))
    class_weights = np.ones(n_states)
    #class_weights *= 21. / np.sum(class_weights)
    print(class_weights)
    problem = GraphCRF(n_states=n_states, n_features=21 * 6,
                       inference_method='qpbo', class_weight=class_weights)
    #ssvm = learners.SubgradientStructuredSVM(
        #problem, verbose=2, C=.001, n_jobs=-1, max_iter=100000,
        #learning_rate=0.00015, show_loss_every=10, decay_exponent=.5,
        #momentum=0.98)
    ssvm = learners.OneSlackSSVM(
        problem, verbose=2, C=0.00001, max_iter=100000, n_jobs=-1,
        tol=0.0001, show_loss_every=200, inference_cache=50, cache_tol='auto',
        logger=SaveLogger("qpbo_0.00001_discard_void_train_val.pickle",
                          save_every=100),
        inactive_threshold=1e-5, break_on_bad=False)
    #ssvm = SaveLogger("qpbo_0.001_discard_void_train_val.pickle").load()
    #ssvm.logger = SaveLogger(file_name=
                     #"qpbo_0.001_discard_void_train_val_refit2.pickle",
                     #save_every=100)
    #ssvm.problem.class_weight = np.ones(ssvm.problem.n_states)
    #ssvm.problem.inference_method = 'ad3'
    #ssvm.cache_tol = 'auto'
    #ssvm.fit(X_, Y_, warm_start=True)
    ssvm.fit(X_, Y_)
    print("fit finished!")
    tracer()

    # do some evaluation on the training set
    print("score on training set: %f" % ssvm.score(X_, Y_))
    Y_pred = ssvm.predict(data_train.X)
    # compute confusion matrix
    confusion = np.zeros((n_states, n_states))
    for y, y_pred in zip(data_train.Y, Y_pred):
        confusion += confusion_matrix(y, y_pred, labels=np.arange(n_states))
    plot_confusion_matrix(confusion, title="confusion")

    # plot pairwise weights
    pairwise_flat = np.asarray(ssvm.w[problem.n_states * problem.n_features:])
    pairwise_params = np.zeros((problem.n_states, problem.n_states))
    pairwise_params[np.tri(problem.n_states, dtype=np.bool)] = pairwise_flat
    plot_confusion_matrix(pairwise_params, title="pairwise_params")
    plt.figure()
    plt.plot(ssvm.objective_curve_)
    plt.show()

    # make figures with predictions
    #plot_results(data_train.images, data_train.file_names, data_train.Y,
                 #Y_pred, data_train.all_superpixels, folder="figures_train")
    data_val = load_data("val", independent=independent)
    X_val_, Y_val_ = discard_void(data_val.X, data_val.Y, 21)
    print("score on validation set: %f" % ssvm.score(X_val_, Y_val_))
    Y_pred_val = ssvm.predict(data_val.X)
    plot_results(data_val, Y_pred_val, folder="figures_val")
    tracer()


if __name__ == "__main__":
    main()
    #train_svm()
    #plot_parts()
    #train_car_parts()
    #train_car()
