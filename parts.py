import numpy as np

#from sklearn.preprocessing import StandardScaler

#from datasets.msrc import MSRCDataset
from pystruct import learners
from pystruct.problems import GraphCRF, LatentGraphCRF

from msrc_helpers import classes, load_data, plot_results

from IPython.core.debugger import Tracer
tracer = Tracer()


def plot_parts():
    from pystruct.problems.latent_graph_crf import kmeans_init
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
