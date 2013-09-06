import numpy as np
#import matplotlib.pyplot as plt


from sklearn.svm import LinearSVC

from datasets.nyu import NYUSegmentation

from nyu_helpers import load_nyu_pixelwise, load_nyu
from latent_crf_experiments.utils import eval_on_pixels, eval_on_sp

from IPython.core.debugger import Tracer

tracer = Tracer()

def eval_pixel_prediction():
    dataset = NYUSegmentation()
    data = load_nyu_pixelwise('val')
    predictions = [np.argmax(x, axis=-1) for x in data.X]
    hamming, jaccard = eval_on_pixels(dataset, data.Y, predictions, print_results=True)


def eval_sp_prediction():
    dataset = NYUSegmentation()
    data = load_nyu('val', n_sp=500, sp='rgbd')
    predictions = [np.argmax(x, axis=-1) for x in data.X]
    #predictions = data.Y
    hamming, jaccard = eval_on_sp(dataset, data, predictions, print_results=True)


def train_svm(C=0.1, grid=False):
    ds = NYUSegmentation()
    data_train = load_nyu("train", n_sp=500, sp='rgbd')
    svm = LinearSVC(C=C, dual=False, class_weight='auto')
    #N_train = []
    #for f, sp in zip(data_train.file_names, data_train.superpixels):
        #normals = ds.get_pointcloud_normals(f)[:, :, 3:]
        #mean_normals = get_sp_normals(normals, sp)
        #N_train.append(mean_normals * .1)
    #N_flat_train = np.vstack(N_train)

    X, y = np.vstack(data_train.X), np.hstack(data_train.Y)
    #X = np.hstack([X, N_flat_train])
    svm.fit(X, y)
    print(svm.score(X, y))
    eval_on_sp(ds, data_train, [svm.predict(x)
                                for x in data_train.X],
               print_results=True)

    data_val = load_nyu("val", n_sp=500, sp='rgbd')
    #N_val = []
    #for f, sp in zip(data_val.file_names, data_val.superpixels):
        #normals = ds.get_pointcloud_normals(f)[:, :, 3:]
        #mean_normals = get_sp_normals(normals, sp)
        #N_val.append(mean_normals * .1)
    eval_on_sp(ds, data_val, [svm.predict(x)
                                for x in data_val.X],
               print_results=True)

if __name__ == "__main__":
    #eval_pixel_prediction()
    #eval_sp_prediction()
    train_svm(C=1)
