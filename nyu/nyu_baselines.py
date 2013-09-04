import numpy as np
#import matplotlib.pyplot as plt


from sklearn.metrics import confusion_matrix
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
    data = load_nyu('train')
    predictions = [np.argmax(x, axis=-1) for x in data.X]
    y_true = np.hstack(data.Y)
    y_pred = np.hstack(predictions)
    print(confusion_matrix(y_true, y_pred))
    hamming, jaccard = eval_on_sp(dataset, data, predictions, print_results=True)
    tracer()

def train_svm(C=0.1, grid=False):
    ds = NYUSegmentation()
    svm = LinearSVC(C=C, dual=False, class_weight='auto')

    data_train = load_nyu("train", n_sp=500, add_covariance=False)
    X, y = np.vstack(data_train.X), np.hstack(data_train.Y)
    svm.fit(X, y)
    print(svm.score(X, y))
    eval_on_sp(ds, data_train, [svm.predict(x) for x in data_train.X],
               print_results=True)

    data_val = load_nyu("val", n_sp=500, add_covariance=False)
    eval_on_sp(ds, data_val, [svm.predict(x) for x in data_val.X],
               print_results=True)

if __name__ == "__main__":
    #eval_pixel_prediction()
    #eval_sp_prediction()
    train_svm(C=1)
