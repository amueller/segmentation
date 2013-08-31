import numpy as np
#import matplotlib.pyplot as plt


from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score, Scorer
from sklearn.svm import LinearSVC
from sklearn.cross_validation import LeavePLabelOut
from sklearn.grid_search import GridSearchCV
from sklearn.utils import shuffle

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
    #for p, y, f, sp in zip(predictions, data.Y, data.file_names,
                           #data.superpixels):
        #fig, ax = plt.subplots(3)
        #ax[0].imshow(dataset.get_ground_truth(f), vmin=0, vmax=5)
        #ax[1].imshow(y[sp], vmin=0, vmax=5)
        #ax[2].imshow(p[sp], vmin=0, vmax=5)
    #plt.show()
    y_true = np.hstack(data.Y)
    y_pred = np.hstack(predictions)
    print(confusion_matrix(y_true, y_pred))
    hamming, jaccard = eval_on_sp(dataset, data, predictions, print_results=True)
    tracer()

def train_svm(C=0.1, grid=False):
    ds = NYUSegmentation()
    svm = LinearSVC(C=C, dual=False, class_weight='auto')

    if grid:
        data_train = load_nyu("train")
        X, y = shuffle(data_train.X, data_train.Y)
        # prepare leave-one-label-out by assigning labels to images
        image_indicators = np.hstack([np.repeat(i, len(x)) for i, x in
                                      enumerate(X)])
        # go down to only 5 "folds"
        labels = image_indicators % 5
        X, y = np.vstack(X), np.hstack(y)

        cv = LeavePLabelOut(labels=labels, p=1)
        param_grid = {'C': 10. ** np.arange(-3, 3)}
        scorer = Scorer(recall_score, average="macro")
        grid_search = GridSearchCV(svm, param_grid=param_grid, cv=cv,
                                   verbose=10, scoring=scorer, n_jobs=-1)
        grid_search.fit(X, y)
    else:
        data_train = load_nyu("train")
        X, y = np.vstack(data_train.X), np.hstack(data_train.Y)
        svm.fit(X, y)
        print(svm.score(X, y))
        eval_on_sp(ds, data_train, [svm.predict(x) for x in data_train.X],
                   print_results=True)

        data_val = load_nyu("val")
        eval_on_sp(ds, data_val, [svm.predict(x) for x in data_val.X],
                   print_results=True)

if __name__ == "__main__":
    #eval_pixel_prediction()
    #eval_sp_prediction()
    train_svm(C=100)
