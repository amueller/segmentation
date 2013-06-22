import numpy as np

from sklearn.svm import LinearSVC

#from sklearn.cross_validation import LeavePLabelOut
#from sklearn.grid_search import GridSearchCV
#from sklearn.utils import shuffle
#from sklearn.metrics import recall_score, Scorer
from slic_python import slic_n

from pascal_helpers import load_pascal

from latent_crf_experiments.utils import eval_on_sp
from latent_crf_experiments.bow import SiftBOW

from datasets.pascal import PascalSegmentation

from IPython.core.debugger import Tracer
tracer = Tracer()


def train_svm(C=0.1, grid=False):
    pascal = PascalSegmentation()

    files_train = pascal.get_split("train1")
    superpixels = [slic_n(pascal.get_image(f), n_superpixels=100,
                          compactness=10)
                   for f in files_train]
    bow = SiftBOW(pascal)
    data_train = bow.fit_transform(files_train, superpixels)
    tracer()

    svm = LinearSVC(C=C, dual=False, class_weight='auto')

    data_train = load_pascal("train1")
    X, y = np.vstack(data_train.X), np.hstack(data_train.Y)
    svm.fit(X, y)
    print(svm.score(X, y))
    eval_on_sp(data_train, [svm.predict(x) for x in data_train.X],
               print_results=True)

    data_val = load_pascal("train2")
    eval_on_sp(data_val, [svm.predict(x) for x in data_val.X],
               print_results=True)

    tracer()

if __name__ == "__main__":
    train_svm(C=0.001, grid=False)
