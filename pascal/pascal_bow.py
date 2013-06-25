import numpy as np

from sklearn.svm import LinearSVC

#from sklearn.cross_validation import LeavePLabelOut
#from sklearn.grid_search import GridSearchCV
#from sklearn.utils import shuffle
#from sklearn.metrics import recall_score, Scorer
from slic_python import slic_n

from latent_crf_experiments.utils import eval_on_sp
from latent_crf_experiments.bow import SiftBOW

from datasets.pascal import PascalSegmentation

from IPython.core.debugger import Tracer
tracer = Tracer()


def train_svm(C=0.1, grid=False):
    pascal = PascalSegmentation()

    files_train = pascal.get_split("kTrain")
    superpixels = [slic_n(pascal.get_image(f), n_superpixels=100,
                          compactness=10)
                   for f in files_train]
    bow = SiftBOW(pascal, n_words=1000)
    data_train = bow.fit_transform(files_train, superpixels)

    svm = LinearSVC(C=C, dual=False, class_weight='auto')

    X, y = np.vstack(data_train.X), np.hstack(data_train.Y)
    svm.fit(X, y)
    print(svm.score(X, y))
    eval_on_sp(pascal, data_train, [svm.predict(x) for x in data_train.X],
               print_results=True)

    files_val = pascal.get_split("kVal")
    superpixels_val = [slic_n(pascal.get_image(f), n_superpixels=100,
                              compactness=10) for f in files_val]
    data_val = bow.transform(files_val, superpixels_val)
    eval_on_sp(pascal, data_val, [svm.predict(x) for x in data_val.X],
               print_results=True)

    tracer()

if __name__ == "__main__":
    train_svm(C=1, grid=False)
