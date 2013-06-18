# a CRF with one node is the same as a multiclass SVM
# evaluation on iris dataset (really easy)

from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix

#from pystruct.problems import GraphCRF
from ignore_void_crf import IgnoreVoidCRF
#from pystruct.learners import OneSlackSSVM
from pystruct.learners import SubgradientStructuredSVM
#from pystruct.learners import StructuredSVM

#from IPython.core.debugger import Tracer
#tracer = Tracer()

iris = load_iris()
X, y = iris.data, iris.target

# make each example into a tuple of a single feature vector and an empty edge
# list
#X_ = [(np.atleast_2d(x), np.empty((0, 2), dtype=np.int)) for x in X]

X_train_org, X_test_org, y_train, y_test = \
    train_test_split(X, y)
X_train = [(X_train_org, np.empty((0, 2), dtype=np.int))]
y_train = y_train[np.newaxis, :]

X_test = [(X_test_org, np.empty((0, 2), dtype=np.int))]
y_test = y_test[np.newaxis, :]

inds = np.arange(y_train.size)
np.random.shuffle(inds)
y_train[0, inds[:20]] = 3
print(y_train.ravel())

pbl = IgnoreVoidCRF(n_features=4, n_states=4, inference_method='lp',
                    void_label=3)
#svm = StructuredSVM(pbl, verbose=1, check_constraints=True, C=100, n_jobs=1,
                    #max_iter=1000, tol=-10)
svm = SubgradientStructuredSVM(pbl, verbose=1, C=100, n_jobs=1, max_iter=1000,
                               learning_rate=.001)


start = time()
svm.fit(X_train, y_train)
time_svm = time() - start

print(svm.w)

y_pred = np.vstack(svm.predict(X_test))
print("Score with pystruct crf svm: %f (took %f seconds)"
      % (np.mean(y_pred == y_test), time_svm))
print(confusion_matrix(y_train.ravel(), np.hstack(svm.predict(X_train))))
print(confusion_matrix(y_test.ravel(), np.hstack(svm.predict(X_test))))
fig, axes = plt.subplots(2)

X_train_pca = PCA(n_components=2).fit_transform(X_train_org)
#plt.prism()
axes[0].scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, vmin=0,
                vmax=3)
axes[1].scatter(X_train_pca[:, 0], X_train_pca[:, 1],
                c=np.hstack(svm.predict(X_train)), vmin=0, vmax=3)
plt.show()
