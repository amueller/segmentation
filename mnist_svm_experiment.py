from sklearn.datasets import fetch_mldata
from sklearn.utils import shuffle

from pystruct.problems import CrammerSingerSVMProblem
#from pystruct.learners import SubgradientStructuredSVM
#from pystruct.learners import StructuredSVM
from pystruct.learners import OneSlackSSVM

mnist = fetch_mldata("MNIST original")

X, y = mnist.data, mnist.target
X = X / 255.

X_train, y_train = X[:60000], y[:60000]
X_test, y_test = X[60000:], y[60000:]

X_train, y_train = shuffle(X_train, y_train)

pblm = CrammerSingerSVMProblem(n_classes=10, n_features=28 ** 2)
#svm = SubgradientStructuredSVM(pblm, verbose=10, n_jobs=1, plot=True,
                               #max_iter=10, batch=False, learning_rate=0.0001,
                               #momentum=0)
#svm = SubgradientStructuredSVM(pblm, verbose=10, n_jobs=1, plot=True,
                               #max_iter=2, batch=False, momentum=.9,
                               #learning_rate=0.001, show_loss='true', C=1000)
svm = OneSlackSSVM(pblm, verbose=2, n_jobs=1, plot=True, max_iter=2, C=1000)
#svm = StructuredSVM(pblm, verbose=50, n_jobs=1, plot=True, max_iter=10,
#C=1000)
svm.fit(X_train, y_train)
print(svm.score(X_train, y_train))
print(svm.score(X_test, y_test))
