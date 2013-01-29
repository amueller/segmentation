import numpy as np
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split

from pystruct.problems import LatentDirectionalGridCRF
from pystruct.learners import LatentSSVM

import pystruct.toy_datasets as toy

from IPython.core.debugger import Tracer
tracer = Tracer()


def main():
    # get some data
    X, Y = toy.generate_bars(n_samples=40, noise=10, total_size=10,
                             separate_labels=False)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.5)

    # train a latent grid crf
    n_labels = len(np.unique(Y_train))
    crf = LatentDirectionalGridCRF(n_labels=n_labels,
                                   n_states_per_label=[1, 6],
                                   inference_method='lp')
    clf = LatentSSVM(problem=crf, max_iter=50, C=10., verbose=2,
                     check_constraints=True, n_jobs=-1, break_on_bad=False,
                     tol=-10, base_svm='1-slack')
    clf.fit(X_train, Y_train)

    # the rest is plotting
    for X_, Y_, H, name in [[X_train, Y_train, clf.H_init_, "train"],
                            [X_test, Y_test, [None] * len(X_test), "test"]]:
        Y_pred = clf.predict(X_)
        i = 0
        loss = 0
        for x, y, h_init, y_pred in zip(X_, Y_, H, Y_pred):
            loss += np.sum(y != y_pred)
            fig, ax = plt.subplots(3, 2)
            ax[0, 0].matshow(y,
                             vmin=0, vmax=crf.n_labels - 1)
            ax[0, 0].set_title("ground truth")
            unary_pred = np.argmax(x, axis=-1)
            ax[0, 1].matshow(unary_pred, vmin=0, vmax=crf.n_labels - 1)
            ax[0, 1].set_title("unaries only")
            if h_init is None:
                ax[1, 0].set_visible(False)
            else:
                ax[1, 0].matshow(h_init, vmin=0, vmax=crf.n_states - 1)
                ax[1, 0].set_title("latent initial")
            ax[1, 1].matshow(crf.latent(x, y, clf.w),
                             vmin=0, vmax=crf.n_states - 1)
            ax[1, 1].set_title("latent final")
            ax[2, 0].matshow(crf.inference(x, clf.w),
                             vmin=0, vmax=crf.n_states - 1)
            ax[2, 0].set_title("prediction")
            ax[2, 1].matshow(y_pred, vmin=0, vmax=crf.n_labels - 1)
            ax[2, 1].set_title("prediction")
            for a in ax.ravel():
                a.set_xticks(())
                a.set_yticks(())
            fig.savefig("data_%s_%03d.png" % (name, i), bbox_inches="tight")
            i += 1
        print("loss %s set: %f" % (name, loss))
    #print(clf.w)

if __name__ == "__main__":
    main()
