import numpy as np
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split

from pystruct.problems import LatentGridCRF
from pystruct.learners import LatentSSVM

import pystruct.toy_datasets as toy

from IPython.core.debugger import Tracer
tracer = Tracer()


def main():
    X, Y = toy.generate_crosses(n_samples=20, noise=5, n_crosses=1,
                                total_size=8)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.5)
    n_labels = len(np.unique(Y_train))
    crf = LatentGridCRF(n_labels=n_labels, n_states_per_label=[1, 2],
                        inference_method='lp')
    clf = LatentSSVM(problem=crf, max_iter=50, C=1000., verbose=2,
                     check_constraints=True, n_jobs=-1, break_on_bad=True)
    clf.fit(X_train, Y_train)

    i = 0
    for X_, Y_, H, name in [[X_train, Y_train, clf.H_init_, "train"],
                            [X_test, Y_test, [None] * len(X_test), "test"]]:
        Y_pred = clf.predict(X_)
        score = clf.score(X_, Y_)
        for x, y, h_init, y_pred in zip(X_, Y_, H, Y_pred):
            fig, ax = plt.subplots(4, 1)
            ax[0].matshow(y, vmin=0, vmax=crf.n_labels - 1)
            ax[0].set_title("Ground truth")
            ax[1].matshow(np.argmax(x, axis=-1), vmin=0, vmax=crf.n_labels - 1)
            ax[1].set_title("Unaries only")
            #if h_init is None:
                #ax[1, 0].set_visible(False)
            #else:
                #ax[1, 0].matshow(h_init, vmin=0, vmax=crf.n_states - 1)
                #ax[1, 0].set_title("latent initial")
            #ax[2].matshow(crf.latent(x, y, clf.w),
                          #vmin=0, vmax=crf.n_states - 1)
            #ax[2].set_title("latent final")
            ax[2].matshow(crf.inference(x, clf.w), vmin=0, vmax=crf.n_states
                          - 1)
            ax[2].set_title("Prediction for h")
            ax[3].matshow(y_pred, vmin=0, vmax=crf.n_labels - 1)
            ax[3].set_title("Prediction for y")
            for a in ax.ravel():
                a.set_xticks(())
                a.set_yticks(())
            plt.subplots_adjust(hspace=.5)
            fig.savefig("data_%s_%03d.png" % (name, i), bbox_inches="tight",
                        dpi=400)
            i += 1
        print("score %s set: %f" % (name, score))
    print(clf.w)

if __name__ == "__main__":
    main()
