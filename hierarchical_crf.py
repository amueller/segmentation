import numpy as np
#import matplotlib.pyplot as plt
#from scipy import sparse

from hierarchical_segmentation import get_segment_features

from msrc_first_try import load_data, plot_results

from sklearn.svm import LinearSVC
#from sklearn.grid_search import GridSearchCV

from IPython.core.debugger import Tracer

tracer = Tracer()


def svm_on_segments():
    X, Y, image_names, images, all_superpixels = load_data(
        "train", independent=False)

    segment_features = [get_segment_features(*stuff)
                        for stuff in zip(X, Y, images, all_superpixels)]
    all_segments, all_features, all_labels = zip(*segment_features)

    features, labels = np.vstack(all_features), np.hstack(all_labels)

    svm = LinearSVC(C=.0001)
    svm.fit(features[labels != 21], labels[labels != 21])
    print("training score: %f"
          % svm.score(features[labels != 21], labels[labels != 21]))
    #grid_search = GridSearchCV(svm, param_grid={'C': 10. ** np.arange(-3, 3)},
                               #compute_training_score=True, verbose=10,
                               #n_jobs=8)
    #grid_search.fit(features[labels != 21], labels[labels != 21])
    #tracer()

    X_val, Y_val, image_names_val, images_val, all_superpixels_val = load_data(
        "val", independent=False)
    #Y_pred = [svm.predict(feats) for feats in all_features]

    #all_segments = [segments[sps]
                    #for segments, sps in zip(all_segments, all_superpixels)]

    #plot_results(images, image_names, all_labels, Y_pred, all_segments,
                 #folder="figures_segments", use_colors_predict=True)

    segment_features_val = [get_segment_features(*stuff) for stuff in
                            zip(X_val, Y_val, images_val, all_superpixels_val)]
    all_segments_val, all_features_val, all_labels_val =\
        zip(*segment_features_val)
    features_val, labels_val = (np.vstack(all_features_val),
                                np.hstack(all_labels_val))

    print("validation score: %f"
          % svm.score(features_val[labels_val != 21],
                      labels_val[labels_val != 21]))
    Y_pred_val = [svm.predict(feats) for feats in all_features]
    plot_results(images_val, image_names_val, all_labels_val, Y_pred_val,
                 all_segments_val, folder="figures_segments_val",
                 use_colors_predict=True)
    tracer()

if __name__ == "__main__":
    svm_on_segments()
