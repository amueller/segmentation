import numpy as np
import matplotlib.pyplot as plt

from datasets.msrc import MSRCDataset, classes
from msrc_helpers import (load_kraehenbuehl, load_data, eval_on_pixels,
                          get_kraehenbuehl_pot_sp)


def pixelwise():
    msrc = MSRCDataset()
    train = msrc.get_split('val')
    predictions = []
    for filename in train:
        probs = load_kraehenbuehl(filename)
        prediction = np.argmax(probs, axis=-1)
        prediction -= 1
        prediction[prediction == -1] = 21
        predictions.append(prediction)

    results = msrc.eval_pixel_performance(train, predictions)
    #plt.matshow(results['confusion'])
    #plt.show()
    print("global: %f, average: %f" % (results['global'], results['average']))
    print(["%s: %.2f" % (c, x) for c, x in zip(classes, results['per_class'])])


def on_slic_superpixels():
    msrc = MSRCDataset()
    data = load_data('test', independent=True)
    predictions = []
    probs = get_kraehenbuehl_pot_sp(data)
    for superpixels, prob in zip(data.superpixels, probs):
        sp_prediction = np.argmax(prob, axis=-1)
        sp_prediction -= 1
        sp_prediction[sp_prediction == -1] = 21
        predictions.append(sp_prediction[superpixels])
    results = msrc.eval_pixel_performance(data.file_names, predictions)
    plt.matshow(results['confusion'])
    plt.show()


def with_aureliens_potentials_svm():
    data = load_data('test', independent=True)
    new_features = []
    sp_probas = get_kraehenbuehl_pot_sp(data)
    new_features = [np.hstack([x[0], probas])
                    for x, probas in zip(data.X, sp_probas)]
    new_features = np.vstack(new_features)
    y = np.hstack(data.Y)
    from IPython.core.debugger import Tracer
    Tracer()()
    from sklearn.linear_model import LogisticRegression
    svm = LogisticRegression(C=.001, dual=False, class_weight='auto')
    svm.fit(new_features[y != 21], y[y != 21])
    eval_on_pixels(data, [svm.predict(x) for x in new_features])
    #msrc = MSRCDataset()


if __name__ == "__main__":
    on_slic_superpixels()
    #with_aureliens_potentials_svm()
    #pixelwise()
