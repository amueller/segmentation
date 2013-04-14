import numpy as np
import matplotlib.pyplot as plt

from datasets.msrc import MSRCDataset
from msrc_helpers import (load_kraehenbuehl, load_data, eval_on_pixels,
                          get_kraehenbuehl_pot_sp, DataBunch)


def sigm(x):
    return 1. / (1 + np.exp(-x))


def pixelwise():
    msrc = MSRCDataset()
    train = msrc.get_split('val')
    predictions = []
    for filename in train:
        probs = load_kraehenbuehl(filename)
        prediction = np.argmax(probs, axis=-1)
        predictions.append(prediction)

    msrc.eval_pixel_performance(train, predictions)
    #plt.matshow(results['confusion'])
    #plt.show()


def on_slic_superpixels():
    data = load_data('test', independent=True)
    probs = get_kraehenbuehl_pot_sp(data)
    results = eval_on_pixels(data, [np.argmax(prob, axis=-1) for prob in
                                    probs])
    plt.matshow(results['confusion'])
    plt.show()


def add_kraehenbuehl_features(data):
    sp_probas = get_kraehenbuehl_pot_sp(data)
    X = [(np.hstack([sigm(x[0]), probas]), x[1])
         for x, probas in zip(data.X, sp_probas)]
    return DataBunch(X, data.Y, data.file_names, data.images, data.superpixels)


def with_aureliens_potentials_svm(test=False):
    data = load_data('train', independent=True)
    data = add_kraehenbuehl_features(data)
    features = [x[0] for x in data.X]
    y = np.hstack(data.Y)

    if test:
        data_ = load_data('val', independent=True)
        data_ = add_kraehenbuehl_features(data_)
        features.extend([x[0] for x in data.X])
        y = np.hstack([y, np.hstack(data_.Y)])

    new_features_flat = np.vstack(features)
    from sklearn.svm import LinearSVC
    print("training svm")
    svm = LinearSVC(C=.001, dual=False, class_weight='auto')
    svm.fit(new_features_flat[y != 21], y[y != 21])
    print(svm.score(new_features_flat[y != 21], y[y != 21]))
    print("evaluating")
    eval_on_pixels(data, [svm.predict(x) for x in features])

    if test:
        print("test data")
        data_val = load_data('test', independent=True)
    else:
        data_val = load_data('val', independent=True)

    data_val = add_kraehenbuehl_features(data_val)
    features_val = [x[0] for x in data_val.X]
    eval_on_pixels(data_val, [svm.predict(x) for x in features_val])
    #msrc = MSRCDataset()


if __name__ == "__main__":
    #on_slic_superpixels()
    with_aureliens_potentials_svm(test=True)
    #pixelwise()
