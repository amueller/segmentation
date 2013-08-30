import numpy as np

from datasets.nyu import NYUSegmentation

from nyu_helpers import load_nyu_pixelwise, load_nyu
from latent_crf_experiments.utils import eval_on_pixels, eval_on_sp

from IPython.core.debugger import Tracer

tracer = Tracer()

def eval_pixel_prediction():
    dataset = NYUSegmentation()
    data = load_nyu_pixelwise('val')
    predictions = [np.argmax(x, axis=-1) for x in data.X]
    #from sklearn.metrics import confusion_matrix
    #asdf = np.hstack([y.ravel() for y in predictions])
    #asdf2 = np.hstack([y.ravel() for y in data.Y])
    #print(confusion_matrix(asdf2, asdf))

    #tracer()
    hamming, jaccard = eval_on_pixels(dataset, data.Y, predictions, print_results=True)


def eval_sp_prediction():
    data = load_nyu('val')
    predictions = [np.argmax(x, axis=-1) for x in data.X]
    hamming, jaccard = eval_on_sp(data, predictions, print_results=True)
    tracer()

if __name__ == "__main__":
    eval_pixel_prediction()
