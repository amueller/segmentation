import numpy as np

from sklearn.metrics import confusion_matrix

from datasets.nyu import NYUSegmentation

from nyu_helpers import load_nyu_pixelwise, load_nyu
from latent_crf_experiments.utils import eval_on_pixels, eval_on_sp

from IPython.core.debugger import Tracer

tracer = Tracer()

def eval_pixel_prediction():
    dataset = NYUSegmentation()
    data = load_nyu_pixelwise('val')
    predictions = [np.argmax(x, axis=-1) for x in data.X]
    hamming, jaccard = eval_on_pixels(dataset, data.Y, predictions, print_results=True)


def eval_sp_prediction():
    dataset = NYUSegmentation()
    data = load_nyu('val')
    predictions = [np.argmax(x, axis=-1) for x in data.X]
    #for p, y, f, sp in zip(predictions, data.Y, data.file_names,
                           #data.superpixels):
        #fig, ax = plt.subplots(3)
        #ax[0].imshow(dataset.get_ground_truth(f), vmin=0, vmax=5)
        #ax[1].imshow(y[sp], vmin=0, vmax=5)
        #ax[2].imshow(p[sp], vmin=0, vmax=5)
    #plt.show()
    y_true = np.hstack(data.Y)
    y_pred = np.hstack(predictions)
    print(confusion_matrix(y_true, y_pred))
    hamming, jaccard = eval_on_sp(dataset, data, predictions, print_results=True)
    tracer()

if __name__ == "__main__":
    #eval_pixel_prediction()
    eval_sp_prediction()
