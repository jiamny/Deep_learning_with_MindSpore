import mindspore as ms
from mindspore import context, Tensor, nn

from sklearn.metrics import accuracy_score
import numpy as np

class SquareLoss:
    def __init__(self):
        pass

    def loss(self, y, y_pred):
        return 0.5 * ms.ops.pow((y - y_pred), 2)

    def gradient(self, y, y_pred):
        return -(y - y_pred)

class CrossEntropy:
    def __init__(self):
        pass

    def loss(self, y, p):
        p = ms.ops.clip(p, 1e-15, 1 - 1e-15)
        return - y * ms.ops.log(p) - (1 - y) * ms.ops.log(1 - p)

    def accuracy(self, y, p):
        return accuracy_score(ms.ops.argmax(y, dim=1), ms.ops.argmax(p, dim=1))

    def gradient(self, y, p):
        p = ms.ops.clip(p, 1e-15, 1 - 1e-15)
        return -(y/p) + (1-y) / (1-p)

def euclidean_distance(x1, x2):
    """
    :param x1: input tensor
    :param x2: input tensor
    :return: distance between tensors
    """

    return ms.ops.cdist(x1.unsqueeze(0), x2.unsqueeze(0))

def to_categorical(X, n_columns=None):
    if not n_columns:
        n_columns = ms.ops.amax(X) + 1
    one_hot = ms.ops.zeros((X.shape[0], n_columns))
    one_hot[ms.ops.arange(X.shape[0])] = 1
    return one_hot

def mean_squared_error(y_true, y_pred):
    mse = ms.ops.mean(ms.ops.pow(y_true - y_pred, 2))
    return mse

def divide_on_feature(X, feature_i, threshold):

    split_func = None
    if isinstance(threshold, int) or isinstance(threshold, float):
        split_func = lambda sample: sample[feature_i] >= threshold
    else:
        split_func = lambda sample: sample[feature_i] == threshold


    X_1 = Tensor([sample.numpy() for sample in X if split_func(sample)])
    X_2 = Tensor([sample.numpy() for sample in X if not split_func(sample)])

    return np.array([X_1.numpy(), X_2.numpy()], dtype='object')

def calculate_variance(X):
    mean = ms.ops.ones(X.shape) * ms.ops.mean(X, axis=0)
    n_samples = X.shape[0]
    variance = (1/ n_samples) * ms.ops.diag(ms.ops.mm((X-mean).T, (X-mean)))
    return variance
