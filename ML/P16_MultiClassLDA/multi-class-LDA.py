"""
Checkout below url on Multi-Class LDA
https://multivariatestatsjl.readthedocs.io/en/latest/mclda.html
"""
import mindspore
import mindspore as ms
import numpy as np
from mindspore import context, Tensor, nn

from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from mindspore.scipy import linalg as lg
print(mindspore.__version__)

ms.set_seed(0)
context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")

class MultiClassLDA:
    def __init__(self, solver='svd'):
        self.solver = solver

    def covariance_matrix(self, X):
        """
        :param X: Input tensor
        :return: cavariance of input tensor
        """
        centering_X = X - ms.ops.mean(X, axis=0)
        cov = ms.ops.mm(centering_X.transpose(), centering_X) / (centering_X.shape[0] - 1)
        return cov

    def scatter_matrix(self, X, y):
        """
        :param X: Input tensor
        :param y: Output tensor
        :return: How features are related to each other in within-class distribution and between class distribution
        """
        n_features = X.shape[1]
        labels = ms.ops.unique(y)[0]

        # Within-Class Scatter Matrix
        sw = ms.ops.zeros((n_features, n_features))
        for label in labels:
            X_class = X[y==label]
            sw += (X_class.shape[0] - 1) * self.covariance_matrix(X_class)

        # Between-Class Scatter Matrix
        n_samples_mean = ms.ops.sum(X, dim=0)
        sb = ms.ops.zeros((n_features, n_features))
        for label in labels:
            X_class = X[y==label]
            mean_class = ms.ops.mean(X_class, axis=0).unsqueeze(0)

            sb += (X_class.shape[0]) * ms.ops.mm(
                (mean_class - n_samples_mean), (mean_class - n_samples_mean).transpose())

        return sw, sb

    def transform(self, X, y, n_components):
        """
        And Why Inverse, In matrices, there is no concepts of division, thus multiplying with inverse
        matrix helps in acheiving what division does.
        :param X:
        :param y:
        :param n_components: Transforming from high dimension data to lower dimension n_components.
        :return: Transformed set of low dimensional X matrix
        """
        sw, sb = self.scatter_matrix(X, y)
        A = ms.ops.mm(ms.ops.pinv(sw), sb)
        eigen_values, eigen_vectors = lg.eigh(A)

        print('eigen_values: ', eigen_values)
        print('eigen_vectors: ', eigen_vectors)
        eigen_sorted_index = ms.ops.argsort(eigen_values, descending=True) # eigen_values[:, 0]
        eigen_vectors_sorted = eigen_vectors[:, eigen_sorted_index]
        component_vector = eigen_vectors_sorted[:, 0:n_components]
        component_vector = Tensor(component_vector, dtype=ms.float32)      #   torch.DoubleTensor)
        transformed = ms.ops.mm(X, component_vector)
        return transformed

    def plot_in_2d(self, X, y, title=None):
        """ Plot the dataset X and the corresponding labels y in 2D using the LDA
        transformation."""
        X_transformed = self.transform(X, y, n_components=2)
        x1 = X_transformed[:, 0]
        x2 = X_transformed[:, 1]
        plt.scatter(x1, x2, c=y)
        if title: plt.title(title)
        plt.show()


if __name__ == '__main__':
    data = load_iris()
    X = Tensor(data.data, dtype=ms.float32)
    y = Tensor(data.target)
    mclda = MultiClassLDA()
    mclda.plot_in_2d(X, y)
    exit(0)