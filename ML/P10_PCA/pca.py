import mindspore as ms
from mindspore import context, Tensor, nn

from sklearn.datasets import load_iris
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np

class pca:
    def __init__(self, n_components):
        """
        :param n_components: Number of principal components the data should be reduced too.
        """
        self.components = n_components

    def fit_transform(self, X):
        """
        * Centering our inputs with mean
        * Finding covariance matrix using centered tensor
        * Finding eigen value and eigen vector using torch.eig()
        * Sorting eigen values in descending order and finding index of high eigen values
        * Using sorted index, get the eigen vectors
        * Tranforming the Input vectors with n columns into PCA components with reduced dimension
        :param X: Input tensor with n columns.
        :return: Output tensor with reduced principal components
        """
        centering_X = Tensor(X - ms.ops.mean(X, axis=0))
        print(type(centering_X), centering_X.shape)
        covariance_matrix = ms.ops.mm(centering_X.transpose(), centering_X)/(centering_X.shape[0] - 1)

        eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix.asnumpy())
        eigen_values = Tensor(eigen_values.astype(np.double))
        eigen_vectors = Tensor(eigen_vectors.astype(np.double))

        eigen_sorted_index = ms.ops.argsort(eigen_values, descending=True)

        eigen_vectors_sorted = Tensor(eigen_vectors[:,eigen_sorted_index])
        component_vector = Tensor(eigen_vectors_sorted[:,0:self.components])
        transformed = ms.ops.mm(component_vector.transpose(), centering_X.transpose()).transpose()

        return transformed

if __name__ == '__main__':
    data = load_iris()
    X = Tensor(data.data,dtype=ms.double)
    y = Tensor(data.target)
    pca = pca(n_components=2)
    pca_vector = pca.fit_transform(X)
    pvt = pca_vector.numpy()
    plt.figure(figsize=(6, 6))
    sb.scatterplot(x=pvt[:,0], y=pvt[:,1], hue=y.numpy(), s=60, palette='icefire')
    plt.show()
    exit(0)