import mindspore as ms
from mindspore import context, Tensor, nn
from mindspore.ops import operations as ops
from mindspore import dtype as mstype

from scipy.stats import mode
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

ms.set_seed(0)
context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

class KNN:
    def __init__(self, k, X):
        """
        :param k: Number of Neighbors
        """
        self.k = k

    def distance(self, point_1, point_2, default='euclidean', p=2):
        if default == 'euclidean':
            return ms.ops.norm(point_1 - point_2, 2, 0)
        elif default == 'manhattan':
            return ms.ops.sum(ops.abs(point_1 - point_2))
        elif default == "minkowski":
            return ms.ops.pow(ops.sum(ops.abs(point_1 - point_2)**p), 1/p)
        else:
            raise ValueError("Unknown similarity distance type")

    def fit_predict(self, X, y, item):
        """
        * Iterate through each datapoints (item/y_test) that needs to be classified
        * Find distance between all train data points and each datapoint (item/y_test)
          using euclidean distance
        * Sort the distance using argsort, it gives indices of the y_test
        * Find the majority label whose distance closest to each datapoint of y_test.
        :param X: Input tensor
        :param y: Ground truth label
        :param item: tensors to be classified
        :return: predicted labels
        """
        y_predict = []
        for i in item:
            point_distances = []
            for ipt in range(X.shape[0]):
                distances = self.distance(X[ipt, :], i)
                point_distances.append(float(distances.item()))

            point_distances = Tensor(point_distances, dtype=mstype.float32)
            k_neighbors = ms.ops.argsort(point_distances)[:self.k]
            y_label = Tensor(y[k_neighbors]).asnumpy()
            major_class = mode(y_label, axis=None, nan_policy='omit')
            major_class = major_class.mode[0]
            y_predict.append(major_class)

        return Tensor(y_predict)

if __name__ == '__main__':
    iris = load_iris()

    X = iris.data
    y = iris.target
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    x_train = ms.Tensor(x_train).astype(mstype.float32)
    y_train = ms.Tensor(y_train)
    x_test = ms.Tensor(x_test).astype(mstype.float32)
    y_test = ms.Tensor(y_test)

    knn = KNN(k=5, X=x_train)
    y_pred = knn.fit_predict(x_train, y_train, x_test)
    print(y_pred)
    print(y_test)
    print(f'Accuracy: {accuracy_score(y_test.asnumpy(), y_pred.asnumpy())}')
