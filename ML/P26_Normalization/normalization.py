import mindspore as ms
from mindspore import context, Tensor, nn

from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

ms.set_seed(42)
context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")

class Normalization:
    def __init__(self, X):
        self.X = X

    def z_score(self):
        mean = ms.ops.mean(self.X, axis=0)
        return self.X.subtract(mean)/ ms.ops.std(self.X, axis=0)

    def min_max(self):
        min = ms.ops.min(self.X, axis=0)
        max = ms.ops.max(self.X, axis=0)
        return self.X.subtract(min[0]) / (max[0] - min[0])

    def log_scaling(self):
        return ms.ops.log(self.X)

    def clipping(self, max, min):
        if self. X > max:
            mask = self. X > max
            self.X = self.X * mask

        if self. X < min:
            mask = self. X < min
            self.X = self.X * mask

        return self.X

if __name__ == '__main__':
    data = load_iris()
    X = Tensor(data.data, dtype=ms.float32)
    y = Tensor(data.target, dtype=ms.int32)   #.unsqueeze(1)
    cls = KNeighborsClassifier()
    normalizer = Normalization(X)
    X_transform = normalizer.z_score()
    cls.fit(X.asnumpy(), y.asnumpy())
    y_pred = cls.predict(X.asnumpy())
    print('Without Normalization',accuracy_score(y.asnumpy(), y_pred))

    cls.fit(X_transform.asnumpy(), y.asnumpy())
    y_pred = cls.predict(X_transform.asnumpy())
    print('Z-Score Normalization' ,accuracy_score(y.asnumpy(), y_pred))
    X_transform = normalizer.min_max()
    cls.fit(X_transform.asnumpy(), y.asnumpy())
    y_pred = cls.predict(X_transform.asnumpy())
    print('Min-Max Normalization' ,accuracy_score(y.asnumpy(), y_pred))
    X_transform = normalizer.log_scaling()
    cls.fit(X_transform.asnumpy(), y.asnumpy())
    y_pred = cls.predict(X_transform.asnumpy())
    print('Log Scaling', accuracy_score(y.asnumpy(), y_pred))

    exit(0)
