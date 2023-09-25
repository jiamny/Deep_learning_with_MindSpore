import mindspore as ms
import numpy as np
from mindspore import context, Tensor, nn

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

ms.set_seed(42)
context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

class ElasticNetRegression:
    def __init__(self, learning_rate, max_iterations, l1_penality, l2_penality):
        self.lr = learning_rate
        self.max_iterations = max_iterations
        self.l1_penality = l1_penality
        self.l2_penality = l2_penality

    def normalization(self, X):
        """
        :param X: Input tensor
        :return: Normalized input using l2 norm.
        """
        l2 = ms.ops.norm(X, ord=2, dim=-1, dtype=ms.float32)
        l2[l2 == 0] = 1
        return X / l2.unsqueeze(1)

    def fit(self, X, y):
        self.m, self.n = X.shape
        self.w = ms.ops.zeros(self.n, dtype=ms.float32).unsqueeze(1)
        self.b = 0.0
        self.X = X
        self.y = y
        for i in range(self.max_iterations):
            self.update_weights()
            if i % 50 == 0:
                print('Iteration: ', i)

        return self

    def update_weights(self):
        y_pred = self.predict(self.X)
        dw = ms.ops.zeros(self.n).unsqueeze(1)

        for j in range(self.n):

            t1 = Tensor(self.X[:, j].unsqueeze(0))
            t2 = Tensor(self.y - y_pred)
            t3 = Tensor((self.l1_penality + 2 * self.l2_penality * self.w[j]))
            t4 = Tensor(2 * ms.ops.mm(t1, t2).squeeze())
            if self.w[j] > 0:
                dw[j] = ( - (t4 + t3)) / self.m
            else:
                dw[j] = (- (t4 - t3)) / self.m

        db = -2 * ms.ops.sum(self.y - y_pred) / self.m
        self.w = self.w - self.lr * dw
        self.b = self.b - self.lr * db
        return self

    def predict(self, X):
        return ms.ops.mm(X, self.w) + self.b

if __name__ == '__main__':
    #data = load_boston()
    import pandas as pd
    import numpy as np

    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]

    regression = ElasticNetRegression(max_iterations=1000, learning_rate=0.001, l1_penality=500, l2_penality=1)
    X, y = regression.normalization(Tensor(data, dtype=ms.float32)), Tensor(target).unsqueeze(1)
    x_train, x_test, y_train, y_test = train_test_split(X.asnumpy(), y.asnumpy(), test_size=0.3)

    regression.fit(Tensor(x_train, dtype=ms.float32), Tensor(y_train, dtype=ms.float32))
    Y_pred = regression.predict(Tensor(x_test, dtype=ms.float32))
    print("Predicted values: ", ms.ops.round(Y_pred[:3]))
    print("Real values: ", y_test[:3])
    print("Trained W: ", ms.ops.round(regression.w[0]))
    print("Trained b: ", ms.ops.round(regression.b))

    exit(0)

