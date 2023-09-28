import mindspore as ms
from mindspore import context, Tensor, nn

from sklearn.datasets import load_boston
ms.set_seed(42)
context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")

class GradientDescent:
    def __init__(self, learning_rate=0.01, max_iterations=100):
        self.lr = learning_rate
        self.max_iterations = max_iterations

    def normalization(self, X):
        """
        :param X: Input tensor
        :return: Normalized input using l2 norm.
        """
        l2 = ms.ops.norm(X, ord=2, dim=-1)
        l2[l2 == 0] = 1
        return X / l2.unsqueeze(1)

    def compute_error(self, b, m, X, y):
        total_error = 0.0
        print('m: ', m.shape, ' X.transpose(): ', X.transpose().shape, ' y: ', y.shape)
        for i in range(0, X.shape[0]):
            total_error += (y - (ms.ops.mm(m , X.transpose())) + b) ** 2

        print('total_error: ', total_error.shape, '\n', total_error)
        return total_error / float(X.shape[0])

    def step(self, b_curr, m_curr, X, y, learning_rate):
        b_gradient = 0
        m_gradient = 0

        N = float(X.shape[0])
        for i in range(X.shape[0]):
            b_gradient += -(2/N) * ms.ops.sum(y - (ms.ops.mm(X, m_curr.transpose()) + b_curr), dim=0)
            m_gradient += -(2/N) * ms.ops.sum(ms.ops.mm(X.transpose(),
                                                        (y - (ms.ops.mm(X, m_curr.transpose()) + b_curr))), dim=0)

        new_b = b_curr - (learning_rate * b_gradient)
        new_m = m_curr - (learning_rate * m_gradient)
        return [new_b, new_m]

    def gradient_descent(self, X, y, start_b, start_m):
        b = start_b
        m = start_m
        for i in range(self.max_iterations):
            b, m = self.step(b_curr=b, m_curr=m, X=X, y=y, learning_rate=self.lr)

        return b, m

if __name__ == '__main__':

    import pandas as pd
    import numpy as np

    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    X = Tensor(np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]]), dtype=ms.float32)
    print('X: ', X.shape)
    y = Tensor(raw_df.values[1::2, 2], dtype=ms.float32).unsqueeze(1)

    initial_b = 0.0
    initial_m = ms.ops.normal((X.shape[1], 1), mean=0.0, stddev=1.0, seed=5).astype(dtype=ms.float32).transpose()
    print(initial_m.shape)

    gd = GradientDescent(learning_rate=0.0001,max_iterations=100)
    gd.compute_error(X=gd.normalization(X), y=y, b=initial_b, m=initial_m)

    bias, slope = gd.gradient_descent(gd.normalization(X), y, start_b=initial_b, start_m=initial_m)
    X = gd.normalization(X)
    print('y: ', y[0].item())
    print('y_pred: ', (ms.ops.mm(slope, X[0].unsqueeze(0).transpose())+bias)[0].item())

    exit(0)

