import mindspore as ms
from mindspore import nn
from mindspore import context
from mindspore import dataset
from mindspore.train.callback import LossMonitor
from mindspore.ops import operations as P
from sklearn.datasets import make_blobs
from mindspore.common import set_seed

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

class LogisticRegression:
    def __init__(self, X):
        """
        :param X: Input tensor
        :keyword lr: learning rate
        :keyword epochs: number of times the model iterates over complete dataset
        :keyword weights: parameters learned during training
        :keyword bias: parameter learned during training
        """
        self.lr = 0.1
        self.epochs = 1000
        self.m, self.n = X.shape
        self.weights = ms.ops.zeros((self.n, 1), dtype=ms.float64)
        self.bias = 0

    def sigmoid(self, z):
        """
        :param z: latent variable represents (wx + b)
        :return: squashes the real value between 0 and 1 representing probability score.
        """
        return 1 / (1 + ms.ops.exp(-z))

    def loss(self, yhat, y):
        """
        :param yhat: Estimated y
        :return: Log loss - When y=1, it cancels out half function, remaining half is considered for loss calculation and vice-versa
        """
        #print('y * torch.log(yhat): ', (y * torch.log(yhat)).shape)
        ls = -(1 / self.m) * ms.ops.sum(y * ms.ops.log(yhat) + (1 - y) * ms.ops.log(1 - yhat))
        print('ls ', ls)
        return ls

    def gradient(self, y_predict, y):
        """
        :param y_predict: Estimated y
        :return: gradient is calculated to find how much change is required in parameters to reduce the loss.
        """
        dw = 1 / self.m * ms.ops.mm(X.transpose(), (y_predict - y))
        db = 1 / self.m * ms.ops.sum(y_predict - y)
        return dw, db

    def run(self, X, y):
        """
        :param X: Input tensor
        :param y: Output tensor
        :var y_predict: Predicted tensor
        :var cost: Difference between ground truth and predicted
        :var dw, db: Weight and bias update for weight tensor and bias scalar
        :return: updated weights and bias
        """
        for epoch in range(1, self.epochs + 1):

            y_predict = self.sigmoid(ms.ops.mm(X, self.weights) + self.bias)
            cost = self.loss(y_predict, y)
            dw, db = self.gradient(y_predict, y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            if epoch % 100 == 0:
                print(f"Cost after iteration {epoch}: {cost}")

        return self.weights, self.bias

    def predict(self, X):
        """
        :param X: Input tensor
        :var y_predict_labels: Converts float value to int/bool true(1) or false(0)
        :return: outputs labels as 0 and 1
        """
        y_predict = self.sigmoid(ms.ops.mm(X, self.weights) + self.bias)
        y_predict_labels = y_predict > 0.5

        return y_predict_labels

if __name__ == '__main__':
    """
    :var manual_seed: for reproducing the results
    :desc unsqueeze: adds a dimension to the tensor at specified position.
    """
    set_seed(1)
    X, y, ct = make_blobs(n_samples=1000, centers=2, return_centers=True)
    print(X.shape)
    print(y.shape)
    print(ct)

    X = ms.Tensor(X)
    y = ms.Tensor(y).unsqueeze(1)
    lr = LogisticRegression(X)
    w, b = lr.run(X, y)
    y_predict = lr.predict(X)

    print('torch.sum(y == y_predict) ', ms.ops.sum(y == y_predict))
    print('X.shape[0] ', X.shape[0])
    ac = (ms.ops.sum(y == y_predict)*1.0 / X.shape[0]) *100
    print("Accuracy: ", ac.item())

