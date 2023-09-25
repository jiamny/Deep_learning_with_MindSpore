import mindspore as ms
from mindspore import context, Tensor, nn

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

ms.set_seed(0)
context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

class SVM:
    def __init__(self, X, y, C=1.0):
        self.total_samples, self.features_count = X.shape[0], X.shape[1]
        self.n_classes = len(ms.ops.unique(y)[0])
        self.learning_rate = 0.001
        self.C = C

    def loss(self, X, W, y):
        """
        C parameter tells the SVM optimization how much you want to avoid misclassifying each training
        example. For large values of C, the optimization will choose a smaller-margin hyperplane if that
        hyperplane does a better job of getting all the training points classified correctly. Conversely,
        a very small value of C will cause the optimizer to look for a larger-margin separating hyperplane,
        even if that hyperplane misclassifies more points. For very tiny values of C, you should get
        misclassified examples, often even if your training data is linearly separable.
        :param X:
        :param W:
        :param y:
        :return:
        """
        num_samples = X.shape[0]
        distances = 1 - y * (ms.ops.matmul(X, W.transpose()))

        distances[distances < 0] = 0
        hinge_loss = self.C * (ms.ops.sum(distances) // num_samples)
        cost = 1 / 2 * ms.ops.matmul(W, W.transpose()) + hinge_loss
        return cost

    def gradient_update(self, W, X, y):
        """
        :param W: Weight Matrix
        :param X: Input Tensor
        :param y: Ground truth tensor
        :return: change in weight
        """
        distance = 1 - (y * ms.ops.matmul(X, W.transpose()))
        dw = ms.ops.zeros((1, X.shape[1]), dtype=ms.float64)

        for idx, dist in enumerate(distance):

            if max(0, dist) == 0:
                di = W
            else:
                di = W - (self.C * y[idx] * X[idx])

            dw += di

        dw = dw / len(y)
        return dw

    def fit(self, X, y, max_epochs):
        """
        :param X: Input Tensor
        :param y: Output tensor
        :param max_epochs: Number of epochs the complete dataset is passed through the model
        :return: learned weight of the svm model
        """
        weight = ms.ops.randn((1, X.shape[1]), dtype=ms.float64) * ms.ops.sqrt(Tensor(1./X.shape[1]))

        cost_threshold = 0.0001
        previous_cost = float('inf')
        nth = 0
        for epoch in range(1, max_epochs+1):
            X, y = shuffle(X.asnumpy(), y.asnumpy())
            X = Tensor(X)
            y = Tensor(y)
            for idx, x in enumerate(X):
                x = Tensor(x.asnumpy().copy()).unsqueeze(0)
                weight_update = self.gradient_update(weight, x, y[idx])
                weight = weight - (self.learning_rate * weight_update)

            if epoch % 100 == 0:
                cost = float(self.loss(X, weight, y).squeeze().item())
                print(f'Loss at epoch {epoch}: {cost}')
                if abs(previous_cost - cost) < cost_threshold * previous_cost:
                    return weight
                previous_cost = cost
                nth += 1
        return weight

if __name__ == '__main__':

    num_epochs = 1000
    breast_cancer = load_breast_cancer()
    X = breast_cancer.data
    print('X.sum-1: ', ms.ops.sum(Tensor(X), dim=0))
    X_normalized = MinMaxScaler().fit_transform(X)

    X = Tensor(X_normalized)
    y = Tensor(breast_cancer.target).unsqueeze(1)
    bias = ms.ops.ones((X.shape[0], 1), dtype=ms.float64)
    X = ms.ops.cat((bias, X), axis=1)
    x_train, x_test, y_train, y_test = train_test_split(X.asnumpy(), y.asnumpy(), test_size=0.2)
    x_train = ms.Tensor(x_train).astype(ms.dtype.float64)
    y_train = ms.Tensor(y_train)
    x_test = ms.Tensor(x_test).astype(ms.dtype.float64)
    y_test = ms.Tensor(y_test)

    svm = SVM(x_train, y_train)
    model_weights = svm.fit(x_train, y_train, max_epochs=num_epochs)
    y_pred = ms.ops.sign(ms.ops.matmul(x_test, model_weights.transpose()))
    print(y_pred)
    print(y_test)
    print(f'Accuracy: {accuracy_score(y_test.numpy(), y_pred.numpy())}')
    '''
    import numpy as np
    x = np.random.randn(1, 3)
    xx = ms.Tensor(x, ms.float32)
    y = np.random.randn(1, 3)
    yy = ms.Tensor(y, ms.float32)
    print(ms.ops.matmul(xx, yy.transpose()))
    '''
    exit(0)
