"""
Reference: https://github.com/eriklindernoren/ML-From-Scratch
This github repository implements high quality code as we see in official libraries like sklearn etc.
Great reference to kickstart your journey for ML programming.
"""
import mindspore as ms
from mindspore import context, Tensor, nn

from sklearn.datasets import load_diabetes, load_breast_cancer
from itertools import combinations_with_replacement
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ms.set_seed(0)
context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

class LassoRegularization:
    def __init__(self, alpha):
        """
        :param alpha:
        * When 0, the lasso regression turns into Linear Regression
        * When increases towards infinity, it turns features coefficients into zero.
        * Try out different value to find out optimized values.
        """
        self.alpha = alpha

    def __call__(self, w):
        """
        :param w: Weight vector
        :return: Penalization value for MSE
        """
        return self.alpha * ms.ops.norm(w, ord=1)

    def grad(self, w):
        """
        :param w: weight vector
        :return: weight update based on sign value, it helps in removing coefficients from W vector
        torch.sign:
        a
        tensor([ 0.7000, -1.2000,  0.0000,  2.3000])
        torch.sign(a)
        tensor([ 1., -1.,  0.,  1.])
        """
        return self.alpha * ms.ops.sign(w)

class RidgeRegularization:
    def __init__(self, alpha):
        """
        :param alpha:
        * When 0, the lasso regression turns into Linear Regression
        * When increases towards infinity, it turns features coefficients into zero.
        * Try out different value to find out optimized values.
        """
        self.alpha = alpha

    def __call__(self, w):
        """
        :param w: Weight vector
        :return: Penalization value for MSE
        """
        return self.alpha * 0.5 * ms.ops.mm(w.transpose(), w)

    def grad(self, w):
        """
        :param w: weight vector
        :return: weight update based on sign value, it helps in reducing the coefficient values from W vector
        """
        return self.alpha * w

class Regression:
    def __init__(self, learning_rate, epochs, regression_type='lasso'):
        """
        :param learning_rate: constant step while updating weight
        :param epochs: Number of epochs the data is passed through the model
        Initalizing regularizer for Lasso Regression.
        """
        self.lr = learning_rate
        self.epochs = epochs
        if regression_type == 'lasso':
            self.regularization = LassoRegularization(alpha=1.0)
        else:
            self.regularization = RidgeRegularization(alpha=2.0)

    def normalization(self, X):
        """
        :param X: Input tensor
        :return: Normalized input using l2 norm.
        """
        l2 = ms.ops.norm(X, ord=2, dim=1)
        l2[l2 == 0] = 1
        print('l2 ', l2.shape)
        print('l2.unsqueeze(1) ', l2.unsqueeze(1).shape)
        print('l2.reshape() ', l2.reshape((l2.shape[0], 1)).shape)
        return X / l2.unsqueeze(1)

    def polynomial_features(self, X, degree):
        """
        It creates polynomial features from existing set of features. For instance,
        X_1, X_2, X_3 are available features, then polynomial features takes combinations of
        these features to create new feature by doing X_1*X_2, X_1*X_3, X_2*X3.

        combinations output: [(), (0,), (1,), (2,), (3,), (0, 0), (0, 1), (0, 2), (0, 3),
        (1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]
        :param X: Input tensor (For Iris Dataset, (150, 4))
        :param degree: Polynomial degree of 2, i.e we'll have product of two feature vector at max.
        :return: Output tensor (After adding polynomial features, the number of features increases to 15)
        """
        n_samples, n_features = X.shape[0], X.shape[1]
        #print('n_features', n_features, ' degree ', degree)
        def index_combination():
            combinations = [combinations_with_replacement(range(n_features), i) for i in range(0, degree+1)]
            flat_combinations = [item for sublists in combinations for item in sublists]
            return flat_combinations

        combinations = index_combination()
        print('combinations: ', combinations)
        n_output_features = len(combinations)
        X_new = ms.numpy.empty((n_samples, n_output_features))

        for i, index_combs in enumerate(combinations):
            if len(index_combs) == 0:
                X_new[:, i] = 1
            else:
                X_new[:, i] = ms.ops.prod(X[:, index_combs], axis=1)

        X_new = X_new.astype(ms.float32)
        return X_new

    def weight_initialization(self, n_features):
        """
        :param n_features: Number of features in the data
        :return: creating weight vector using uniform distribution.
        """
        print('n_features', n_features,
              ' torch.scalar_tensor(n_features) ', ms.Tensor(n_features),
              ' shape ', ms.Tensor(n_features).shape,
              ' sqrt ', ms.ops.sqrt(ms.Tensor(n_features)))
        limit = 1.0 / ms.ops.sqrt(ms.Tensor(n_features))
        print(' limit ', limit)

        u1 = ms.nn.probability.distribution.Uniform(-limit, limit)
        self.w = u1.sample((n_features, 1))
        print(self.w.shape)
        print(self.w)
        self.w = self.w.astype(ms.float32)

    def fit(self, X, y):
        """
        :param X: Input tensor
        :param y: ground truth tensor
        :return: updated weight vector for prediction
        """
        self.training_error = {}
        self.weight_initialization(n_features=X.shape[1])
        for epoch in range(1, self.epochs+1):
            y_pred = ms.ops.mm(X, self.w)
            mse = ms.ops.mean(0.5 * (y - y_pred)**2 + self.regularization(self.w))
            self.training_error[epoch] = float(mse.item())
            grad_w = ms.ops.mm(-(y - y_pred).transpose(), X).transpose() + self.regularization.grad(self.w)
            self.w -= self.lr * grad_w


    def predict(self, X):
        """
        :param X: input tensor
        :return: predicted output using learned weight vector
        """
        y_pred = ms.ops.mm(X, self.w)
        return y_pred

if __name__ == '__main__':

    diabetes =  load_diabetes() # load_breast_cancer()

    print(diabetes.data)
    print(diabetes.target)
    X = diabetes.data
    y = diabetes.target
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    x_train = ms.Tensor(x_train, dtype=ms.float32)
    x_test = ms.Tensor(x_test, dtype=ms.float32)
    y_train = ms.Tensor(y_train, dtype=ms.float32).unsqueeze(dim=1)
    y_test = ms.Tensor(y_test, dtype=ms.float32).unsqueeze(dim=1)
    print('y_train.shape: ', y_train.shape)

    regression = Regression(learning_rate=0.0001, epochs=3000, regression_type='lasso')
    regression.fit(regression.normalization(regression.polynomial_features(x_train, degree=1)), y_train)
    y_pred = regression.predict(regression.normalization(regression.polynomial_features(x_test, degree=1)))

    plt.figure(figsize=(6, 6))
    sb.scatterplot(x=list(regression.training_error.keys()), y=list(regression.training_error.values()))
    plt.show()

    '''
    n_features = 5
    degree = 3
    def index_combination():
        combinations = [combinations_with_replacement(range(n_features), i) for i in range(0, degree + 1)]
        flat_combinations = [item for sublists in combinations for item in sublists]
        return flat_combinations


    combinations = index_combination()
    print(combinations)
    n_output_features = len(combinations)
    '''
