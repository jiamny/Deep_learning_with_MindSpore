import mindspore as ms
from mindspore import context, Tensor, nn

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from MLP import MultiLayerPerceptron, CrossEntropy, normalization, accuracy_score, to_categorical

ms.set_seed(42)
context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")


class Sigmoid:
    def __call__(self, X):
        return 1 / (1 + ms.ops.exp(-X))

    def gradient(self, X):
        return self.__call__(X) * (1 - self.__call__(X))

class Softmax:
    def __call__(self, X):
        e_x = ms.ops.exp(X - ms.ops.max(X, axis=-1, keepdims=True)[0])
        return e_x / ms.ops.sum(e_x, dim=1, keepdim=True)

    def gradient(self, X):
        p = self.__call__(X)
        return p * (1 - p)

class TanH:
    def __call__(self, X):
        return 2 / (1 + ms.ops.exp(-2 * X)) - 1

    def gradient(self,X):
        return 1 - ms.ops.pow(self.__call__(X), 2)

class Relu:
    def __call__(self, X):
        return ms.ops.where(X>0.0, X, 0.0)

    def gradient(self, X):
        return ms.ops.where(X >=0.0, 1.0, 0.0)

class LeakyRelu:
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, X):
        return ms.ops.where(X > 0.0, X, self.alpha * X)

    def gradient(self, X):
        return ms.ops.where(X > 0.0, 1.0, self.alpha)

class ELU:
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, X):
        return ms.ops.where(X>=0.0, X, self.alpha * (ms.ops.exp(X) - 1))

    def gradient(self, X):
        return ms.ops.where(X >= 0.0, 1.0, self.__call__(X) + self.alpha)

class SELU():
    def __init__(self):
        self.alpha = 1.6732632423543772848170429916717
        self.scale = 1.0507009873554804934193349852946

    def __call__(self, x):
        return self.scale * ms.ops.where(x >= 0.0, x, self.alpha*(ms.ops.exp(x)-1))

    def gradient(self, x):
        return self.scale * ms.ops.where(x >= 0.0, 1.0, self.alpha * ms.ops.exp(x))

class SoftPlus():
    def __call__(self, x):
        return ms.ops.log(1 + ms.ops.exp(x))

    def gradient(self, x):
        return 1 / (1 + ms.ops.exp(-x))

if __name__ == '__main__':
    data = load_digits()
    X = normalization(Tensor(data.data, dtype=ms.float32))
    y = Tensor(data.target, dtype=ms.int32)

    # Convert the nominal y values to binary
    y = to_categorical(y)

    X_train, X_test, y_train, y_test = train_test_split(X.asnumpy(), y.asnumpy(), test_size=0.4, random_state=1)

    X_train = Tensor(X_train, dtype=ms.float32)
    X_test = Tensor(X_test, dtype=ms.float32)
    y_train = Tensor(y_train, dtype=ms.int32)
    y_test = Tensor(y_test, dtype=ms.int32)

    # MLP
    clf = MultiLayerPerceptron(n_hidden=16,
                               n_iterations=1000,
                               learning_rate=0.01, activation_function_hidden_layer=Sigmoid(),
                               activation_function_output_layer=Softmax())

    clf.fit(X_train, y_train)
    y_pred = ms.ops.argmax(clf.predict(X_test), dim=1)
    y_test = ms.ops.argmax(y_test, dim=1)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    exit(0)
