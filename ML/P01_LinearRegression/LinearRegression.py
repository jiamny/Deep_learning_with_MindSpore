
import mindspore as ms
from mindspore import nn
from mindspore import context

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

class LinearRegression:

    def __init__(self):
        """
        :desc lr: Learning Rate
        :desc iteration: Number of iterations over complete data set
        """

        self.lr = 0.01
        self.iterations = 1

    def y_pred(self, X, w):
        """
        :desc w: weight tensor
        :desc X: input tensor
        """
        #print('=====')
        #print(torch.mm(torch.transpose(w, 0, 1), X))
        return ms.ops.mm(ms.ops.swapaxes(w, 0, 1), X)

    def loss(self, ypred, y):
        """
        :desc c: cost function - to measure the loss between estimated vs ground truth
        """
        #l = 1 / self.m * torch.sum(torch.pow(ypred - y, 2))
        l = 1 / self.m * ms.ops.sum(ms.ops.pow(ypred - y, 2))
        return l

    def gradient_descent(self, w, X, y, ypred):
        """
        :desc dCdW: derivative of cost function
        :desc w_update: change in weight tensor after each iteration
        """
        #print( torch.mm(X, torch.transpose(ypred - y, 0, 1)))
        dCdW = 2 / self.m * ms.ops.mm(X, ms.ops.swapaxes(ypred - y, 0, 1))
        w_update = w - self.lr * dCdW
        return w_update

    def run(self, X, y):
        """
        :type y: tensor object
        :type X: tensor object
        """
        bias = ms.ops.ones((1, X.shape[1]))
        #print('bias ', bias.shape)
        #print(bias)
        X = ms.ops.cat((bias, X), axis=0)
        #print('X ', X.shape)
        self.m = X.shape[1]
        self.n = X.shape[0]
        w = ms.ops.zeros((self.n, 1))

        for iteration in range(1, self.iterations + 1):
            ypred = self.y_pred(X, w)
            cost = self.loss(ypred, y)

            if iteration % 100 == 0:
                print(f'Loss at iteration {iteration} is {cost}')
            w = self.gradient_descent(w, X, y, ypred)
            print('===')
            print(X)
            print(w)
            print(ypred)

        return w


if __name__ == '__main__':
    """
    :desc X: random initialization of input tensor
    :desc y: random initialization of output tensor
    """
    X = ms.ops.rand(1, 10)
    #print(X)
    y = 2 * X + 3 + ms.ops.randn(1, 10) * 0.1

    regression = LinearRegression()
    w = regression.run(X, y)
    print('w: ', w)
