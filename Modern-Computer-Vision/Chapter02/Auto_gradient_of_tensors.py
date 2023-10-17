import mindspore.nn as nn
import mindspore.ops as ops
import mindspore as ms
from mindspore import Tensor
import mindspore.context as context
from mindspore import grad, get_grad
from mindspore import ParameterTuple, Parameter

context.set_context(device_target="GPU")

def forward(x):
    return ms.ops.pow(x, 2).sum()


x = Tensor([[2., -1.], [1., 1.]], dtype=ms.float32)
print('x: ', x)
x_grad = grad(forward)(x)
print('x_grad: ', x_grad)

# Computing gradients for the same case that was present in `Chain_rule.ipynb` notebook in previous chapter

x = Tensor([[1,1]], dtype=ms.float32)
y = Tensor([[0]], dtype=ms.float32)


class Net(nn.Cell):
    def __init__(self, W):
        super(Net, self).__init__()
        self.matmul = ops.MatMul()
        self.W0 = Parameter(W[0], name='W0')
        self.W1 = Parameter(W[1], name='W1')
        self.W2 = Parameter(W[2], name='W2')
        self.W3 = Parameter(W[3], name='W3')

    def construct(self, inputs, outputs):
        pre_hidden = self.matmul(inputs, self.W0) + self.W1
        hidden = 1/(1+ms.ops.exp(-pre_hidden))
        out = self.matmul(hidden, self.W2) + self.W3
        mean_squared_error = ms.ops.mean(ms.ops.square(out - outputs))
        return mean_squared_error


class GradNetWrtX(nn.Cell):
    def __init__(self, net):
        super(GradNetWrtX, self).__init__()
        self.net = net
        self.params = ParameterTuple(net.trainable_params())
        self.grad_op = ops.GradOperation(get_all=True, get_by_list=True)

    def construct(self, x, y):
        gradient_function = self.grad_op(self.net, self.params)
        return gradient_function(x, y)
W = [
    Tensor([[-0.0053, 0.3793],
              [-0.5820, -0.5204],
              [-0.2723, 0.1896]], dtype=ms.float32).transpose(),
    Tensor([-0.0140, 0.5607, -0.0628], dtype=ms.float32),
    Tensor([[ 0.1528, -0.1745, -0.1135]], dtype=ms.float32).transpose(),
    Tensor([-0.5516], dtype=ms.float32)
]

model = Net(W)
loss = model(x, y)
print('loss: ', loss)
output = GradNetWrtX(model)(x, y)
print(len(output))

print('='*30)
print('W grad:')
for wg in output[1]:
    print(wg)
print('='*30)
print('updated_W:')
for i in range(len(W)):
    print(W[i] - output[1][i])
