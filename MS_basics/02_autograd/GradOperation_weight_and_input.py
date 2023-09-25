'''
如果对网络内部权重求梯度同时也想对输入求梯度，只能显示的设置 get_all=True, 即，

self.grad_op = ops.GradOperation(get_all=True, get_by_list=True)
此时输出的梯度为 元组 ，    tuple（所有输入的梯度， 内部权重梯度）

也就是说这种情况返回的是一个两个元素的元组，元组中第一个元素是所有输入的梯度，第二个是所有内部权重的梯度。
'''
import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore import ParameterTuple, Parameter
from mindspore import dtype as mstype

from mindspore import context, set_seed

context.set_context(device_target="GPU")
set_seed(123)

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.matmul = ops.MatMul()
        self.z = Parameter(Tensor(np.array([1.0, 1.0, 1.0], np.float32)), name='z')

    def construct(self, x, y):
        x = x * self.z
        out = self.matmul(x, y)
        return out


class GradNetWrtX(nn.Cell):
    def __init__(self, net):
        super(GradNetWrtX, self).__init__()
        self.net = net
        self.params = ParameterTuple(net.trainable_params())
        self.grad_op = ops.GradOperation(get_all=True, get_by_list=True)

    def construct(self, x, y):
        gradient_function = self.grad_op(self.net, self.params)
        return gradient_function(x, y)


model = Net()

x = Tensor([[0.8, 0.6, 0.2], [1.8, 1.3, 1.1]], dtype=mstype.float32)
y = Tensor([[0.11, 3.3, 1.1], [1.1, 0.2, 1.4], [1.1, 2.2, 0.3]], dtype=mstype.float32)
output = GradNetWrtX(model)(x, y)
print(len(output))
print('='*30)
print(output[0])
print('='*30)
print(output[1])