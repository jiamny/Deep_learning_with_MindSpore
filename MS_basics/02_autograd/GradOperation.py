'''
对权重求一阶导
对权重求一阶导数其实与前面相比有两个地方要更改：

1.   求导函数要写明对权重求导，即传入参数 get_by_list=True 即，

self.grad_op = ops.GradOperation(get_by_list=True)

2.  具体求导时要传入具体待求导的参数（即，权重）：

self.params = ParameterTuple(net.trainable_params())
gradient_function = self.grad_op(self.net, self.params)

需要知道的一点是如果我们设置了对权重求梯度，则默认不会再对输入求梯度：
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
        self.grad_op = ops.GradOperation(get_by_list=True)

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
