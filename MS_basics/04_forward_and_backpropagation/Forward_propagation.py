import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore import ParameterTuple, Parameter
from mindspore import dtype as mstype

from mindspore import context, set_seed

context.set_context(device_target="GPU")
set_seed(123)

'''
MindSpore计算一阶导数方法  mindspore.ops.GradOperation (get_all=False, get_by_list=False, sens_param=False)，
其中get_all为False时，只会对第一个输入求导，为True时，会对所有输入求导；

get_by_list为False时，不会对权重求导，为True时，会对权重求导；

sens_param对网络的输出值做缩放以改变最终梯度。

get_all : 决定着是否根据输出对输入进行求导。

get_by_list : 决定着是否对神经网络内的参数权重求导。

sens_param : 对网络的输出进行乘积运算后再求导。（通过对网络的输出值进行缩放后再进行求导）
'''

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.matmul = ops.MatMul()
        self.z = Parameter(Tensor(np.array([1.0, 1.0, 1.0], np.float32)), name='z')

    def construct(self, x, y):
        x = x * self.z
        out = self.matmul(x, y)
        return out


model = Net()

for m in model.parameters_and_names():
    print(m)

x = Tensor([[0.8, 0.6, 0.2], [1.8, 1.3, 1.1]], dtype=mstype.float32)
y = Tensor([[0.11, 3.3, 1.1], [1.1, 0.2, 1.4], [1.1, 2.2, 0.3]], dtype=mstype.float32)
result = model(x, y)
print(result)

n_x = np.array([[0.8, 0.6, 0.2], [1.8, 1.3, 1.1]], dtype=np.float32)
n_y = np.array([[0.11, 3.3, 1.1], [1.1, 0.2, 1.4], [1.1, 2.2, 0.3]], dtype=np.float32)
result = model(x, y)
print(result)