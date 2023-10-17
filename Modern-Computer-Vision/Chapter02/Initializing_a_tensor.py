
import mindspore as ms
from mindspore import Tensor
import mindspore.context as context

ms.set_seed(10)
context.set_context(device_target="GPU")


x = Tensor([[1,2]])
y = Tensor([[1],[2]])
print(x.shape)
# Size([1,2]) # one entity of two items
print(y.shape)
# Size([2,1]) # two entities of one item each
print('x.dtype: ', x.dtype)
# int64
x = Tensor([False, 1, 2.0])
print('Tensor([False, 1, 2.0]): ', x)
# tensor([0., 1., 2.])
print('ms.ops.zeros((3, 4)): \n', ms.ops.zeros((3, 4)))
print('ms.ops.ones((3, 4)): \n', ms.ops.ones((3, 4)))
print('ms.ops.randint(low=0, high=10, size=(3,4)): \n', ms.ops.randint(low=0, high=10, size=(3,4)))
print('ms.ops.rand(3, 4): \n', ms.ops.rand(3, 4))
print('ms.ops.randn((3,4)): \n', ms.ops.randn((3,4)))
print('-'*80)

import numpy as np
x = np.array([[10,20,30],[2,3,4]])
y = Tensor(x)
print(type(x), type(y))

