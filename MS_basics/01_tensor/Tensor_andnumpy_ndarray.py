import mindspore as ms
from mindspore.common.tensor import Tensor
from mindspore import ops
from mindspore import dtype as mstype
import numpy as np

from mindspore import context
context.set_context(device_target="CPU")

'''
ms.Tensor 是用小括号表示的，而不是中括号。

1. 数据类型转换
ms.Tensor 默认整数类型是 int64 , 默认浮点数类型是 float64
'''
a = ms.Tensor([1])
print('a.type: ', a.dtype, type(a))
b = a.astype(dtype=mstype.float32)
c = Tensor(a, dtype=mstype.float64)
print('c.type: ', c.dtype)
print('device: ', ms.get_context('device_target'))

'''
2、基本用法

t.shape: 查看形状
t.size: 查看元素总数（这个需要特别注意，t.size查看的不是Tensor的形状）
t.ndim: 查看维数
t.numel(): 查看元素总数
type(t): 查看数据结构类型
t.dtype: 查看元素数据类型

tensor ( [ [ 1, 2, 3, 4, 5 ],[10, 20, 30, 40, 50 ] ] ) 的尺寸是 2×5，而不是 1×2×5 。最外面的小括号不算在维数内，只是用于将Tensor包起来。
'''
t = ms.Tensor([[1,2,3,4,5],[6,7,8,9,10]])
print('t.shape: ', t.shape)
print('t.size: ', t.size)
print('t.ndim: ', t.ndim)
print('type(t): ', type(t))
print('t.dtype: ', t.dtype)


'''
mindspore.Tensor与numpy的转换

t.asnumpy(): 将ms.Tensor转换为numpy
mindspore.common.tensor.Tensor(np): 将numpy转换为ms.Tensor

转换之后，Tensor与numpy独立，不共享内存，对其中一个修改，另一个不会随之改变。

所以下面展示下MindSpore中张量和Numpy类型的互相转换。
'''

'''
张量转换为NumPy
'''
zeros = ops.Zeros()

output = zeros((2,2), mstype.float32)

print("output: {}".format(type(output)))

n_output = output.asnumpy()

print("n_output: {}".format(type(n_output)))

'''
NumPy转换为张量
'''
output = np.array([1, 0, 1, 0])
print(output)
print("output: {}".format(type(output)))

t_output = Tensor(output)
print(t_output)
print("t_output: {}".format(type(t_output)))

exit(0)
