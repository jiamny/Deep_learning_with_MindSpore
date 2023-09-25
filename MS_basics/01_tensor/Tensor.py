from mindspore import Tensor
import numpy as np
from mindspore import context, Tensor, nn, set_seed

context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")

'''
Creating a Tensor
'''

# Generating a tensor based on data
x = Tensor(0.1)
print(x)

# Generating a tensor from the NumPy array
try:
    arr = np.array([1, 0, 1, 0])
    tensor_arr = Tensor(arr)
    print(arr)
    print(type(arr))
    print(tensor_arr)
    print(type(tensor_arr))
    print('Try using KeyboardInterrupt')
except KeyboardInterrupt:
    print('KeyboardInterrupt exception is caught')
else:
    print('No exceptions are caught')

# Generating a tensor by using init

from mindspore import Tensor
from mindspore import set_seed
from mindspore import dtype as mstype
from mindspore.common.initializer import One, Normal

set_seed(1)

tensor = Tensor(np.array([[0, 1], [2, 3]]).astype(np.float32))
print(tensor[0])

tensor1 = Tensor(shape=(2, 2), dtype=mstype.float32, init=One())
# Generates an array with values sampled from Normal distribution N(0.01,0.1) in order to initialize a tensor
tensor2 = Tensor(shape=(2, 2), dtype=mstype.float32, init=Normal())

print("tensor1:\n", tensor1)
print("tensor2:\n", tensor2)

'''
The init is used for delayed initialization in parallel mode. Usually, 
it is not recommended to use init interface to initialize parameters.

Inheriting attributes of another tensor to form a new tensor
'''
from mindspore import ops

oneslike = ops.OnesLike()
x = Tensor(np.array([[0, 1], [2, 1]]).astype(np.int32))
output = oneslike(x)

print(output)
print("input shape:", x.shape)
print("output shape:", output.shape)

# Outputting a constant tensor of a specified size
zeros = ops.Zeros()
output = zeros((2, 2), mstype.float32)
print(output)

'''
During Tensor initialization, dtype can be specified to, for example, mstype.int32, mstype.float32 or mstype.bool_.

Tensor Attributes
'''
x = Tensor(np.array([[1, 2], [3, 4]]), mstype.int32)

print("x_shape:", x.shape)
print("x_dtype:", x.dtype)
print("x_transposed:\n", x.T)
print("x_itemsize:", x.itemsize)
print("x_nbytes:", x.nbytes)
print("x_ndim:", x.ndim)
print("x_size:", x.size)
print("x_strides:", x.strides)

'''
Tensor Indexing
'''
tensor = Tensor(np.array([[0, 1], [2, 3]]).astype(np.float32))
print(tensor[0])
print("First row: {}".format(tensor[0]))
print("value of top right corner: {}".format(tensor[1, 1]))
print("Last column: {}".format(tensor[:, -1]))
print("First column: {}".format(tensor[..., 0]))

'''
Tensor Operation

Common arithmetic operations include: addition (+), subtraction (-), multiplication (*), 
division (/), modulo (%), and exact division (//)
'''
x = Tensor(np.array([1, 2, 3]), mstype.float32)
y = Tensor(np.array([4, 5, 6]), mstype.float32)

output_add = x + y
output_sub = x - y
output_mul = x * y
output_div = y / x
output_mod = y % x
output_floordiv = y // x

print("add:", output_add)
print("sub:", output_sub)
print("mul:", output_mul)
print("div:", output_div)
print("mod:", output_mod)
print("floordiv:", output_floordiv)

x1 = np.array([1.0, 2.0, 4.0])
y1 = 3.0

print("pow(x,y):", Tensor(x1.__pow__(y1), dtype=mstype.float32))

# connects a series of tensors in a given dimension.
data1 = Tensor(np.array([[0, 1], [2, 3]]).astype(np.float32))
data2 = Tensor(np.array([[4, 5], [6, 7]]).astype(np.float32))
op = ops.Concat()
output = op((data1, data2))

print(output)
print("shape:\n", output.shape)

# Stack combines two tensors from another dimension.
data1 = Tensor(np.array([[0, 1], [2, 3]]).astype(np.float32))
data2 = Tensor(np.array([[4, 5], [6, 7]]).astype(np.float32))
op = ops.Stack()
output = op([data1, data2])

print(output)
print("shape:\n", output.shape)

'''
Conversion Between Tensor and NumPy

Tensor and NumPy can be converted to each other.
Tensor to NumPy
Use asnumpy() to convert Tensor to NumPy.
'''
zeros = ops.Zeros()

output = zeros((2, 2), mstype.float32)
print("output: {}".format(type(output)))

n_output = output.asnumpy()
print("n_output: {}".format(type(n_output)))

'''
NumPy to Tensor

Use asnumpy() to convert NumPy to Tensor.
'''
output = np.array([1, 0, 1, 0])
print("output: {}".format(type(output)))

t_output = Tensor(output)
print("t_output: {}".format(type(t_output)))

'''
Sparse Tensor

MindSpore now supports the two most commonly used CSR and COO sparse data formats.
'''
import mindspore as ms
from mindspore import Tensor, CSRTensor

indptr = Tensor([0, 1, 2])
indices = Tensor([0, 1])
values = Tensor([1, 2], dtype=ms.float32)
shape = (2, 4)

# CSRTensor construction
csr_tensor = CSRTensor(indptr, indices, values, shape)
print("csr_tensor\n", csr_tensor)
print(csr_tensor.astype(ms.float64).dtype)

# COOTensor
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, COOTensor

indices = Tensor([[0, 1], [1, 2]], dtype=ms.int32)
values = Tensor([1, 2], dtype=ms.float32)
shape = (3, 4)

# COOTensor construction
coo_tensor = COOTensor(indices, values, shape)

print(coo_tensor.values)
print(coo_tensor.indices)
print(coo_tensor.shape)
print(coo_tensor.astype(ms.float64).dtype)  # COOTensor cast to another data type

'''
RowTensor

RowTensor is used to compress tensors that are sparse in the dimension 0. If the dimension of 
RowTensor is [L0, D1, D2, ..., DN ] and the number of non-zero elements in the dimension 0 is D0, then L0 >> D0.
'''
from mindspore import RowTensor
import mindspore.nn as nn


class Net(nn.Cell):
    def __init__(self, dense_shape):
        super(Net, self).__init__()
        self.dense_shape = dense_shape

    def construct(self, indices, values):
        x = RowTensor(indices, values, self.dense_shape)
        return x.values, x.indices, x.dense_shape


indices = Tensor([0])
values = Tensor([[1, 2]], dtype=mstype.float32)
out = Net((3, 2))(indices, values)

print("non-zero values:", out[0])
print("non-zero indices:", out[1])
print("shape:", out[2])

import mindspore.numpy as mnp
u = Tensor([0., 1., 2., 3., 4., 5., 6., 7., 8.], dtype=mstype.float32)
#u = mnp.array([0., 1., 2., 3., 4., 5., 6., 7., 8.]).astype(mnp.float32)
#print(type(u))
#print(u)
#print(u.dtype)
w = mnp.arange(9).astype(mnp.float32)
#print(type(w))
#print(w)
#print(w.dtype)
print('w: ', w.mean())
#print('u: ', ops.lp_norm(u, [0], p = 2))
exit(0)
