
import mindspore as ms
from mindspore import Tensor
import mindspore.context as context

ms.set_seed(10)
context.set_context(device_target="GPU")

x = Tensor([[1,2,3,4], [5,6,7,8]])
print(x * 10)
x = Tensor([[1,2,3,4], [5,6,7,8]])
y = x.add(10)
print('y:\n', y)

y = Tensor([2, 3, 1, 0])        # y.shape == (4)
y = y.view(4,1)                 # y.shape == (4, 1)
print('y.view(4,1):\n', y.view(4,1))
x = ms.ops.randn(10,1,10)
print('x:\n', x)
z1 = ms.ops.squeeze(x, 1)       # similar to np.squeeze()
print('squeeze(x, 1):\n', z1)

# The same operation can be directly performed on
# x by calling squeeze and the dimension to squeeze out
z2 = x.squeeze(1)
print('x.squeeze(1):\n', z2)
assert ms.ops.all(z1 == z2) # all the elements in both tensors are equal
print('Squeeze:\n', x.shape, z1.shape)
x = ms.ops.randn(10,10)
print('x.shape: ', x.shape)
# torch.size(10,10)
z1 = x.unsqueeze(0)
print('z1.shape: ', z1.shape)

# size(1,10,10)
# The same can be achieved using [None] indexing
# Adding None will auto create a fake dim at the
# specified axis
print('using [None] indexing')
x = ms.ops.randn(10,10)
z2, z3, z4 = x[None], x[:,None], x[:,:,None]
print('z2.shape: ', z2.shape, 'z3.shape: ', z3.shape, 'z4.shape: ', z4.shape)

x = Tensor([[1,2,3,4], [5,6,7,8]], dtype=ms.float32)
y = Tensor(y, dtype=ms.float32)
print(y.shape)
print('matmul(x, y): \n', ms.ops.matmul(x, y))
print('x@y: \n', x@y)

# Cat axis 0:  torch.Size([10, 10, 10]) Size([20, 10, 10])
print('Cat axis 0:  torch.Size([10, 10, 10]) Size([20, 10, 10])')
x = ms.ops.randn(10,10,10)
z = ms.ops.cat([x,x], axis=0) # np.concatenate()
print('Cat axis 0:', x.shape, z.shape)

# Cat axis 1: torch.Size([10, 10, 10]) Size([10, 20, 10])
print('Cat axis 1: torch.Size([10, 10, 10]) Size([10, 20, 10])')
z = ms.ops.cat([x,x], axis=1) # np.concatenate()
print('Cat axis 1:', x.shape, z.shape)


x = ms.ops.arange(25).reshape(5,5)
print('x:\n', x)
print('Max:', x.shape, x.max()) 
print(x.max(axis=0))
val, index =x.max(axis=1, return_indices=True)
print('Max in axis 1:\n', val, index)
# output, index
m, argm = ms.ops.max(x, axis=1)
print('Max in axis 1:\n', m, argm)
x = ms.ops.randn(10,20,30)
z = x.permute(2,0,1) # np.permute()
print('Permute dimensions:', x.shape, z.shape)
dir(ms.Tensor)
help(ms.Tensor.view)

