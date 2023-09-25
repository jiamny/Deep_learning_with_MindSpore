

import mindspore as ms
import mindspore.ops as ops
import mindspore.context as context
context.set_context(device_target="GPU")

x = ops.arange(12)
print('x: ', x, ' x.shape: ', x.shape, ' x.size: ', x.size)

X = x.reshape(3, 4)
X
ops.zeros((2, 3, 4))
ops.ones((2, 3, 4))
ops.randn(3, 4)
ms.Tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
x = ms.Tensor([1.0, 2, 4, 8])
y = ms.Tensor([2, 2, 2, 2])
print('x: ', x, ' y: ', y)
x + y, x - y, x * y, x / y, x ** y  # **运算符是求幂运算
print('x**y: ', x**y)
print('x: ', x, ' ops.exp(x): ', ops.exp(x))

X = ops.arange(12, dtype=ms.float32).reshape((3,4))
Y = ms.Tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
ops.cat((X, Y), axis=0), ops.cat((X, Y), axis=1)
X == Y
X.sum()
a = ops.arange(3).reshape((3, 1))
b = ops.arange(2).reshape((1, 2))
a, b
a + b
X[-1], X[1:3]
X[1, 2] = 9
X
X[0:2, :] = 12
X
before = id(Y)
Y = Y + X
id(Y) == before


Z = ops.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))


before = id(X)
X[:] = X + Y
print('id(X) == before: ', id(X) == before)


A = X.numpy()
B = ms.Tensor(A)
type(A), type(B)
a = ms.Tensor([3.5])
# mindspore里item()返回的是tensor标量，而不是python标量
print('a: ', a, ' a.item(): ', type(a.item()), ' float(a): ', float(a), ' int(a): ', int(a))

exit(0)