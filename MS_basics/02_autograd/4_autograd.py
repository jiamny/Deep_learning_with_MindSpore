import mindspore as ms
import mindspore.ops as ops
import mindspore.context as context
context.set_context(device_target="GPU")

x = ops.arange(4.0)
x



import mindspore.numpy as mnp
from mindspore import grad

def forward(x):
    return 2 * mnp.dot(x, x)

y = forward(x)
y


x_grad = grad(forward)(x)
x_grad
x_grad == 4 * x


def forward(x):
    return x.sum()

x_grad = grad(forward)(x)
x_grad
def forward(x):
    y = x * x
    return y.sum()

x_grad = grad(forward)(x)
x_grad
def forward(x):
    y = x * x
    u = ops.stop_gradient(y)
    z = u * x
    return z, u

z, u = forward(x)
x_grad = grad(forward)(x)
x_grad == u
def forward(x):
    y = x * x
    return y.sum()
x_grad = grad(forward)(x)
x_grad == 2 * x
def f(a):
    b = a * 2
    while ops.norm(b, dim=0) < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
a = ops.randn(())
d = f(a)
a_grad = grad(f)(a)
a_grad == d / a
print('a_grad: ', a_grad)
