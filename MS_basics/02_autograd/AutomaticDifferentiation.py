import numpy as np
import mindspore.nn as nn
from mindspore import Parameter, Tensor
from mindspore import context, set_seed

context.set_context(device_target="GPU")
set_seed(123)

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.w = Parameter(np.array([6.0]), name='w')
        self.b = Parameter(np.array([1.0]), name='b')

    def construct(self, x):
        f = self.w * x + self.b
        return f


# Define the derivative class GradNet.
from mindspore import dtype as mstype
import mindspore.ops as ops


class GradNet(nn.Cell):
    def __init__(self, net):
        super(GradNet, self).__init__()
        self.net = net
        self.grad_op = ops.GradOperation()

    def construct(self, x):
        gradient_function = self.grad_op(self.net)
        return gradient_function(x)


# f(x) = 6*x + 1
x = Tensor([100], dtype=mstype.float32)
output = GradNet(Net())(x)
print('----- GradNet -----')
print('6*x + 1: ', output)


# First-order Derivative of the Weight
from mindspore import ParameterTuple


class GradNet(nn.Cell):
    def __init__(self, net):
        super(GradNet, self).__init__()
        self.net = net
        self.params = ParameterTuple(net.trainable_params())
        self.grad_op = ops.GradOperation(get_by_list=True)  # Set the first-order derivative of the weight parameters.

    def construct(self, x):
        gradient_function = self.grad_op(self.net, self.params)
        return gradient_function(x)


# next, derive the function:
# Perform a derivative calculation on the function.
x = Tensor([100], dtype=mstype.float32)
fx = GradNet(Net())(x)

# Print the result.
print('----- First-order Derivative of the Weight -----')
print(f"wgrad: {fx[0]}\nbgrad: {fx[1]}")


class Net2(nn.Cell):
    def __init__(self):
        super(Net2, self).__init__()
        self.w = Parameter(Tensor(np.array([6], np.float32)), name='w')
        self.b = Parameter(Tensor(np.array([1.0], np.float32)), name='b', requires_grad=False)

    def construct(self, x):
        out = x * self.w + self.b
        return out


class GradNet2(nn.Cell):
    def __init__(self, net):
        super(GradNet2, self).__init__()
        self.net = net
        self.params = ParameterTuple(net.trainable_params())
        self.grad_op = ops.GradOperation(get_by_list=True)

    def construct(self, x):
        gradient_function = self.grad_op(self.net, self.params)
        return gradient_function(x)


# Construct a derivative network.
x = Tensor([5], dtype=mstype.float32)
fw = GradNet2(Net2())(x)
print('----- Construct a derivative network -----')
print(fw)


# Gradient Value Scaling
class GradNet3(nn.Cell):
    def __init__(self, net):
        super(GradNet3, self).__init__()
        self.net = net
        # Derivative operation.
        self.grad_op = ops.GradOperation(sens_param=True)
        # Scale an index.
        self.grad_wrt_output = Tensor([0.1], dtype=mstype.float32)

    def construct(self, x):
        gradient_function = self.grad_op(self.net)
        return gradient_function(x, self.grad_wrt_output)


x = Tensor([6], dtype=mstype.float32)
output = GradNet3(Net2())(x)
print('----- Gradient Value Scaling -----')
print(output)

'''
Stopping Gradient Calculation

You can use ops.stop_gradient to stop calculating gradients. The following is an example:
'''
from mindspore.ops import stop_gradient


class Net4(nn.Cell):
    def __init__(self):
        super(Net4, self).__init__()
        self.w = Parameter(Tensor(np.array([6], np.float32)), name='w')
        self.b = Parameter(Tensor(np.array([1.0], np.float32)), name='b')

    def construct(self, x):
        out = x * self.w + self.b
        # Stop updating the gradient. The out does not contribute to gradient calculations.
        out = stop_gradient(out)
        return out


class GradNet4(nn.Cell):
    def __init__(self, net):
        super(GradNet4, self).__init__()
        self.net = net
        self.params = ParameterTuple(net.trainable_params())
        self.grad_op = ops.GradOperation(get_by_list=True)

    def construct(self, x):
        gradient_function = self.grad_op(self.net, self.params)
        return gradient_function(x)


x = Tensor([100], dtype=mstype.float32)
output = GradNet4(Net4())(x)
print('----- Stopping Gradient Calculation -----')
print(f"wgrad: {output[0]}\nbgrad: {output[1]}")

exit(0)
