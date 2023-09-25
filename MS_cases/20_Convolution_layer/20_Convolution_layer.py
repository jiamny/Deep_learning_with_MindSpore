
# 基于MindSpore构造卷积层

from mindspore import Tensor
import mindspore.nn as nn
import numpy as np
from mindspore import dtype as mstype
from mindspore import context, set_seed
set_seed(0)
context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")

# 给出Tensor的shape为(1,120,1024,640)计算二维卷积。
net = nn.Conv2d(120, 240, 4, has_bias=False, weight_init='normal')
# 输入
x = Tensor(np.ones([1, 120, 1024, 640]), mstype.float32)

## 模型构建


# 导入科学计算库
import numpy as np
# 导入神经网络模块
import mindspore.nn as nn
# 被初始化的Tensor的数据类型
from mindspore import dtype as mstype
# 常见操作
from mindspore.nn import Conv2d
# 用于初始化Tensor的Tensor
from mindspore import Tensor
# MindSpore中神经网络的基本构成单元
from mindspore.nn import Cell

## 模型预测


net = Conv2d(120, 240, 3, has_bias=False, weight_init='normal')
# 输入
x = Tensor(np.ones([1, 120, 1024, 640]), mstype.float32)
# 输出
output = net(x)
print(output[:,6])
print(output.shape)

## 使用numpy实现二维卷积


# 导入科学计算库
import numpy as np


class my_Conv2d:
    # 初始化
    def __init__(self, 
                 input_channel,    # 输入Tensor的空间维度
                 output_channel,   # 输出Tensor的空间维度
                 kernel_size,      # 指定二维卷积核的高度和宽度
                 stride=1,         # 二维卷积核的移动步长
                 padding=1,        # 输入的高度和宽度方向上填充的数量
                 bias=False,       # 二维卷积层是否添加偏置参数，默认值：False
                 dilation=1):      # 二维卷积核膨胀尺寸
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.stride = stride
        self.padding = padding
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.weight = np.random.randn(output_channel, input_channel, self.kernel_size[0], self.kernel_size[1])
        self.bias = None
        if bias:
            self.bias = np.random.randn(output_channel)

    def __call__(self, inputs):
        return self.infer(inputs)

    def infer(self, inputs):
        # 根据参数，算出输出的shape
        # print(inputs.shape)
        batch_size, input_channel, height, width = inputs.shape
        output_h = (height + 2 * self.padding - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride + 1
        output_w = (width + 2 * self.padding - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride + 1
        outputs = np.zeros([batch_size, self.output_channel, output_h, output_w])
        
        # 计算padding之后的inputs_array
        inputs_padding = np.zeros([batch_size, input_channel, height + 2 * self.padding, width + 2 * self.padding])
        inputs_padding[:, :, self.padding: self.padding + height, self.padding:self.padding + width] = inputs

        # 如果有dilation，根据dilation之后的shape往kernel中插入0（注意，原self.weight不变）
        dilation_shape = self.dilation[0] * (self.kernel_size[0] - 1) + 1, self.dilation[1] * (self.kernel_size[1] - 1) + 1
        kernel = np.zeros((self.output_channel, input_channel, dilation_shape[0], dilation_shape[1]))
        # print(kernel)
        if self.dilation[0] > 1:
            for i in range(self.kernel_size[0]):
                for j in range(self.kernel_size[1]):
                    kernel[:, :, self.dilation[0] * i, self.dilation[1] * j] = self.weight[:, :, i, j]
        else:
            kernel = self.weight
        # print(output_h,output_w)
        
        # 开始前向计算
        for h in range(output_h):
            for w in range(output_w):
                input_ = inputs_padding[
                         :,
                         :,
                         h * self.stride:h * self.stride + dilation_shape[0],
                         w * self.stride:w * self.stride + dilation_shape[1]
                         ]
                # input_ shape : batch_size, output_channel, input_channel, dilation_shape[0], dilation_shape[1]
                input_ = np.repeat(input_[:, np.newaxis, :, :, :], self.output_channel, axis=1)
                # print(input_)
                # kernel_ shape: batch_size, output_channel, input_channel, dilation_shape[0], dilation_shape[1]
                kernel_ = np.repeat(kernel[np.newaxis, :, :, :, :], batch_size, axis=0)
                # print(kernel_.shape, input_.shape)
                # output shape: batch_size, output_channel
                output = input_ * kernel_
                output = np.sum(output, axis=(-1, -2, -3))
                outputs[:, :, h, w] = output
                # print(h,w)

        if self.bias is not None:
            bias_ = np.tile(self.bias.reshape(-1, 1), (1, output_h * output_w)).\
                reshape(self.output_channel, output_h, output_w)
            outputs += bias_
        return outputs

## 实验对比

import time

# record start time
start = time.time()

net = Conv2d(120, 240, 3, has_bias=False, weight_init='normal')
# mindspore
input_1 = Tensor(np.ones([1, 120, 1024, 640]), mstype.float32)
output_1 = net(input_1)
print('mindspore result shape:\n',output_1.shape)
print('mindspore result:\n',output_1[:,6])

# print the difference between start and end time in milli. secs
print("The time of execution of above mindspore net is: {:8.1f} ms".format((time.time() -start) * 10**3))

# numpy
# record start time
start = time.time()

input_2 = np.ones([1, 120, 1024, 640])
my_net = my_Conv2d(120, 240, 3, stride=1, dilation=1)
my_net.weight = net.weight.numpy()
output_2 = my_net(input_2)
print('numpy result shape:\n',output_2.shape)
print('numpy result:\n',output_2[:,6])
print(output.shape)

# print the difference between start and end time in milli. secs
print("The time of execution of above numpy net is: {:8.1f} ms".format((time.time() -start) * 10**3))
exit(0)
