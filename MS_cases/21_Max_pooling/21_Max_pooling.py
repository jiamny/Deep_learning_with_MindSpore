
#  基于MindSpore构建Max Pooling层

#导入mindspore
import mindspore
#导入numpy库
import numpy as np
#导入mindspore中的nn模块
import mindspore.nn as nn
#从MindSpore中导入Tensor库
from mindspore import Tensor
from mindspore import context, set_seed
set_seed(0)
context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

#模型构建部分加载数据
x= Tensor(np.random.randint(0, 10, [1, 2, 4]), mindspore.float32)
#模型测试部分加载数据
y= Tensor(np.random.randint(0, 10, [1, 2, 4, 4]), mindspore.float32)
print("x:")
print(x)
print("y:")
print(y)

## 模型构建

#定义池化层大小与步长
max_pool = nn.MaxPool1d(kernel_size=3, stride=1)
#输入数据维度，并用Tensor将其转化为32浮点数
x= Tensor(np.random.randint(0, 10, [1, 2, 4]), mindspore.float32)
#经过池化后，输出数据维度
output = max_pool(x)
result = output.shape
print(result)

## 模型测试

#参考答案
#设置池化层kernel_size=3, 步长stride=1
pool = nn.MaxPool2d(kernel_size=3, stride=1)
#输入数据维度，并用Tensor将其转化为32浮点数
y= Tensor(np.random.randint(0, 10, [1, 2, 4, 4]), mindspore.float32)
#经过池化后，输出数据维度
output = pool(y)
print(output.shape)
exit(0)