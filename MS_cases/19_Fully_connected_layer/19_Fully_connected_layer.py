
# 基于Mindspore构造全连接层

from mindspore import Tensor                # 张量
import numpy as np                          # 科学计算库
from mindspore import dtype as mstype       # 数据类型模块

from mindspore import context, set_seed
set_seed(0)
context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

# 构建一个2*3的矩阵并转换为Tensor
x = Tensor(np.array([[180, 234, 154], [244, 48, 247]]), mstype.float32)

# 模型构建

import mindspore.nn as nn                              # 神经网络模块                            
from mindspore import Parameter                        # 封装权重模块，初始化后的Parameter是Tensor的子类
from mindspore import ops                              # 常见算子操作
from mindspore.common.initializer import initializer   # 初始化神经元参数
from mindspore.nn import get_activation                #获取激活函数模块


class Dense(nn.Cell):
    # 初始化全连接层
    def __init__(self,
                 in_channels,           #输入Tensor的空间维度
                 out_channels,          #输出Tensor的空间维度
                 weight_init='normal',  #权重参数的初始化方法，采用normal
                 bias_init='zeros',     #偏置参数的初始化方法，采用zeros
                 has_bias=True,         #是否使用偏置向量 bias，默认为True
                 activation=None        #应用于全连接层输出的激活函数，采用None
                 ):
        #调用父类初始化函数完成初始化        
        super(Dense, self).__init__()

        self.reshape = ops.Reshape()  #使用ops模块的Reshape，Reshape基于给定的shape，对输入Tensor进行重新排列
        self.shape_op = ops.Shape()   #使用ops模块的Shape，Shape返回输入Tensor的shape
        self.matmul = ops.MatMul(transpose_b=True)    #使用ops模块的MatMul
        self.weight = Parameter(initializer(weight_init, [out_channels, in_channels]), name="weight")   #初始化权重
        self.bias = None    #bias初始为空
        self.has_bias = has_bias
        if self.has_bias:
            self.bias = Parameter(initializer(bias_init, [out_channels]), name="bias")
            self.bias_add =ops.BiasAdd()     #使用ops模块的BiasAdd，将来用于加上偏置

        #定义激活函数 如果activation是字符类型则获取相应激活函数
        #否则意味activation对象是一个函数，然后将其赋值给self.activation
        self.activation = get_activation(activation) if isinstance(activation, str) else activation     
        self.activation_flag = self.activation is not None    #判断是否设置激活函数，从而设置激活函数的flag

    def construct(self, x):
        x_shape = self.shape_op(x)  #获得x的shape

        if len(x_shape) != 2:       #若x的维度不是2，对x进行调整，x的一个维度是数据的个数
            x = self.reshape(x, (-1, x_shape[-1]))
        x = self.matmul(x, self.weight)    #x与权重
        if self.has_bias:           #若使用偏置，对偏置与x进行相加
            x = self.bias_add(x, self.bias)
        if self.activation_flag:    #若使用激活函数，传入x进行计算
            x = self.activation(x)
        if len(x_shape) != 2:       #调整输出格式
            out_shape = x_shape[:-1] + (-1,)
            x = self.reshape(x, out_shape)
        return x

# 模型测试

# 构建输入为3个节点，输出为4个节点的全连接层
net = Dense(3, 4)
# 输出为2*4矩阵
output = net(x)
print(output)
print(output.shape)
exit(0)