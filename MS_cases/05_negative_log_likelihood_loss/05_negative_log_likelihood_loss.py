
# 基于Mindspore构造负对数似然损失函数

#导入mindspore框架。
import mindspore
from mindspore import context, set_seed
set_seed(0)
context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

#将Model用于创建模型对象，完成网络搭建和编译，并用于训练和评估
from mindspore import nn 
#导入可用于Cell的构造函数的算子。
from mindspore import ops 
# 给定一个矩阵X，我们可以对所有元素求和。
X = mindspore.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], mindspore.float32)
X.sum(0, keepdims=True), X.sum(1, keepdims=True)

# 实验过程


## 导入相关库
import numpy as np
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as mnp
from mindspore import Tensor, Parameter
from mindspore.common.initializer import initializer, Zero, Normal

# 模型构建

# softmax函数用于将其转换为概率分布。softmax函数对每个样本（行）的元素进行指数运算，
# 然后对每个样本的元素求和，得到一个分区（partition）向量，其中每个元素表示对应样本的所有类别的指数和。
def softmax(X):
    X_exp = ops.exp(X)
    partition = X_exp.sum(1, keepdims=True)
    return X_exp / partition
# 定义模型
class Net(nn.Cell):
    def __init__(self, num_inputs, num_outputs):
        super().__init__
        self.W = Parameter(initializer(Normal(0.01, 0), (num_inputs, num_outputs), mindspore.float32))
        self.b = Parameter(initializer(Zero(), num_outputs, mindspore.float32))

    def construct(self, X):
        return softmax(ops.matmul(X.reshape((-1, self.W.shape[0])), self.W) + self.b)


# 生成标签
x = mindspore.Tensor([0, 2], mindspore.int32)
# 生成样本
x_sample = mindspore.Tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]], mindspore.float32)
x_sample[[0, 1], x]
# 实现负对数似然损失函数
def NLLLoss(x_sample, x):
    return -mnp.log(x_sample[mnp.arange(x_sample.shape[0]), x])
NLLLoss(x_sample, x)

# 模型预测

# 计算预测正确的数量
def accuracy(x_sample, x): 
    # 函数检查x_sample的形状是否大于1维且第二个维度大于1。
    # 如果满足条件，说明x_sample是多类别预测的概率分布结果，需要使用argmax函数获取最大概率对应的类别索引。 
    if len(x_sample.shape) > 1 and x_sample.shape[1] > 1:  
        x_sample = x_sample.argmax(axis=1)
    # 使用float(cmp.sum())计算预测正确的数量，cmp.sum()返回布尔数组中为True的元素数量，即预测正确的数量。              
    cmp = x_sample.asnumpy() == x.asnumpy()            
    return float(cmp.sum())

# 将预测正确的数量除以样本的总数量，得到准确率。   
print(accuracy(x_sample, x) / len(x))
exit(0)