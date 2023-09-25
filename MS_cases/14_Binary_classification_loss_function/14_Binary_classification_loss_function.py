
# 基于Mindspore实现二分类损失函数

import numpy as np                         # Python 的科学计算库，用于处理矩阵和数组等数据结构。
from mindspore import context, set_seed
set_seed(0)
context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

def get_data(num, w=2.0, b=3.0):           # 生成数据及对应标签   
    for _ in range(num):                   #  num=160,生成160个样本点
        x = np.random.uniform(-10.0, 10.0) # 生成服从(-10.0, 10.0)范围内的均匀分布的元素，返回值的元素类型为浮点型。
        noise = np.random.normal(0, 1)     # 随机产生一个服从正态分布(0,1)的数值
        y = x * w + b + noise              # 增加噪音生成y
        yield np.array([x]).astype(np.float32), np.array([y]).astype(np.float32)#将数组元素类型转换为float32位;
                                           # 我们为了提高效率，并不一次性返回所有数据，而是采用迭代器形式返回单一数据。
#可视化部分生成数据
eval_data=list(get_data(5))
#zip()函数迭代eval_data，将eval_data中的元素打包成一个个元组，然后返回由这些元组组成的列表。
x,y=zip(*eval_data)
#可视化生成的5个样本点
eval_data 

### 数据加载

from mindspore import dataset as ds
def create_dataset(num_data, batch_size=16):                           #加载数据集
    data=list(get_data(num_data))
    dataset = ds.GeneratorDataset(data, column_names=['data', 'label'])#指定生成数据集的列名为data和lable
    dataset = dataset.batch(batch_size)                                #设置数据批次
    return dataset        

## 几个损失函数
### 内置损失函数

# 内置损失函数
import mindspore as ms
import mindspore.nn as nn 
loss = nn.L1Loss()                       # 输出loss均值
loss_sum = nn.L1Loss(reduction='sum')    # 输出loss和
loss_none = nn.L1Loss(reduction='none')  # 输出loss原值
input_data = ms.Tensor(np.array([1, 0, 1, 0, 1, 0]).astype(np.float32)) # 定义输入数据
target_data = ms.Tensor(np.array([0, 0, 1, 1, 1, 0]).astype(np.float32)) # 定义标签
print("loss:", loss(input_data, target_data))             # 打印loss均值
print("loss_sum:", loss_sum(input_data, target_data))     # 打印所有loss和
print("loss_none:\n", loss_none(input_data, target_data)) # 打印每个样本点loss的原值

### 基于nn.Cell构造损失函数

# 基于nn.Cell构造损失函数
import mindspore.ops as ops
class MAELoss(nn.Cell):                 # 自定义损失函数MAELoss
    def __init__(self):                 # 初始化
        super(MAELoss, self).__init__()
        self.abs = ops.abs
        self.reduce_mean = ops.ReduceMean()
    def construct(self, base, target):  # 调用算子        
        x = self.abs(base - target)
        return self.reduce_mean(x)
loss = MAELoss()                        # 定义损失函数
input_data = ms.Tensor(np.array([1, 0, 1, 0, 1, 0]).astype(np.float32))  # 定义输入数据
target_data = ms.Tensor(np.array([0, 0, 1, 1, 1, 0]).astype(np.float32)) # 定义标签
output = loss(input_data, target_data)  # 计算损失
print(output)                           # 打印损失

### 基于nn.LossBase构造损失函数

# 基于nn.LossBase构造损失函数
class MAELoss(nn.LossBase):               # 自定义损失函数MAELoss
    def __init__(self, reduction="mean"): # 初始化并求loss均值       
        super(MAELoss, self).__init__(reduction)
        self.abs = ops.abs              # 求绝对值算子
    def construct(self, base, target):    # 调用算子
        x = self.abs(base - target)
        return self.get_loss(x)           # 返回loss均值
loss = MAELoss()                          # 定义损失函数
input_data = ms.Tensor(np.array([1, 0, 1, 0, 1, 0]).astype(np.float32))  # 生成预测值
target_data = ms.Tensor(np.array([0, 0, 1, 1, 1, 0]).astype(np.float32))  # 生成真实值
output = loss(input_data, target_data)    # 计算损失
print(output)                             # 打印损失

## 模型构建

# 定义模型
#使用了MindSpore的神经网络模块中的dense函数，该函数用于创建全连接。这里创建了一个输入维度为1，输出维度为1的全连接层。
net = nn.Dense(1, 1)

# 定义损失函数
#使用了MindSpore的神经网络模块中的L1Loss函数，该函数用于计算 L1 Loss（也称为绝对值误差）。
loss_fn = nn.L1Loss()

# 定义优化器
#使用了MindSpore的优化器模块中的Momentum函数，它是一个带动量随机梯度下降法（SGD）。
#Momentum算法的基本思路是使用上一步的梯度方向来决定本步的梯度方向，从而解决随机梯度下降法中的震荡问题。
#它可以加速收敛，而且对于参数的自适应能力也更强。
#learning_rate为学习率，momentum为动量参数。
optimizer = nn.Momentum(net.trainable_params(), learning_rate=0.005, momentum=0.9)

## 模型训练

from mindspore.train import Model, MAE, LossMonitor 

#将定义好的神经网络模型、损失函数和优化器用Model函数封装，指定了评价指标为MAE（平均绝对误差）。
#这个函数在训练过程中会自动计算每个batch的损失值和评价指标，并使用优化器更新模型参数。
model = Model(net, loss_fn, optimizer, metrics={"MAE": MAE()})
train_dataset = create_dataset(num_data=160)          # 生成训练集
eval_dataset = create_dataset(num_data=160)           # 生成测试集
train_dataset_size = train_dataset.get_dataset_size() # 训练集大小

## 模型预测

#指定了一个回调函数LossMonitor来监控训练过程中的loss值，并将训练集大小传递给它。回调函数可以在特定的阶段中被调用，如个epoch结束时。
#此处LossMonitor函数计算并输出每个epoch中的平均训练损失和损失。
model.fit(20, train_dataset, eval_dataset, callbacks=LossMonitor(train_dataset_size))
exit(0)