
# 基于MindSpore构造非对称损失函数

import numpy as np
from mindspore import context, set_seed
set_seed(0)
context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

# 生成带有两个标签的数据集
def get_multilabel_data(num, w=2.0, b=3.0):
    for _ in range(num):
        x = np.random.uniform(-10.0, 10.0) # 生成服从(-10.0, 10.0)范围内的均匀分布的元素，返回值的元素类型为浮点型。
        # noise1和noise2为服从标准正态分布的随机值
        noise1 = np.random.normal(0, 1) # 随机产生一个服从正态分布(0,1)的数值
        noise2 = np.random.normal(-1, 1)
        # 定义第一个标签
        y1 = x * w + b + noise1 # 增加噪音生成y                    
        # 定义第二个标签
        y2 = x * w + b + noise2                   
        yield np.array([x]).astype(np.float32), np.array([y1]).astype(np.float32), np.array([y2]).astype(np.float32)
# 定义了一个生成带有两个标签的数据集的函数get_multilabel_data。
# 函数的参数num表示要生成的数据数量，w和b是用于计算标签的参数。
# 在函数内部，使用NumPy生成随机数来生成输入数据x，以及服从标准正态分布的随机噪声noise1和noise2。
# 然后，根据给定的公式计算两个标签y1和y2。最后，使用yield语句返回每个数据样本的输入和两个标签。

def create_multilabel_dataset(num_data, batch_size=16):
    # 加载数据集eval_data=list(get_multilabel_data(5))
    train_data=list(get_multilabel_data(num_data))
    dataset = ds.GeneratorDataset(list(get_multilabel_data(num_data)), column_names=['data', 'label1', 'label2'])
    # 每个batch有16个数据
    dataset = dataset.batch(batch_size) 
    return dataset

# 可视化生成的数据
eval_data=list(get_multilabel_data(5))
x,y1,y2=zip(*eval_data) # zip()函数迭代eval_data，将eval_data中的元素打包成一个个元组，然后返回由这些元组组成的列表。
eval_data

## 模型构建

### 导入Python库和模块


# 时间处理模块
import time
# 科学计算库
import numpy as np
# MindSpore库
import mindspore as ms
# 常见算子操作
import mindspore.ops as ops
# 数据集处理模块
from mindspore import dataset as ds
# 神经网络模块，张量，模型编译
from mindspore import Tensor
# 模型训练设置
from mindspore.train import Callback, LossMonitor, Model
# L1型损失函数
from mindspore.nn import L1Loss
# MindSpore环境设置的0号种子
ms.common.set_seed(0)
import mindspore.nn as nn
from mindspore.common.initializer import Normal

### 定义模型


# 定义线性回归网络
class LinearNet(nn.Cell):
    def __init__(self):
        super(LinearNet, self).__init__()
        self.fc = nn.Dense(1, 1, Normal(0.02), Normal(0.02))

    def construct(self, x):
        return self.fc(x)

### 自定义Focal Loss损失函数


# 定义多标签损失函数
class MAELossForMultiLabel(nn.LossBase):
    def __init__(self, reduction="mean"):
        super(MAELossForMultiLabel, self).__init__(reduction)
        self.abs = ops.abs

    def construct(self, base, target1, target2):
        # 计算第一个标签的误差
        x1 = self.abs(base - target1)
        # 计算第二个标签的误差
        x2 = self.abs(base - target2)
        # 将两误差取平均后作为最终的损失函数值                           
        return (self.get_loss(x1) + self.get_loss(x2))/2   
    
# 自定义Focal Loss损失函数
class FocalLoss(nn.Cell):
    def __init__(self, backbone, loss_fn):
        super(FocalLoss, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, data, label1, label2):
        output = self._backbone(data)
        return self._loss_fn(output, label1, label2)

## 模型训练


ds_train = create_multilabel_dataset(num_data=160)
net = LinearNet()
# 定义多标签损失函数
loss = MAELossForMultiLabel()
# 定义损失网络，连接前向网络和多标签损失函数
loss_net = FocalLoss(net, loss)
# 定义优化器
opt = nn.Momentum(net.trainable_params(), learning_rate=0.005, momentum=0.9)
# 定义Model，多标签场景下Model无需指定损失函数
model = ms.train.Model(network=loss_net, optimizer=opt)
model.train(epoch=10, train_dataset=ds_train, callbacks=[LossMonitor(5)])

## 模型预测


from mindspore.train import Model
from mindspore import Tensor

model_predict = Model(net,loss_net,opt,metrics={"loss"})
# 生成测试数据
w=2.0
b=3.0
x = np.random.uniform(-10.0, 10.0, (1,1))
x1 = np.array([x]).astype(np.float32)
# 定义第一个标签
true_result1 = x * w + b      
# 定义第二个标签
true_result2 = x * w + b            
print('data:' + '%s'%x)
# 模型测试
test_result = model_predict.predict(Tensor(x1))
print('predict result:' + '%s'%test_result)
print('true result1:' + '%s'%true_result1)
print('true result2:' + '%s'%true_result2)
exit(0)