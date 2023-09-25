
# 基于MindSpore实现多分类损失函数

import numpy as np

from mindspore import dataset as ds
import mindspore.nn as nn
import mindspore as ms
from mindspore import context, set_seed
set_seed(0)
context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

# 生成带有两个标签的数据集
def get_multilabel_data(num, w=2.0, b=3.0):
    for _ in range(num):
        x = np.random.uniform(-10.0, 10.0)
        # noise1和noise2为服从标准正态分布的随机值
        noise1 = np.random.normal(0, 1)
        noise2 = np.random.normal(-1, 1)
        # 定义第一个标签
        y1 = x * w + b + noise1                   
        # 定义第二个标签
        y2 = x * w + b + noise2                   
        yield np.array([x]).astype(np.float32), np.array([y1]).astype(np.float32), np.array([y2]).astype(np.float32)

def create_multilabel_dataset(num_data, batch_size=16):
    dataset = ds.GeneratorDataset(list(get_multilabel_data(num_data)), column_names=['data', 'label1', 'label2'])
    # 每个batch有16个数据
    dataset = dataset.batch(batch_size) 
    return dataset

## 模型构建

### 多标签损失函数

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

### 定义损失函数

# 自定义损失网络
class CustomWithLossCell(nn.Cell):
    def __init__(self, backbone, loss_fn):
        super(CustomWithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, data, label1, label2):
        output = self._backbone(data)
        return self._loss_fn(output, label1, label2)

### 定义网络模型

from mindspore.common.initializer import Normal
import mindspore.ops as ops
from mindspore.train import LossMonitor
# 定义线性回归网络
class LinearNet(nn.Cell):
    def __init__(self):
        super(LinearNet, self).__init__()
        self.fc = nn.Dense(1, 1, Normal(0.02), Normal(0.02))

    def construct(self, x):
        return self.fc(x)

## 模型训练

ds_train = create_multilabel_dataset(num_data=160)
net = LinearNet()
# 定义多标签损失函数
loss = MAELossForMultiLabel()
# 定义损失网络，连接前向网络和多标签损失函数
loss_net = CustomWithLossCell(net, loss)
# 定义优化器
opt = nn.Momentum(net.trainable_params(), learning_rate=0.005, momentum=0.9)
# 定义Model，多标签场景下Model无需指定损失函数
model = ms.train.Model(network=loss_net, optimizer=opt)

model.train(epoch=10, train_dataset=ds_train, callbacks=[LossMonitor(1)])

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