
# 基于MindSpore构造多层网络模型

import numpy as np

import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.dataset as ds
from mindspore import ParameterTuple
from mindspore.train import Model, Callback
from mindspore import context, set_seed
set_seed(0)
context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

# 随机种子设为4，使每次得到的数据相同
np.random.seed(4)

# 产生数据和标签
class DatasetGenerator:
    def __init__(self):
        self.data = np.random.randn(5, 5).astype(np.float32)
        self.label = np.random.randn(5, 1).astype(np.int32)

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

# 对输入数据进行处理
dataset_generator = DatasetGenerator()
dataset = ds.GeneratorDataset(dataset_generator, ["data", "label"], shuffle=True)
dataset = dataset.batch(32)

## 模型构建

# 定义多层神经网络
class LinearNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.dense1 = nn.Dense(5, 32)
        self.dense2 = nn.Dense(32, 1)

    def construct(self, x):
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        return x

## 网络训练

# 定义LossMonitor回调函数
class LossMonitor(Callback):
    def __init__(self):
        super(LossMonitor, self).__init__()

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        print("Step: {}, Loss: {}".format(cb_params.cur_step_num, cb_params.net_outputs.asnumpy()))

# 定义模型和损失函数
net = LinearNet()
loss = nn.MSELoss()

# 定义优化器
optim = nn.Adam(net.trainable_params())

model = Model(net, loss_fn=loss, optimizer=optim, metrics=None)
model.train(300, dataset, callbacks=[LossMonitor()])

## 模型预测

from mindspore import Tensor

# 生成测试数据
np.random.seed(1)
test_data = np.random.randn(1, 5).astype(np.float32)
print('data:' + '%s'%test_data)
# 模型测试
test_result = model.predict(Tensor(test_data))
print('predict result:' + '%s'%test_result)
exit(0)
