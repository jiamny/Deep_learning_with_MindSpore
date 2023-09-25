
# 基于MindSpore构造平方损失函数 


import numpy as np
from mindspore import dataset as ds

from mindspore import context, set_seed
set_seed(0)
context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

# 获取数据
def get_data(num, a=2.0, b=3.0, c=5.0):
    
    for _ in range(num):
        # 均匀分布
        x = np.random.uniform(-1.0, 1.0)
        y = np.random.uniform(-1.0, 1.0)
        # 添加噪声
        noise = np.random.normal(0, 0.03)
        z = a * x ** 2 + b * y ** 3 + c + noise
        yield np.array([[x**2], [y**3]],dtype=np.float32).reshape(1,2), np.array([z]).astype(np.float32)

# 生成数据集并增强
def create_dataset(num_data, batch_size=16, repeat_size=1):
    # 生成数据集
    input_data = ds.GeneratorDataset(list(get_data(num_data)), column_names=['xy','z'])
    # 划分批次
    input_data = input_data.batch(batch_size)
    # 增强数据集
    input_data = input_data.repeat(repeat_size)
    return input_data
 
data_number = 160       # 数据集大小
batch_number = 10       # 批量大小  
repeat_number = 10      # 增强次数

# 训练集
ds_train = create_dataset(data_number, batch_size=batch_number, repeat_size=repeat_number)
# 测试集
ds_test = create_dataset(data_number, batch_size=batch_number, repeat_size=repeat_number)

# 模型构建


## 导入所需库和函数

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
# 神经网络模块，张量
from mindspore import nn, Tensor
# 模型训练设置
from mindspore.train import Callback, LossMonitor, Model
# L1型损失函数
from mindspore.nn import L1Loss
# MindSpore环境设置的0号种子
ms.set_seed(0)

## 定义一个简单的线性模型

# 定义线性模型
class LinearNet(nn.Cell):
    
    def __init__(self):
        super(LinearNet, self).__init__()
        # 全连接层
        self.fc = nn.Dense(2, 1, 0.02, 0.02)
 
    def construct(self, x):
        # 前向传播
        x = self.fc(x)
        return x


start_time = time.time()
net = LinearNet()
model_params = net.trainable_params()

# 显示模型的参数及其大小
print ('Param Shape is: {}'.format(len(model_params)))
for net_param in net.trainable_params():
    print(net_param, net_param.asnumpy())

## 重写L1型损失函数，实现自定义的平方损失函数

#自定义平方损失函数
class quadratic_loss(L1Loss):
    def __init__(self, reduction="mean"):
        super(quadratic_loss, self).__init__(reduction)
 
    def construct(self, base, target):
        # 平方损失函数
        x = 0.5 * ops.square(base - target)
        return self.get_loss(x)

user_loss = quadratic_loss()

# 模型训练

# 选择动量优化器
optim = nn.Momentum(net.trainable_params(), learning_rate=0.01, momentum=0.6)
# 使用Model接口将网络、损失函数和优化器关联起来
model = Model(net, user_loss, optim)
 
# 开始训练
epoch = 1
model.train(epoch, ds_train, callbacks=[LossMonitor(10)], dataset_sink_mode=True)
 
# 显示模型参数
for net_param in net.trainable_params():
    print(net_param, net_param.asnumpy())

# 显示训练时间
print ('The total time cost is: {}s'.format(time.time() - start_time))

# 模型预测

# 模型预测
model = Model(net, loss_fn=user_loss, optimizer=None, metrics={'loss'})
# 计算测试集的平方损失函数值
pred_loss = model.eval(ds_test, dataset_sink_mode=False)
print(f'prediction loss is {pred_loss}')
exit(0)