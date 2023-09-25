
## 基于MindSpore实现二维线性回归


import random
import numpy as np
import mindspore
from mindspore import dtype as mstype
import mindspore.ops as ops
from matplotlib import pyplot as plt
import sys
sys.path.append('..')

from mindspore import context, set_seed
set_seed(0)
context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

def synthetic_data(w, b, num_examples):  
    print((num_examples, len(w)))
    # 生成X
    X = np.random.normal(0, 1, (num_examples, len(w))).astype(np.float32)
    # y = Xw + b
    y = np.matmul(X, w) + b 
    # y = Xw + b + 噪声。
    y += np.random.normal(0, 0.01, len(y)).astype(np.float32)          
    return X, y.reshape((-1, 1))

mindspore.set_seed(1)
true_w = np.array([2, -3.4]).astype(np.float32)
true_b = np.float32(4.2)
# 人造数据
features, labels = synthetic_data(true_w, true_b, 1000)

#### 数据加载

print('features:', features[0:4],'\nlabel:', labels[0:4])
# 画出第二个特征与真实值的散点图
plt.scatter(features[:, (1)], labels, 1); 
plt.show()


from mindspore import dataset as ds  

class DatasetGenerator:
    def __init__(self):
        self.data = features
        self.label = labels

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)
        
batch_size = 10
dataset_generator = DatasetGenerator()
dataset = ds.GeneratorDataset(dataset_generator, ["data", "label"], shuffle=True)
# 将数据集中连续10条数据合并为一个批处理数据
dataset = dataset.batch(batch_size)                

### 模型构建


import mindspore.nn as nn
from mindspore import Parameter
from mindspore.common.initializer import initializer, Zero, Normal


def linreg(x, w, b):
    # y = Xw+b
    return ops.matmul(x, w) + b    


class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.w = Parameter(initializer(Normal(0.01, 0), (2, 1), mstype.float32))
        self.b = Parameter(initializer(Zero(), 1, mstype.float32))
        
    def construct(self, x):
        # y_hat = Xw+b
        y_hat = linreg(x, self.w, self.b)  
        return y_hat
    
    
# Net用于实现二维线性回归
net = Net()


# 训练的eopch为3
num_epochs = 3
# 学习率为0.03
lr = 0.03
# Adam优化器
optim = nn.Adam(net.trainable_params(), learning_rate=lr)          
# 计算预测值与标签值之间的均方误差
loss = nn.MSELoss()   

### 模型训练

from mindspore.train import Model                                       
from mindspore.train import LossMonitor 
# 模型训练或推理的高阶接口。Model 会根据用户传入的参数封装可训练或推理的实例
model = Model(net, loss_fn=loss, optimizer=optim)  
# 模型训练接口。训练场景下，LossMonitor监控训练的loss；边训练边推理场景下，监控训练的loss和推理的metrics。如果loss是NAN或INF，则终止训练
model.train(num_epochs, dataset, callbacks=[LossMonitor()])  

### 模型预测

# w的真实值和训练值之差
print(f'w的估计误差: {true_w - net.trainable_params()[0].reshape(true_w.shape)}')  
# b的真实值和训练值之差
print(f'b的估计误差: {true_b - net.trainable_params()[1]}')
exit(0)
