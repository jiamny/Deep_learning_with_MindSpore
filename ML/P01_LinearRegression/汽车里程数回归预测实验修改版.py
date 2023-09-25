
# 汽车里程数回归预测实验


## 1、实验目的


## 2、全连接神经网络的原理介绍


### 2.1 正向传播


### 2.2 反向传播


## 3、实验环境


## 4、数据处理

### 4.1数据准备

from download import download

# 下载汽车里程auto-mpg数据集
url = " https://ascend-professional-construction-dataset.obs.cn-north-4.myhuaweicloud.com:443/deep-learning/auto-mpg.zip"  
path = download(url, "./", kind="zip", replace=True)

### 4.2数据加载

#导入相关依赖库
import  os
import csv
import time
import numpy as np
import pandas as pd  #版本采用1.3.0
from matplotlib import pyplot as plt

import mindspore as ms
import mindspore.dataset as ds
from mindspore.dataset import GeneratorDataset
import mindspore.context as context
import mindspore.dataset.transforms as C
import mindspore.dataset.vision as CV

from mindspore import nn, Tensor, ops
from mindspore.train import Model
from mindspore.train import Accuracy, MAE, MSE
from mindspore import load_checkpoint, load_param_into_net
from mindspore.train import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')  # device_target支持"Ascend"、"CPU"。


#加载数据集
with open('./auto-mpg.data') as csv_file:
    data = list(csv.reader(csv_file, delimiter=','))
print(data[20:40]) # 打印部分数据


#使用pandas读取数据
column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin']
#遇到？换成nan，忽略\t之后的内容，已空格作为分隔符。
raw_data = pd.read_csv('./auto-mpg.data', names=column_names,
                      na_values = "?", comment='\t',
                      sep=" ", skipinitialspace=True)

data = raw_data.copy()

#查看数据形状
data.shape


#对于数据集中的空值，我们要在建模前进行处理。此处空值的数据较少，我们直接进行删除。
#清洗空数据
data = data.dropna()
data.tail()
#Pandas库提供了简单的数据集统计信息，我们可直接调用函数describe()进行查看。
#查看训练数据集的结构
origin = data.pop('Origin')
data_labels = data.pop('MPG')
train_stats = data.describe()
train_stats = train_stats.transpose()
train_stats
#归一化数据
def norm(x):
    return (x - train_stats['mean']) / train_stats['std']

normed_data = norm(data)
# 将MPG放回归一化后的数据中
normed_data['MPG'] = data_labels
# 离散特征处理
# 特征Origin代表着车辆的归属区域信息，此处总共分为三种，欧洲，美国，日本，我们需要对此特征进行one-hot编码。
# 对origin属性进行one-hot编码
normed_data['USA'] = (origin == 1)*1.0
normed_data['Europe'] = (origin == 2)*1.0
normed_data['Japan'] = (origin == 3)*1.0


#将数据集按照4：1划分成训练集和测试集
train_dataset = normed_data.sample(frac=0.8,random_state=0)
test_dataset = normed_data.drop(train_dataset.index)

#模型训练需要区分特征值与目标值，也就是我们常说的X值与Y值，此处MPG为Y值，其余的特征为X值。
#将目标值和特征分开
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

X_train, Y_train = np.array(train_dataset), np.array(train_labels)
X_test, Y_test = np.array(test_dataset), np.array(test_labels)

#查看数据集尺寸
print('训练数据x尺寸：',X_train.shape)
print('训练数据y尺寸：',Y_train.shape)
print('测试数据x尺寸：',X_test.shape)
print('测试数据y尺寸：',Y_test.shape)
#将数据集转换为Tensor格式
ds_xtrain= Tensor(X_train, ms.float32)
print(ds_xtrain[2])
ds_ytrain= Tensor(Y_train, ms.int32)[:, np.newaxis]
print(ds_ytrain[2])


ds_xtest=Tensor(X_test, ms.float32)
ds_ytest=Tensor(Y_test, ms.int32)
# Iterable object as input source
class Iterable:
    def __init__(self, X_train, Y_train):
        self._data = X_train
        self._label = Y_train[:, np.newaxis]

    def __getitem__(self, index):
        return self._data[index], self._label[index]

    def __len__(self):
        return len(self._data)

data = Iterable(X_train, Y_train)
print(type(data))
dataset_train = GeneratorDataset(source=data, column_names=["data", "label"])
print(type(dataset_train))
print(list(dataset_train.create_tuple_iterator())[2])

## 5、模型构建

# 定义网络
class Regression_car(nn.Cell):
    def __init__(self):
        super(Regression_car, self).__init__()
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.fc1 = nn.Dense(9,64, activation='relu')
        self.fc2 = nn.Dense(64,64, activation='relu')
        self.fc3 = nn.Dense(64,1)
        
        
    def construct(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

## 6、模型训练

#定义网络，损失函数，评估指标  优化器
network = Regression_car()
net_loss = nn.MSELoss()
net_opt = nn.RMSProp(network.trainable_params(), 0.001)

# 定义用于训练的train_loop函数。
def train_loop(model, dataset, loss_fn, optimizer):
    # 定义正向计算函数
    def forward_fn(data, label):
        logits = model(data)
        loss = loss_fn(logits, label)
        return loss, logits

    # 定义自动微分函数，使用mindspore.value_and_grad获得微分函数grad_fn。
    grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    # 定义 one-step training函数
    def train_step(data, label):
        (loss, _), grads = grad_fn(data, label)
        loss = ops.depend(loss, optimizer(grads))
        return loss

    size = dataset.get_dataset_size()
    model.set_train()
    for batch, (data, label) in enumerate(dataset.create_tuple_iterator()):
        # 将喂给网络的数据转置，使其满足网络所需的数据结构
        data, label = ms.Tensor(data[:, np.newaxis].T, ms.float32), ms.Tensor(label[:, np.newaxis], ms.float32)
        #print(data.shape,label.shape)
        loss = train_step(data.astype(np.float32), label.astype(np.float32))

    return loss

#WithEvalCell返回loss、输出和标签的单元，用于评估。此单元接受网络和loss函数作为参数，并计算模型的loss。它返回loss、输出和标签来计算度量。
evalcell=nn.WithEvalCell(network,net_loss)
#创建指标类

mae = nn.MAE()
mse = nn.MSE()
val_mae = nn.MAE()
val_mse = nn.MSE()

#创建一个空的Dataframe
result =pd.DataFrame(columns=('_epoch','_loss','_mae','_mse','val_mae','val_mse'))
print("============== Starting Training ==============")
epochs = 100
for epoch in range(epochs):
    #print(f"Epoch {epoch+1}\n-------------------------------")
    loss = train_loop(network, dataset_train, net_loss, net_opt)
    #print(loss)
    #利用evalcell接收训练集获取训练过程的输出用于计算mae和mse，接收测试集获取测试集输出
    _, outputs, label = evalcell(ds_xtrain,ds_ytrain)
    _, val_outputs, val_label = evalcell(ds_xtest,ds_ytest)
    
    #每次循环都更新MAE、MSE等的值。
    mae.clear()
    mae.update(outputs, label)
    mse.clear()
    mse.update(outputs, label)
    val_mae.clear()
    val_mae.update(val_outputs, val_label)
    val_mse.clear()
    val_mse.update(val_outputs, val_label)
    
    Mae = mae.eval()
    Mse = mse.eval()
    Val_Mae = val_mae.eval()
    Val_Mse = val_mse.eval()
    
    nd_loss = loss.asnumpy()
    fl_loss = float(nd_loss)/24.0

    #将计算结果逐行插入result,注意变量要用[]括起来,同时ignore_index=True，否则会报错，ValueError: If using all scalar values, you must pass an index
    
    result=result.append(pd.DataFrame({'_epoch':[epoch],'_loss':fl_loss,'_mae':Mae,'_mse':Mse,
                                       'val_mae':Val_Mae,'val_mse':Val_Mse}),ignore_index=True)

    if epoch%10==0:
        print('epoch:{0},loss:{1},mae:{2},mse:{3},val_mae:{4},val_mse:{5}'.format(epoch,fl_loss,Mae,Mse,
                                                                                               Val_Mae,Val_Mse))
        print("*" * 110)
print(result)

## 7、模型预测

#绘制模型损失函数图
def plot_history(result):

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(result['_epoch'], result['_mae'],
           label='Train Error')
    plt.plot(result['_epoch'], result['val_mae'],
           label = 'Val Error')
    plt.ylim([0,20])
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(result['_epoch'], result['_mse'],
           label='Train Error')
    plt.plot(result['_epoch'], result['val_mse'],
           label = 'Val Error')
    plt.ylim([0,20])
    plt.legend()
    plt.show()


plot_history(result)
