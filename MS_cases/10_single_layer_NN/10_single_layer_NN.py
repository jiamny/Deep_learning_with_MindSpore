
# 单层网络模型

# 处理文件和目录
import os
# 处理csv文件
import csv
# 处理大型矩阵
import numpy as np
# 美观输出
from pprint import pprint
# 数据集处理库
from mindspore import dataset
# 导入sklearn获取鸢尾花数据集
from sklearn.datasets import load_iris
from mindspore import context, set_seed
set_seed(0)
context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

### 数据加载

file = open('../../data/iris.data', 'w', encoding='utf-8')
iris = load_iris()
x = iris.data
y = iris.target
for i in range(len(x)):
    row = ""
    for j in x[i]:
        if row == "":
            row = row + str(j)
        else:
            row = row + "," + str(j)
    if y[i] == 0:
        row = row + "," + "\"setosa\""
    if y[i] == 1:
        row = row + "," + "\"versicolor\""
    if y[i] == 2:
        row = row + "," + "\"virginica\""
    if i < 6:
        print(row)
    file.write(row + "\n")
file.close()


def create_dataset(data_path):
    # 读取文件
    with open(data_path) as csv_file:
        data = list(csv.reader(csv_file, delimiter=','))
    # 输出部分数据
    pprint(data[0:5]); pprint(data[50:55]); pprint(data[100:105])
    # 设置数据标签
    label_map = {
        'setosa': 0,
        'versicolor': 1,
        'virginica': 2
    }
    # 获取数据集x
    X = np.array([[float(x) for x in s[:-1]] for s in data[:150]], np.float32)
    # 获取数据集标签y
    Y = np.array([label_map[s[-1]] for s in data[:150]], np.int32)


    # 分割数据集，训练集和测试集比例为8:2
    train_idx = np.random.choice(150, 120, replace=False)
    test_idx = np.array(list(set(range(150)) - set(train_idx)))
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]

    # 将数据转换为MindSpore的数据格式
    XY_train = list(zip(X_train, Y_train))
    ds_train = dataset.GeneratorDataset(XY_train, ['x', 'y'])
    ds_train = ds_train.shuffle(buffer_size=120).batch(32, drop_remainder=True)

    XY_test = list(zip(X_test, Y_test))
    ds_test = dataset.GeneratorDataset(XY_test, ['x', 'y'])
    ds_test = ds_test.batch(30)

    return ds_train, ds_test


# 读取数据集
data_url = "../../data/iris.data"
# 创建数据集
data_train, data_test = create_dataset(data_url)

# 模型构建

# 导入MindSpore
import mindspore as ms
# 神经网络Cell，用于构建神经网络中的预定义构建块或计算单元。
from mindspore import nn
# LossMonitor训练场景下，监控训练的loss；边训练边推理场景下，监控训练的loss和推理的metrics。
from mindspore.train import LossMonitor
def softmax_regression(ds_train, ds_test):
    # 构建softmax回归分类模型
    # Dense层in_channels=4，out_channels=3。
    net = nn.Dense(4, 3)
    # 使用交叉熵损失函数计算出输入概率（使用softmax函数计算）和真实值之间的误差。
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    # 定义优化器
    opt = nn.Momentum(net.trainable_params(), learning_rate=0.05, momentum=0.9)
    # 模型训练或推理的高阶接口。 Model 会根据用户传入的参数封装可训练或推理的实例。
    model = ms.train.Model(net, loss, opt, metrics={'acc', 'loss'})
    return model

# 模型训练

# 设置运行环境
ms.set_context(mode=0, device_target="CPU")
# 声明一个模型
model = softmax_regression(data_train, data_test)
# 模型训练
model.train(25, data_train, callbacks=[LossMonitor(per_print_times=data_train.get_dataset_size())],
            dataset_sink_mode=False)

# 7.模型预测¶

# 模型预测
metrics = model.eval(data_test)
print(metrics)
exit(0)