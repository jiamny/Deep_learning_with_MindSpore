
# K近邻算法实现红酒聚类

# 下载红酒数据集
#"https://ascend-professional-construction-dataset.obs.cn-north-4.myhuaweicloud.com:443/MachineLearning/wine.zip"

### 数据读取与处理
#### 导入MindSpore模块和辅助模块

import csv
import numpy as np
import matplotlib.pyplot as plt

import mindspore as ms
from mindspore import nn, ops

ms.set_context(device_target="CPU")

#### 读取Wine数据集`wine.data`，并查看部分数据。
with open('wine.data') as csv_file:
    data = list(csv.reader(csv_file, delimiter=','))
print(data[56:62]+data[130:133])

#### 取三类样本（共178条），将数据集的13个属性作为自变量$X$。将数据集的3个类别作为因变量$Y$。
X = np.array([[float(x) for x in s[1:]] for s in data[:178]], np.float32)
Y = np.array([s[0] for s in data[:178]], np.int32)

#### 取样本的某两个属性进行2维可视化，可以看到在某两个属性上样本的分布情况以及可分性。
attrs = ['Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols',
         'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue',
         'OD280/OD315 of diluted wines', 'Proline']
plt.figure(figsize=(10, 8))
for i in range(0, 4):
    plt.subplot(2, 2, i+1)
    a1, a2 = 2 * i, 2 * i + 1
    plt.scatter(X[:59, a1], X[:59, a2], label='1')
    plt.scatter(X[59:130, a1], X[59:130, a2], label='2')
    plt.scatter(X[130:, a1], X[130:, a2], label='3')
    plt.xlabel(attrs[a1])
    plt.ylabel(attrs[a2])
    plt.legend()
plt.show()

#### 将数据集按128:50划分为训练集（已知类别样本）和验证集（待验证样本）：
train_idx = np.random.choice(178, 128, replace=False)
test_idx = np.array(list(set(range(178)) - set(train_idx)))
X_train, Y_train = X[train_idx], Y[train_idx]
X_test, Y_test = X[test_idx], Y[test_idx]

## 模型构建--计算距离

class KnnNet(nn.Cell):
    def __init__(self, k):
        super(KnnNet, self).__init__()
        self.k = k

    def construct(self, x, X_train):
        #平铺输入x以匹配X_train中的样本数
        x_tile = ops.tile(x, (128, 1))
        square_diff = ops.square(x_tile - X_train)
        square_dist = ops.sum(square_diff, 1)
        dist = ops.sqrt(square_dist)
        #-dist表示值越大，样本就越接近
        values, indices = ops.topk(-dist, self.k)
        return indices

def knn(knn_net, x, X_train, Y_train):
    x, X_train = ms.Tensor(x), ms.Tensor(X_train)
    indices = knn_net(x, X_train)
    topk_cls = [0]*len(indices.asnumpy())
    for idx in indices.asnumpy():
        topk_cls[Y_train[idx]] += 1
    cls = np.argmax(topk_cls)
    return cls

## 模型预测

acc = 0
knn_net = KnnNet(5)
for x, y in zip(X_test, Y_test):
    pred = knn(knn_net, x, X_train, Y_train)
    acc += (pred == y)
    print('label: %d, prediction: %s' % (y, pred))
print('Validation accuracy is %f' % (acc/len(Y_test)))

