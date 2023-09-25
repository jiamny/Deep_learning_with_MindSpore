
## 基于MindSpore实现一个推荐系统示例


import os  
import mindspore 
import urllib.request

'''
def download_data(file):
    # 文件下载地址
    url = 'https://files.grouplens.org/datasets/movielens/ml-100k/' + file
    print(url)
    # 将文件下载到data文件夹中
    urllib.request.urlretrieve(url, 'data/'+file)
    print('File download completed')    
    
if os.path.exists('data') is False:
    # data文件夹不存在则创建
    os.mkdir('data')
# 下载本实验用到的五个数据集
download_data('ua.base')
download_data('ua.test')
download_data('u.user')
download_data('u.item')
download_data('u.occupation')
'''

base_path='/media/hhj/localssd/DL_data/movielens'
# 训练集路径
train_path = os.path.join(base_path, 'ua.base')                        
# 测试集路径
test_path = os.path.join(base_path, 'ua.test')                         
# 用户信息的文件路径
user_path = os.path.join(base_path, 'u.user')
# 作品信息的文件路径
item_path = os.path.join(base_path, 'u.item')
# 用户职业的文件路径
occupation_path = os.path.join(base_path, 'u.occupation')
# 设置全局种子
mindspore.set_seed(1)


# 默认编码为UTF-8
def show_data(data_path, encoding='UTF-8'):
    count = 1
    with open(data_path, 'r', encoding=encoding) as f:
        for cur_line in f.readlines():
            print(cur_line)
            if count == 5:
                break
            count += 1

            
print('ua.base & ua.test:')
show_data(train_path)
print('u.user:')
show_data(user_path)
print('u.item:')
show_data(item_path, 'ISO-8859-1')
print('u.occupation:')
show_data(occupation_path)


# numpy工具包提供了一系列类NumPy接口。
import numpy as np

def get1or0(r):
    # 评分大于3视为用户已点击，否则未点击
    return 1.0 if r > 3 else 0.0                                

# 以字典的形式返回评分数据
def __read_rating_data(path):                                          
    dataSet = {}
    with open(path, 'r') as f:
        # 读取每一行
        for line in f.readlines():  
            # 读取以制表符隔开的用户id、作品id、评分
            d = line.strip().split('\t')   
            # 会将评分转换为是否点击
            dataSet[(int(d[0]), int(d[1]))] = [get1or0(int(d[2]))]     
    return dataSet


# 以字典形式返回电影信息
def __read_item_hot():                                                 
    items = {}
    with open(item_path, 'r', encoding='ISO-8859-1') as f:
        for line in f.readlines():
            # 读取以'|'隔开的每个元素
            d = line.strip().split('|') 
            # 字典键为作品id，值为作品类型被one-hot编码的向量
            items[int(d[0])] = np.array(d[5:], dtype='float64')        
    return items


# 以字典形式返回每个职业
def __read_occupation_hot():                                           
    occupations = {}
    with open(occupation_path, 'r') as f:
        # 读取以换行符隔开的每个职业
        names = f.read().strip().split('\n')                           
    length = len(names)
    for i in range(length):  
        # 为每个职业都生成一个one-hot向量
        l = np.zeros(length, dtype='float64')
        l[i] = 1
        occupations[names[i]] = l
    return occupations


# 以字典形式返回每个用户信息
def __read_user_hot():                                                  
    users = {}
    gender_dict = {'M': 1, 'F': 0}
    # 读取职业信息
    occupation_dict = __read_occupation_hot()                           
    with open(user_path, 'r') as f:
        for line in f.readlines():
            # 读取以'|'隔开的用户id、年龄、性别、职业、邮政编码
            d = line.strip().split('|')                                 
            a = np.array([int(d[1]), gender_dict[d[2]]])
            # 字典键为用户id，值为年龄、性别、职业组成的向量
            users[int(d[0])] = np.append(a, occupation_dict[d[3]])      
    return users


def read_dataSet(user_dict, item_dict, path):
    X, Y = [], []
    # 读取评分数据
    ratings = __read_rating_data(path)                                  
    for k in ratings:
        # X为年龄、性别、职业、作品类型组成的向量
        X.append(np.append(user_dict[k[0]], item_dict[k[1]])) 
        # Y为是否已点击
        Y.append(ratings[k])                                            
    return X, Y


def read_data():
    user_dict = __read_user_hot()
    item_dict = __read_item_hot()
    # 返回训练集
    trainX, trainY = read_dataSet(user_dict, item_dict, train_path) 
    # 返回测试集
    testX, testY = read_dataSet(user_dict, item_dict, test_path)        
    return trainX, trainY, testX, testY

#### 数据加载

from mindspore import dataset as ds  

def get_data(train=True):
    # 读取训练集和测试集
    trainX, trainY, testX, testY = read_data()                          
    train_size = len(trainX)
    test_size = len(testX)
    if train == True:
        for i in range(train_size):
            # 训练集标签
            y = trainY[i]     
            # 训练集数据
            x = trainX[i]                                               
            yield np.array(x[:]).astype(np.float32), np.array([y[0]]).astype(np.float32)
    else:
        for i in range(test_size):
            # 测试集标签
            y = testY[i]          
            # 测试集数据
            x = testX[i]                                                
            yield np.array(x[:]).astype(np.float32), np.array([y[:]]).astype(np.float32)

def create_dataset(batch_size=32, repeat_size=1, train=True):
    # 使用GeneratorDataset创建可迭代数据
    input_data = ds.GeneratorDataset(list(get_data(train)), column_names=['data', 'label'])  
    # 将数据集中连续32条数据合并为一个批处理数据
    input_data = input_data.batch(batch_size)                
    # 重复数据集1次
    input_data = input_data.repeat(repeat_size)                          
    return input_data

### 模型构建


import mindspore.nn as nn                                               
from mindspore.common.initializer import Normal 
class Network(nn.Cell):
    def __init__(self):
        super(Network, self).__init__()
        # 全连接层的输入维度为42， 输出维度为21
        self.fc1 = nn.Dense(42, 21, Normal(0.02), Normal(0.02)) 
        # 输入维度为21，输出维度为1
        self.fc2 = nn.Dense(21, 1, Normal(0.02), Normal(0.02))    
        # ReLU函数
        self.relu = nn.ReLU()                          
        # Sigmoid函数
        self.sigmoid = nn.Sigmoid()                                   
        
    def construct(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x    
    
net = Network()
# 查看网络结构
print(net)                                                                  


# epoch为10
epochs = 10      
# batch_size为32
batch_size = 32        
# 学习率为0.01
learning_rate = 1e-3                                                     
# 计算预测值和真实值之间的二值交叉熵损失。
loss_fn = nn.BCEWithLogitsLoss(reduction="mean")       
# Adam优化器
optimizer = nn.Adam(net.trainable_params(), learning_rate=learning_rate)

### 模型训练

from mindspore.train import Model                                       
from mindspore.train import LossMonitor 
# 创建训练集
train_dataset = create_dataset(batch_size)         
# 创建测试集
test_dataset = create_dataset(batch_size, train=False)                      
# 模型训练或推理的高阶接口。Model 会根据用户传入的参数封装可训练或推理的实例
model = Model(net, loss_fn=loss_fn, optimizer=optimizer, metrics={"acc"})  
# 模型训练接口。训练场景下，LossMonitor监控训练的loss；边训练边推理场景下，监控训练的loss和推理的metrics。如果loss是NAN或INF，则终止训练
model.train(epochs, train_dataset, callbacks=[LossMonitor(per_print_times=75)])     
# 将网络权重保存到checkpoint文件中
mindspore.save_checkpoint(net, "./MyNet.ckpt")                                    

### 模型预测

count = 1
pre_ctr_list = []
real_ctr_list = []
for data, label in test_dataset:
    # 模型预测
    pre = model.predict(data)                                            
    
    # 若预测点击概率大于等于0.5，则表示为点击，点击计数加一
    click_count = 0
    for i in range(pre.shape[0]):
        if pre[i] >=0.5:
            click_count += 1
    # 输出前十个batch的CTR预测结果
    if count < 10:
        print("the predicted CTR is {}".format(click_count/batch_size))
    pre_ctr_list.append(click_count/batch_size)
    
    # 点击计数的真实值统计
    click_count = 0
    for i in range(label.shape[0]):
        if label[i][0] == 1:
            click_count += 1
    # 输出前十个batch的CTR真实值
    if count < 10:
        print("the real CTR is {}".format(click_count/batch_size))
    real_ctr_list.append(click_count/batch_size)
    
    count += 1

# 平均CTR预测值
print('The average predicted CTR is {}'.format(np.mean(pre_ctr_list)))
# 平均CTR真实值
print('The average real CTR is {}'.format(np.mean(real_ctr_list)))
exit(0)