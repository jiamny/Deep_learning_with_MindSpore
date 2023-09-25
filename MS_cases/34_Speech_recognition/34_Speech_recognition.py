
# 基于MindSpore实现语音识别

'''
# 使用download 来下载数据集
# !pip install download
from download import download
url = "http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz"
download(url, "./data", kind="zip",replace=True)
'''

import numpy as np
import scipy.io.wavfile as wav
import mindspore

mindspore.set_context(mode=mindspore.GRAPH_MODE, device_target="GPU")

# 读取文件并进行特征提取的函数
def get_spectrogram(file_path):
    fs, waveform = wav.read(file_path)
    # 声谱的矩阵大小[124,129]
    # spectrogram = np.zeros([124, 129]).astype(np.float32)
    spectrogram = np.zeros([124, 129]).astype(np.float32)
    # 边距
    zero_padding = np.zeros([16000 - waveform.shape[0]], dtype=np.float32)
    waveform = waveform.astype(np.float32)
    # 扩充到16000
    equal_length = np.concatenate([waveform, zero_padding])
    # 生成0-254每个整数
    x = np.linspace(0, 254, 255, dtype=np.int32)
    # 在数字信号处理中，加窗是音频信号预处理重要的一步
    window = 0.54 - 0.46 * np.cos(2 * np.pi * (x) / (255 - 1))
    for i in range(124):
        # 帧头
        p_start = i * 128
        # 帧尾
        p_end = p_start + 255
        frame_data = equal_length[p_start:p_end]
        frame_data = frame_data * window
        # 离散傅里叶变化
        spectrum = np.fft.rfft(frame_data, n=256)
        # 经过修改后可以使得特征输出为[124,129]
        spectrogram[i,:] = np.abs(spectrum)
    return spectrogram


import os
import random
import glob
import sys

data_dir = '/media/hhj/localssd/DL_data/speech_commands'

# 获取命令列表
# commands = [dir_name for dir_name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, dir_name)) and dir_name != 'README.md']

# 具体识别的8个词
commands = ['yes', 'no', 'up', 'down', 'right', 'left', 'go', 'stop']

# 设置随机种子
seed = 40
random.seed(seed)



# 保存训练和测试文件列表到文件中
train_file_path = 'train_file.txt'
val_file_path = 'val_file.txt'
test_file_path = 'test_file.txt'

# 打乱命令顺序
random.shuffle(commands)

# 获取所有文件名
all_files = []
for command in commands:
    command_path = os.path.join(data_dir, command)
    files = glob.glob(os.path.join(command_path, '*.wav'))
    all_files.extend(files)

# 打乱文件顺序
random.shuffle(all_files)

# train_num = int(len(all_files) * 0.8)
# val_num = int(len(all_files) * 0.1) + 1
train_num = 16000
val_num = 1472
print('len(all_files): ', len(all_files))

# 划分训练、验证和测试集
train_files = all_files[:train_num]
val_files = all_files[train_num: train_num + val_num]
test_files = all_files[-val_num:]

if sys.platform == 'win32':
    # windows系统使用如下代码
    with open(train_file_path, 'w', encoding='utf-8') as file1:
        for f in train_files:
            file1.write(f.replace('\\', '\\\\').replace('/', '\\\\') + '\n')

    with open(val_file_path, 'w', encoding='utf-8') as file1:
        for f in val_files:
            file1.write(f.replace('\\', '\\\\').replace('/', '\\\\') + '\n')

    with open(test_file_path, 'w', encoding='utf-8') as file2:
        for f in test_files:
            file2.write(f.replace('\\', '\\\\').replace('/', '\\\\') + '\n')

if sys.platform == 'linux':
    # Linux系统使用如下代码
    with open(train_file_path, 'w', encoding='utf-8') as file1:
        for f in train_files:
            file1.write(f + '\n')

    with open(val_file_path, 'w', encoding='utf-8') as file1:
        for f in val_files:
            file1.write(f + '\n')

    with open(test_file_path, 'w', encoding='utf-8') as file2:
        for f in test_files:
            file2.write(f + '\n')


# 获取音频标签
# 读取特征文件中的数据
def get_data(file_path):
    with open(file_path, 'r', encoding='utf8') as f:
        files = f.readlines()
    # 逐行读取
    for line in files:
        line = line.strip()
        # 提取label
        data = get_spectrogram(line) 
        if sys.platform == 'win32':
            # windows系统使用如下代码
            label = line.split('\\\\')[-2]
        if sys.platform == 'linux':
            # Linux系统使用如下代码
            label = line.split('/')[-2]        
        label_id = commands.index(label)
        # print(data,label_id,"##")
        yield np.array(data,dtype=np.float32), label_id


import mindspore.dataset as ds
batch_size = 16

# 意为数据集本身每一条数据都可以通过索引直接访问
ds_train = ds.GeneratorDataset(list(get_data(train_file_path)), column_names=['data', 'label'])
# 批处理,分为64批
ds_train = ds_train.batch(batch_size)


## 模型构建

from mindspore.nn import Conv2d
from mindspore.nn import MaxPool2d
from mindspore.nn import Cell
import mindspore.ops as P
from mindspore.nn import Dense
from mindspore.nn import ReLU
from mindspore.nn import Flatten
from mindspore.nn import Dropout
from mindspore.nn import BatchNorm2d

# 实现二维卷积操作
def conv2d(in_channels, out_channels):
    return Conv2d(in_channels=in_channels, out_channels=out_channels,
                  kernel_size=3, stride=1, pad_mode='valid',
                  has_bias=True, weight_init='he_normal')               
# 池化层
def maxpool():
    return MaxPool2d(kernel_size=(2, 2), stride=(2, 2), pad_mode='valid')

# 定义网络
class Net(Cell):
    def __init__(self, batch_size):
        super(Net, self).__init__()
        # 向网络中加层
        self.batch_size = batch_size
        self.reshape = P.Reshape()
        self.resize = P.ResizeNearestNeighbor(size=(32, 32))
        self.norm = BatchNorm2d(num_features=1)
        self.conv1 = conv2d(1, 32)
        self.relu1 = ReLU()
        self.conv2 = conv2d(32,64)
        self.relu2 = ReLU()
        self.maxpool = maxpool()
        self.dropout1 = Dropout(p=0.25)
        self.flatten = Flatten()
        self.dense1 = Dense(in_channels=12544, out_channels=128)
        self.relu3 = ReLU()
        self.dropout2 = Dropout(p=0.5)
        self.dense2 = Dense(in_channels=128, out_channels=8)
    
    def construct(self, input_x):
        x = self.reshape(input_x, (self.batch_size,1, 124, 129))
        x = self.resize(x)
        x = self.norm(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool(x)
        x = self.dropout1(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.relu3(x)
        x = self.dropout2(x)
        x = self.dense2(x)
        return x


from mindspore.train import Model
from mindspore.nn import Adam
from mindspore.nn import SoftmaxCrossEntropyWithLogits

# 构建网络
net = Net(batch_size=batch_size)
# 优化器
opt = Adam(net.trainable_params(), learning_rate=0.0008,
           beta1=0.9, beta2=0.999, eps=10e-8, weight_decay=0.01)
# softmax损失函数
loss_fn = SoftmaxCrossEntropyWithLogits(sparse=True,reduction='mean')
# 利用训练数据集训练模型
model = Model(net, loss_fn, opt)  

## 模型训练

from mindspore.train import Callback

# 定义LossMonitor回调函数
class LossMonitor(Callback):
    def __init__(self):
        super(LossMonitor, self).__init__()

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        print("Step: {}, Loss: {}".format(cb_params.cur_step_num, cb_params.net_outputs.asnumpy()))

# 进行模型训练
model.train(10, ds_train, callbacks=[LossMonitor()])

# 保存模型
from mindspore import save_checkpoint
save_checkpoint(net, 'model.ckpt')

## 模型预测

net = Net(batch_size=1)

from mindspore import load_checkpoint
from mindspore import Tensor

# 读取训练的模型文件
ckpt_file_name = "./model.ckpt"
param_dict = load_checkpoint(ckpt_file_name, net)

print("****start test****")
# 获取测试文件
batch = get_data(test_file_path) 
print(batch)
# 初始化准确率
accu = 0 
size = val_num

# 根据训练好的模型进行预测
for i in range(size):
    input_x, label = next(batch)
    output = net(Tensor(input_x))
    index = np.argmax(output.asnumpy())
    # 输出期望值、预测值
    print(commands[index], commands[label]) 
    if index == label:
        # 若预测成功则成功数量+1，记录预测成功的样本数量
        accu += 1      
# 准确率
print("accuracy: ", accu*1.0 / size )

exit(0)
