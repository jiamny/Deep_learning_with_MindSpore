
# 基于MindSpore实现图像分类示例


# 处理文件和目录
import os
# 该模块提供了加载和处理各种通用数据集的API
from mindspore import dataset as ds
# 此模块用于图像数据增强
import mindspore.dataset.vision as CV
# 图像插值方式枚举类
from mindspore.dataset.vision import Inter
# 此模块用于通用数据增强
import mindspore.dataset.transforms as C
# MindSpore数据类型的对象
from mindspore import dtype as mstype
# 绘图库
import matplotlib.pyplot as plt
# 数据下载
from mindspore.dataset import MnistDataset

from mindspore import context, set_seed
set_seed(0)
context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

'''
### 数据加载
from download import download

url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/" \
      "notebook/datasets/MNIST_Data.zip"
path = download(url, "./", kind="zip", replace=True)
'''

train_dataset = MnistDataset("/media/hhj/localssd/DL_data/mnist/MNIST_Data/train", shuffle=False)
test_dataset = MnistDataset("/media/hhj/localssd/DL_data/mnist/MNIST_Data/test", shuffle=False)
# 查看训练集数据
def visualize(dataset):
    figure = plt.figure(figsize=(6, 6))
    cols, rows = 3, 3

    for idx, (image, label) in enumerate(dataset.create_tuple_iterator()):
        figure.add_subplot(rows, cols, idx + 1)
        plt.title(int(label))
        plt.axis("off")
        plt.imshow(image.asnumpy().squeeze(), cmap="gray")
        if idx == cols * rows - 1:
            break
    plt.show()

visualize(train_dataset)
def create_dataset(mnist_ds, batch_size=32, repeat_size=1,
                   num_parallel_workers=1):
    # 为训练和测试生成数据集
    # 参数:
    # mnist_ds: 数据集
    # batch_size: 指定每个批处理数据包含的数据条目。
    # repeat_size: 数据集重复次数
    # num_parallel_workers: 指定map操作的多进程/多线程并发数，加快处理速度
    # 定义数据集
    # 生成的数据集有两列: [image, label]。 image 列的数据类型为uint8。 label 列的数据类型为uint32。
    # 定义操作参数
    # 调整后图片的尺寸
    resize_height, resize_width = 32, 32  
    # 缩放因子
    rescale = 1.0 / 255.0   
    # 平移因子
    shift = 0.0   
    # 标准化图像缩放因子
    rescale_nml = 1 / 0.3081   
    # 标准化图像平移因子
    shift_nml = -1 * 0.1307 / 0.3081         

    # 定义映射操作
    # 调整图片尺寸为 (32, 32)，interpolation线性插值
    resize_op = CV.Resize((resize_height, resize_width), interpolation=Inter.LINEAR)
    # 标准化图片
    rescale_nml_op = CV.Rescale(rescale_nml, shift_nml) 
    # 重新缩放图像
    rescale_op = CV.Rescale(rescale, shift) 
    # 将输入图像的shape从 <H, W, C> 转换为 <C, H, W>。 如果输入图像的shape为 <H, W> ，图像将保持不变。
    hwc2chw_op = CV.HWC2CHW() 
    # 将输入的Tensor转换为指定的数据类型。
    type_cast_op = C.TypeCast(mstype.int32) 

    # 将定义好的映射操作依次应用在数据集上。
    # 将输入的Tensor转换为指定的mstype.int32类型，输入数据为“label”列，num_parallel_workers指定线程并发数，此处为单线程处理。
    mnist_ds = mnist_ds.map(operations=type_cast_op, input_columns="label", num_parallel_workers=num_parallel_workers)
    # 调整图片的尺寸为（32，32），输入数据为“image”列
    mnist_ds = mnist_ds.map(operations=resize_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    # 缩放图像，输入数据为“image”列，输出图像的像素大小为：output = image * rescale + shift。
    mnist_ds = mnist_ds.map(operations=rescale_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    # 标准化图像，输入数据为“image”列，输出图像的像素大小为：output = image * rescale_nml + shift_nml。
    mnist_ds = mnist_ds.map(operations=rescale_nml_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    # 将输入图像的shape从 <High, Wide, Channel> 转换为 <Channel, High, Wide>。 如果输入图像的shape为 <High, Wide> ，图像将保持不变。
    mnist_ds = mnist_ds.map(operations=hwc2chw_op, input_columns="image", num_parallel_workers=num_parallel_workers)

    # 应用数据集操作
    # 混洗缓冲区大小
    buffer_size = 10000
    # 混洗数据集
    mnist_ds = mnist_ds.shuffle(buffer_size=buffer_size) 
    # batch操作将数据集中连续 batch_size 条数据组合为一个批数据，并可通过可选参数 per_batch_map 指定组合前要进行的预处理操作。
    # drop_remainder，当最后一个批处理数据包含的数据条目小于 batch_size 时，是否将该批处理丢弃，不传递给下一个操作。默认值：False，不丢弃。此处为True，丢弃。
    mnist_ds = mnist_ds.batch(batch_size, drop_remainder=True)
    # 重复此数据集 repeat_size 次。如果 repeat_size 为None或-1，则无限重复迭代。
    mnist_ds = mnist_ds.repeat(repeat_size)
    
    # 返回处理好的数据集
    return mnist_ds
# 查看训练数据集
ds_train = create_dataset(train_dataset, 32, 1)

# 下载测试数据集
ds_eval = create_dataset(test_dataset)
print("dataset handled over")

# 构建模型

# 神经网络Cell，用于构建神经网络中的预定义构建块或计算单元。
import mindspore.nn as nn
import mindspore as ms
# Model建立模型
# load_checkpoint加载checkpoint文件
# load_param_into_net将参数加载到网络中，返回网络中没有被加载的参数列表。
from mindspore import load_checkpoint, load_param_into_net
# 生成一个服从正态分布 N(sigma,mean) 的随机数组用于初始化Tensor
from mindspore.common.initializer import Normal
# ModelCheckpointcheckpoint的回调函数，在训练过程中调用该方法可以保存网络参数。
# CheckpointConfig保存checkpoint时的配置策略。
# LossMonitor训练场景下，监控训练的loss；边训练边推理场景下，监控训练的loss和推理的metrics。
# Accuracy计算数据分类的正确率，包括二分类和多分类。
from mindspore.train import Model, ModelCheckpoint, CheckpointConfig, LossMonitor, Accuracy
# from mindspore.nn.metrics import Accuracy
# 计算预测值与真实值之间的交叉熵。
from mindspore.nn import SoftmaxCrossEntropyWithLogits
class LeNet5(nn.Cell):
    # Lenet 网络结构
    # 定义需要的操作
    def __init__(self, num_class=10, num_channel=1):
        super(LeNet5, self).__init__()
        # 对输入Tensor计算二维卷积，输入空间维度in_channels为num_channel，输出空间维度out_channels为6，kernel_size为5
        #pad_mode (str) - 指定填充模式。可选值为”same”、”valid”、”pad”。默认值：”same”。
        #same：输出的高度和宽度分别与输入整除 stride 后的值相同。若设置该模式，padding 的值必须为0。
        #valid：在不填充的前提下返回有效计算所得的输出。不满足计算的多余像素会被丢弃。如果设置此模式，则 padding 的值必须为0。
        #pad：对输入进行填充。在输入的高度和宽度方向上填充 padding 大小的0。如果设置此模式， padding 必须大于或等于0。
        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        # 对输入Tensor计算二维卷积，in_channels=6，out_channels=16，kernel_size=5
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        # 全连接层，适用于输入的密集连接层。
        # Dense层输入Tensor的空间维度in_channels=16 * 5 * 5，Dense层输出Tensor的空间维度out_channels=120，
        # weight_init权重参数初始化，Normal(0.02)生成一个正太分布的随机数
        self.fc1 = nn.Dense(16 * 5 * 5, 120, weight_init=Normal(0.02))
        # Dense层in_channels=120，out_channels=84。
        self.fc2 = nn.Dense(120, 84, weight_init=Normal(0.02))
        # Dense层in_channels=84，out_channels=num_class，此处为10。
        self.fc3 = nn.Dense(84, num_class, weight_init=Normal(0.02))
        # 定义激活函数为ReLU
        self.relu = nn.ReLU()
        # 定义最大二维池化，卷积核尺寸为2，步长为2
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        # 对输入Tensor的第零维之外的维度进行展平操作。
        self.flatten = nn.Flatten()

    # 网络构造函数，使用上面定义的操作来构建网络
    def construct(self, x):
        # 对输入数据进行self.conv1卷积操作后，self.relu激活，再self.max_pool2d最大2维池化
        x = self.max_pool2d(self.relu(self.conv1(x)))
        # 对输入数据进行self.conv2卷积操作后，self.relu激活，再self.max_pool2d最大2维池化
        x = self.max_pool2d(self.relu(self.conv2(x)))
        # 对输入数据进行展平操作
        x = self.flatten(x)
        # 输入数据经过self.fc1全链接层处理后，self.relu函数激活
        x = self.relu(self.fc1(x))
        # 输入数据经过self.fc2全链接层处理后，self.relu函数激活
        x = self.relu(self.fc2(x))
        # 对输入数据进行self.fc3全链接层处理
        x = self.fc3(x)
        return x
# 创建网络模型
net = LeNet5()

# 训练模型

# 设置运行环境的context。
ms.set_context(mode=0, device_target="CPU")
dataset_sink_mode = False
# 设置学习率lr
lr = 0.01
# 设置移动平均的动量
momentum = 0.9
# 设置数据集重复次数
dataset_size = 1
# 数据下载路径
mnist_path = "./MNIST_Data"
# 定义损失函数，计算预测值与真实值之间的交叉熵。
# sparse (bool) - 指定目标值是否使用稀疏格式。默认值：False。
# reduction (str) - 指定应用于输出结果的计算方式。取值为”mean”，”sum”，或”none”。取值为”none”，则不执行reduction。默认值：”none”。
net_loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
# 定义训练执行轮次
train_epoch = 1

# 定义优化器
net_opt = nn.Momentum(net.trainable_params(), lr, momentum)
# 保存checkpoint时的配置策略，
# save_checkpoint_steps (int) - 每隔多少个step保存一次checkpoint。默认值：1。
# keep_checkpoint_max (int) - 最多保存多少个checkpoint文件。默认值：5。
config_ck = CheckpointConfig(save_checkpoint_steps=1875, keep_checkpoint_max=10)
# 保存网络模型和参数以进行子序列微调
# prefix (str) - checkpoint文件的前缀名称。默认值：’CKP’
# config (CheckpointConfig) - checkpoint策略配置。默认值：None。
ckpoint = ModelCheckpoint(prefix="checkpoint_lenet", config=config_ck)
# 将层分组到具有训练和评估特征的对象中，输出参数指定输出精确度
model = Model(net, net_loss, net_opt, metrics={"Accuracy": Accuracy()})
# 定义训练方法
def train_net(ds_train, network_model, epoch_size, data_path, repeat_size, ckpoint_cb, sink_mode):
    print("============== Starting Training ==============")
    # 训练执行轮次epoch_size，训练数据集ds_train
    # callbacks (Optional[list[Callback], Callback]) - 训练过程中需要执行的回调对象或者回调对象列表。默认值：None。
    # dataset_sink_mode (bool) - 数据是否直接下沉至处理器进行处理。使用PYNATIVE_MODE模式或CPU处理器时，模型训练流程将以非下沉模式执行。默认值：False。
    network_model.train(epoch_size, ds_train, callbacks=[ckpoint_cb, LossMonitor(per_print_times=75)], dataset_sink_mode=sink_mode)
#调用训练方法
train_net(ds_train, model, train_epoch, mnist_path, dataset_size, ckpoint, dataset_sink_mode)

# 模型预测

def tst_net(ds_eval, network, network_model, data_path):
    # 定义测试方法
    print("============== Starting Testing ==============")
    # 下载并保存模型用以评估
    param_dict = load_checkpoint("checkpoint_lenet-1_1875.ckpt")
    # 将参数加载到网络中，返回网络中没有被加载的参数列表
    load_param_into_net(network, param_dict)
    # 获取测试结果的精确度并输出。
    acc = network_model.eval(ds_eval, dataset_sink_mode=False)
    print("============== Accuracy:{} ==============".format(acc))
# 调用测试方法
tst_net(ds_eval, net, model, mnist_path)
exit(0)