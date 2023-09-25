
## 基于mindspore实现语义分割案例

# 导入依赖库
# os库
import os
# 引入numpy
import numpy as np
# 引入读写不同数据文件格式的函数
import scipy.io
# 引入数据序列化和反序列化
import pickle
# 引入操作图像方法
from PIL import Image
# 引入高级的文件,文件夹,压缩包处理模块
import shutil
# 引入计算机视觉库
import cv2
# 引入归一化提供训练测试所用的数据集
from mindspore.mindrecord import FileWriter
# 引入数据读取
import mindspore.dataset as de
# 引入MindSpore
import mindspore as ms
# 引入神经网络模块
import mindspore.nn as nn
#导入mindspore中的ops模块
import mindspore.ops  as P
# 引入张量模块
from mindspore import Tensor, ParallelMode
from mindspore import load_checkpoint
from mindspore import load_param_into_net
from mindspore import dtype as mstype
# 引入python解释器和它的环境有关的函数
import sys
# 将字典转为easydict
from easydict import EasyDict as edict
# 引入模型训练或推理的高阶接口。
# 引入用于构建Callback函数基类。
from mindspore.train import ModelCheckpoint, CheckpointConfig,LossMonitor, TimeMonitor,Model
# 引入集合通信接口
from mindspore.communication import init, get_rank, get_group_size
# 引入用于构建Callback函数的基类。
from mindspore import amp
from mindspore import set_seed
import PIL
# 引入绘图模块
import matplotlib.pyplot as plt
# 引入可视化库
import matplotlib as mpl
# 引入可视化库色彩模块
import matplotlib.colors as colors

batch_SZ = 2

## 数据预处理
# 设置Opencv的线程数量为0
cv2.setNumThreads(0)

# 数据集对象，用于载入语义分割数据集
class SegDataset:
    def __init__(self,
                 image_mean,            # 图像像素值平均值
                 image_std,             # 图像像素值标准差
                 data_file='',          # 数据集文件路径
                 batch_size=batch_SZ,   # 单次训练所使用样本的数量
                 crop_size=512,         # 随机裁剪后的图片大小
                 max_scale=2.0,         # 最大缩放比例
                 min_scale=0.5,         # 最小缩放比例
                 ignore_label=255,      # 忽略标签值
                 num_classes=21,        # 图像和标签中的类别数量
                 num_readers=2,         # 读取数据的IO线程数量
                 num_parallel_calls=1,  # 数据集batch的并行度
                 shard_id=None,         # 数据集分片ID，None表示无分片
                 shard_num=None         # 数据集分片数量，None表示无分片
                               ):
        # 定义数据集文件路径
        self.data_file = data_file
        # 定义单次训练所使用样本的数量
        self.batch_size = batch_size
        # 定义随机裁剪后的图片大小
        self.crop_size = crop_size
        # 定义图像像素值平均值
        self.image_mean = np.array(image_mean, dtype=np.float32)
        # 定义图像像素值标准差
        self.image_std = np.array(image_std, dtype=np.float32)
        # 定义最大缩放比例
        self.max_scale = max_scale
        # 定义最小缩放比例
        self.min_scale = min_scale
        # 定义忽略标签值
        self.ignore_label = ignore_label
        # 定义图像和标签中的类别数量
        self.num_classes = num_classes
        # 定义读取数据的IO线程数量
        self.num_readers = num_readers
        # 定义数据集batch的并行度
        self.num_parallel_calls = num_parallel_calls
        # 定义数据集分片ID
        self.shard_id = shard_id
        # 定义数据集分片数量
        self.shard_num = shard_num
        # VOC数据集原始图片文件夹路径
        self.voc_img_dir = os.path.join(self.data_file,'JPEG')
        # VOC数据集语义标注图片文件夹路径
        self.voc_anno_dir = os.path.join(self.data_file,'MASK1')
        # VOC数据集训练集文件列表路径
        self.voc_train_lst = os.path.join(self.data_file,'train.txt')
        # VOC数据集验证集文件列表路径
        self.voc_val_lst = os.path.join(self.data_file,'val.txt')
        #  VOC数据集使用的灰度标注图片文件夹路径
        self.voc_anno_gray_dir = os.path.join(self.data_file,'SegmentationClassGray')
        # 生成的MindRecord文件保存路径
        self.mindrecord_save =  os.path.join(self.data_file,'VOC_mindrecord')
        # 最大缩放比例必须大于最小缩放比例                               
        assert max_scale > min_scale

    #数据预处理，包括图像的解码，尺度缩放，随机裁剪等操作
    def preprocess_(self, image, label):
        #bgr图像解码
        image_out = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_COLOR)
        #灰度图像解码
        label_out = cv2.imdecode(np.frombuffer(label, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        #尺度缩放
        sc = np.random.uniform(self.min_scale, self.max_scale)
        new_h, new_w = int(sc * image_out.shape[0]), int(sc * image_out.shape[1])
        image_out = cv2.resize(image_out, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        label_out = cv2.resize(label_out, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        #图像标准化
        image_out = (image_out - self.image_mean) / self.image_std
        #随机裁剪
        h_, w_ = max(new_h, self.crop_size), max(new_w, self.crop_size)
        pad_h, pad_w = h_ - new_h, w_ - new_w
        if pad_h > 0 or pad_w > 0:
            image_out = cv2.copyMakeBorder(image_out, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
            label_out = cv2.copyMakeBorder(label_out, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=self.ignore_label)
        offset_h = np.random.randint(0, h_ - self.crop_size + 1)
        offset_w = np.random.randint(0, w_ - self.crop_size + 1)
        image_out = image_out[offset_h: offset_h + self.crop_size, offset_w: offset_w + self.crop_size, :]
        label_out = label_out[offset_h: offset_h + self.crop_size, offset_w: offset_w+self.crop_size]

        #随机水平翻转
        if np.random.uniform(0.0, 1.0) > 0.5:
            image_out = image_out[:, ::-1, :]
            label_out = label_out[:, ::-1]

        #图像转置以符合模型输入要求
        image_out = image_out.transpose((2, 0, 1))
        image_out = image_out.copy()
        label_out = label_out.copy()
        return image_out, label_out

    #得到灰度数据集的方法，若文件已存在，则直接返回
    def get_gray_dataset(self):
        if os.path.exists(self.voc_anno_gray_dir):
            print('the gray file is already exists！')
            return
        os.makedirs(self.voc_anno_gray_dir)

        #转换彩色图像为灰度图像，并保存到指定路径
        print('converting voc color png to gray png ...')
        for ann in os.listdir(self.voc_anno_dir):
            ann_im = Image.open(os.path.join(self.voc_anno_dir, ann))
            #将该图像转换为灰度图像
            ann_im = Image.fromarray(np.array(ann_im))
            ann_im.save(os.path.join(self.voc_anno_gray_dir, ann))
        print('converting done')

    #获取MindRecord格式的数据集，num_shards为生成MindRecord的分片数，shuffle为是否对数据做洗牌处理    
    def get_mindrecord_dataset(self, is_training, num_shards=1, shuffle=True):

        print('self.voc_train_lst: ', self.voc_train_lst)
        print(self.mindrecord_save) # VOC_mindrecord
        datas = []
        if is_training:
            data_lst = self.voc_train_lst
            self.mindrecord_save = os.path.join(self.mindrecord_save,'train')
        else:
            data_lst = self.voc_val_lst
            self.mindrecord_save = os.path.join(self.mindrecord_save,'eval')
        
        if os.path.exists(self.mindrecord_save):
            #shutil.rmtree(self.mindrecord_save)
            print('mindrecord file is already exists！')
            self.mindrecord_save = os.path.join(self.mindrecord_save,'VOC_mindrecord')
            return
        
        with open(data_lst) as f:
            lines = f.readlines()
        if shuffle:
            np.random.shuffle(lines)
            
        print('creating mindrecord dataset...')
        os.makedirs(self.mindrecord_save)
        self.mindrecord_save = os.path.join(self.mindrecord_save,'VOC_mindrecord')
        print('number of samples:', len(lines))
        #定义MindRecord的schema
        seg_schema = {"file_name": {"type": "string"}, "label": {"type": "bytes"}, "data": {"type": "bytes"}}
        writer = FileWriter(file_name=self.mindrecord_save, shard_num=num_shards)
        writer.add_schema(seg_schema, "seg_schema")

        #将schema写入MindRecord
        cnt = 0
        for l in lines:
            id_ = l.strip()
            img_path = os.path.join(self.voc_img_dir, id_ + '.jpg')
            label_path = os.path.join(self.voc_anno_gray_dir, id_ + '.png')
            
            sample_ = {"file_name": img_path.split('/')[-1]}
            with open(img_path, 'rb') as f:
                sample_['data'] = f.read()
            with open(label_path, 'rb') as f:
                sample_['label'] = f.read()
            datas.append(sample_)
            cnt += 1
            if cnt % 1000 == 0:
                writer.write_raw_data(datas)
                print('number of samples written:', cnt)
                datas = []

        if datas:
            writer.write_raw_data(datas)
        writer.commit()
        print('number of samples written:', cnt)
        print('Create Mindrecord Done')

    #生成文件    
    def get_dataset(self, repeat=1):
        data_set = de.MindDataset(dataset_files=self.mindrecord_save, columns_list=["data", "label"],
                                  shuffle=True, num_parallel_workers=self.num_readers,
                                  num_shards=self.shard_num, shard_id=self.shard_id)
        transforms_list = self.preprocess_
        data_set = data_set.map(operations=transforms_list, input_columns=["data", "label"],
                                output_columns=["data", "label"],
                                num_parallel_workers=self.num_parallel_calls)
        data_set = data_set.shuffle(buffer_size=self.batch_size * 10)
        data_set = data_set.batch(self.batch_size, drop_remainder=True)
        data_set = data_set.repeat(repeat)
        return data_set

# 实验过程
## 模型构建

# 定义1x1卷积层
def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, weight_init='xavier_uniform')

# 定义3x3卷积层
def conv3x3(in_planes, out_planes, stride=1, dilation=1, padding=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, pad_mode='pad', padding=padding,
                     dilation=dilation, weight_init='xavier_uniform')

#定义Resnet主体网络
class Resnet(nn.Cell):
    def __init__(self, block, block_num, output_stride, use_batch_statistics=True):
        super(Resnet, self).__init__()
        self.inplanes = 64# 输入通道数
        # 第一层卷积层
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, pad_mode='pad', padding=3,
                               weight_init='xavier_uniform')
        self.bn1 = nn.BatchNorm2d(self.inplanes, use_batch_statistics=use_batch_statistics)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
        # 第一层残差块
        self.layer1 = self._make_layer(block, 64, block_num[0], use_batch_statistics=use_batch_statistics)
        # 第二层残差块
        self.layer2 = self._make_layer(block, 128, block_num[1], stride=2, use_batch_statistics=use_batch_statistics)
        # 根据输出步长选择第三、第四层残差块
        if output_stride == 16:
            self.layer3 = self._make_layer(block, 256, block_num[2], stride=2,
                                           use_batch_statistics=use_batch_statistics)
            self.layer4 = self._make_layer(block, 512, block_num[3], stride=1, base_dilation=2, grids=[1, 2, 4],
                                           use_batch_statistics=use_batch_statistics)
        elif output_stride == 8:
            self.layer3 = self._make_layer(block, 256, block_num[2], stride=1, base_dilation=2,
                                           use_batch_statistics=use_batch_statistics)
            self.layer4 = self._make_layer(block, 512, block_num[3], stride=1, base_dilation=4, grids=[1, 2, 4],
                                           use_batch_statistics=use_batch_statistics)
    # 构建残差块
    def _make_layer(self, block, planes, blocks, stride=1, base_dilation=1, grids=None, use_batch_statistics=True):
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.SequentialCell([
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, use_batch_statistics=use_batch_statistics)
            ])

        if grids is None:
            grids = [1] * blocks

        layers = [
            block(self.inplanes, planes, stride, downsample, dilation=base_dilation * grids[0],
                  use_batch_statistics=use_batch_statistics)
        ]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, dilation=base_dilation * grids[i],
                      use_batch_statistics=use_batch_statistics))

        return nn.SequentialCell(layers)
    # 前向推理
    def construct(self, x):
        out = self.conv1(x)     # 第一层卷积
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out) # 第一层池化
        out = self.layer1(out)  # 第一层残差块
        out = self.layer2(out)  # 第二层残差块
        out = self.layer3(out)  # 第三层残差块
        out = self.layer4(out)  # 第四层残差块

        return out

#构建Bottleneck，用于ResNeXt中构建残差块
class Bottleneck(nn.Cell):
    # 扩充率为4
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, use_batch_statistics=True):
        super(Bottleneck, self).__init__()
        # 第一个1x1卷积层
        self.conv1 = conv1x1(inplanes, planes)
        # 第一个BatchNorm层
        self.bn1 = nn.BatchNorm2d(planes, use_batch_statistics=use_batch_statistics)
        # 第二个3x3卷积层
        self.conv2 = conv3x3(planes, planes, stride, dilation, dilation)
        # 第二个BatchNorm层
        self.bn2 = nn.BatchNorm2d(planes, use_batch_statistics=use_batch_statistics)
        # 第三个1x1卷积层，维度扩充
        self.conv3 = conv1x1(planes, planes * self.expansion)
        # 第三个BatchNorm层
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, use_batch_statistics=use_batch_statistics)
        # Relu激活函数
        self.relu = nn.ReLU()
        # 下采样层，使维度匹配
        self.downsample = downsample
        # 张量相加操作
        self.add = P.Add()

    def construct(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        # 将identity和out相加
        out = self.add(out, identity)
        out = self.relu(out)
        return out

#构建ASPP，用于DeepLabV3+中的ASPP模块
class ASPP(nn.Cell):
    def __init__(self, atrous_rates, phase='train', in_channels=2048, num_classes=21,
                 use_batch_statistics=True):
        super(ASPP, self).__init__()
        # 训练或者测试阶段
        self.phase = phase
        # 输出通道数
        out_channels = 256
        # ASPP卷积层1
        self.aspp1 = ASPPConv(in_channels, out_channels, atrous_rates[0], use_batch_statistics=use_batch_statistics)
         # ASPP卷积层2
        self.aspp2 = ASPPConv(in_channels, out_channels, atrous_rates[1], use_batch_statistics=use_batch_statistics)
         # ASPP卷积层3
        self.aspp3 = ASPPConv(in_channels, out_channels, atrous_rates[2], use_batch_statistics=use_batch_statistics)
         # ASPP卷积层4
        self.aspp4 = ASPPConv(in_channels, out_channels, atrous_rates[3], use_batch_statistics=use_batch_statistics)
        # ASPP池化层
        self.aspp_pooling = ASPPPooling(in_channels, out_channels)
        # 输出通道数为(out_channels * (len(atrous_rates) + 1))
        self.conv1 = nn.Conv2d(out_channels * (len(atrous_rates) + 1), out_channels, kernel_size=1,
                               weight_init='xavier_uniform')
        # BatchNorm层
        self.bn1 = nn.BatchNorm2d(out_channels, use_batch_statistics=use_batch_statistics)
        # Relu激活函数
        self.relu = nn.ReLU()
        # 输出通道数为num_classes
        self.conv2 = nn.Conv2d(out_channels, num_classes, kernel_size=1, weight_init='xavier_uniform', has_bias=True)
        # 沿着通道维度拼接
        self.concat = P.Concat(axis=1)
        # 随机失活，防止过拟合
        self.drop = nn.Dropout(p=0.3)

    def construct(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.aspp_pooling(x)

        x = self.concat((x1, x2))
        x = self.concat((x, x3))
        x = self.concat((x, x4))
        x = self.concat((x, x5))

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # 如果是训练阶段，则进行随机失活
        if self.phase == 'train':
            x = self.drop(x)
        x = self.conv2(x)
        return x

##定义ASPPPooling类，用于定义ASPP池操作
class ASPPPooling(nn.Cell):
    #定义参数
    def __init__(self, in_channels, out_channels, use_batch_statistics=True):
        super(ASPPPooling, self).__init__()
        #定义卷积层操作
        self.conv = nn.SequentialCell([
            #1x1卷积操作
            nn.Conv2d(in_channels, out_channels, kernel_size=1, weight_init='xavier_uniform'),
            #批量归一化操作
            nn.BatchNorm2d(out_channels, use_batch_statistics=use_batch_statistics),
            #激活函数操作
            nn.ReLU()
        ])
        #定义shape操作
        self.shape = P.Shape()

    def construct(self, x):
        #获取输入x的大小
        size = self.shape(x)
        #先对输入x进行全局平均池化操作
        out = nn.AvgPool2d(size[2])(x)
        #再进行卷积、批量归一化、激活操作
        out = self.conv(out)
        #大小变化为输入x的大小
        out = P.ResizeNearestNeighbor((size[2], size[3]), True)(out)
        return out

#定义ASPPConv类，用于定义ASPP卷积操作
class ASPPConv(nn.Cell):
    #定义参数
    def __init__(self, in_channels, out_channels, atrous_rate=1, use_batch_statistics=True):
        super(ASPPConv, self).__init__()
        #根据不同的空洞卷积率定义不同的卷积操作
        if atrous_rate == 1:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, has_bias=False, weight_init='xavier_uniform')
        else:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, pad_mode='pad', padding=atrous_rate,
                             dilation=atrous_rate, weight_init='xavier_uniform')
        #批量归一化操作
        bn = nn.BatchNorm2d(out_channels, use_batch_statistics=use_batch_statistics)
        #激活函数操作
        relu = nn.ReLU()
        #定义卷积层操作
        self.aspp_conv = nn.SequentialCell([conv, bn, relu])

    def construct(self, x):
        #进行卷积、批量归一化、激活操作
        out = self.aspp_conv(x)
        return out

#定义DeepLabV3类，用于定义整个DeepLabV3网络
class DeepLabV3(nn.Cell):
    def __init__(self, phase='train', num_classes=21, output_stride=16, freeze_bn=False):
        super(DeepLabV3, self).__init__()
        #根据输入的参数freeze_bn来判断是否使用批量归一化操作
        use_batch_statistics = not freeze_bn
        #调用Resnet类来构建ResNet网络
        self.resnet = Resnet(Bottleneck, [3, 4, 23, 3], output_stride=output_stride,
                             use_batch_statistics=use_batch_statistics)
        #调用ASPP类来构建ASPP网络
        self.aspp = ASPP([1, 6, 12, 18], phase, 2048, num_classes,
                         use_batch_statistics=use_batch_statistics)
        #定义shape操作
        self.shape = P.Shape()

    def construct(self, x):
        #获取输入x的大小
        size = self.shape(x)
        #将输入x输入到ResNet网络中，得到输出
        out = self.resnet(x)
        #将ResNet的输出输入到ASPP网络中，得到ASPP的输出
        out = self.aspp(out)
        #将ASPP的输出进行大小变换，变成与输入x相同的大小
        out = P.ResizeBilinear((size[2], size[3]), True)(out)
        return out


#定义不同的学习率
#生成cosine学习率下降序列
def cosine_lr(base_lr, decay_steps, total_steps):
    for i in range(int(total_steps)):
        step_ = min(i, decay_steps)
        yield base_lr * 0.5 * (1 + np.cos(np.pi * step_ / decay_steps))

#生成polynomial学习率下降序列
def poly_lr(base_lr, decay_steps, total_steps, end_lr=0.0001, power=0.9):
    for i in range(int(total_steps)):
        step_ = min(i, decay_steps)
        yield (base_lr - end_lr) * ((1.0 - step_ / decay_steps) ** power) + end_lr

#生成exponential学习率下降序列
def exponential_lr(base_lr, decay_steps, decay_rate, total_steps, staircase=False):
    for i in range(total_steps):
        if staircase:
            power_ = i // decay_steps
        else:
            power_ = float(i) / decay_steps
        yield base_lr * (decay_rate ** power_)


#定义损失函数
class SoftmaxCrossEntropyLoss(nn.Cell):
    def __init__(self, num_cls=21, ignore_label=255):
        super(SoftmaxCrossEntropyLoss, self).__init__()
        #one-hot编码相关操作
        self.one_hot = P.OneHot(axis=-1)
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        #类型转换相关操作
        self.cast = P.Cast()
        #softmax交叉熵损失函数及相关操作
        self.ce = nn.SoftmaxCrossEntropyWithLogits()
        #类别数及忽略标签
        self.not_equal = P.NotEqual()
        self.num_cls = num_cls
        self.ignore_label = ignore_label
        #矩阵乘法及求和相关操作
        self.mul = P.Mul()
        self.sum = P.ReduceSum(False)
        self.div = P.RealDiv()
        #转置及形状变换相关操作
        self.transpose = P.Transpose()
        self.reshape = P.Reshape()
    #将标签转换为整形
    def construct(self, logits, labels):
        #将标签拉成一维并转换形状
        labels_int = self.cast(labels, mstype.int32)
        labels_int = self.reshape(labels_int, (-1,))
         #将logits转置并转换形状
        logits_ = self.transpose(logits, (0, 2, 3, 1))
        logits_ = self.reshape(logits_, (-1, self.num_cls))
         #生成权重
        weights = self.not_equal(labels_int, self.ignore_label)
        weights = self.cast(weights, mstype.float32)
        #生成one-hot标签
        one_hot_labels = self.one_hot(labels_int, self.num_cls, self.on_value, self.off_value)
        #计算softmax交叉熵损失
        loss = self.ce(logits_, one_hot_labels)
         #加权
        loss = self.mul(weights, loss)
        #求平均损失
        loss = self.div(self.sum(loss), self.sum(weights))
        return loss

# 模型训练
## 构建训练函数

# 设置随机种子
set_seed(1)
# 设置上下文
ms.set_context(mode=ms.GRAPH_MODE, save_graphs=False,
                    device_target="CPU")
# 建立训练网络
class BuildTrainNetwork(nn.Cell):
    def __init__(self, network, criterion):
        super(BuildTrainNetwork, self).__init__()
        self.network = network
        self.criterion = criterion

    def construct(self, input_data, label):
        output = self.network(input_data)
        net_loss = self.criterion(output, label)
        return net_loss

# 训练函数
def train(args):
    # 如果使用分布式训练，则初始化
    if args.is_distributed:
        init()
        args.rank = get_rank()
        args.group_size = get_group_size()

        parallel_mode = ParallelMode.DATA_PARALLEL
        ms.set_auto_parallel_context(parallel_mode=parallel_mode, gradients_mean=True, device_num=args.group_size)
    
    
    # 构建数据集
    dataset = SegDataset(image_mean=args.image_mean,
                                        image_std=args.image_std,
                                        data_file=args.data_file,
                                        batch_size=args.batch_size,
                                        crop_size=args.crop_size,
                                        max_scale=args.max_scale,
                                        min_scale=args.min_scale,
                                        ignore_label=args.ignore_label,
                                        num_classes=args.num_classes,
                                        num_readers=2,
                                        num_parallel_calls=4,
                                        shard_id=args.rank,
                                        shard_num=args.group_size)
    dataset.get_gray_dataset()
    dataset.get_mindrecord_dataset(is_training=True)
    dataset = dataset.get_dataset(repeat=1)
    

    # 构建相关网络
    if args.model == 'deeplab_v3_s16':
        network = DeepLabV3('train', args.num_classes, 16, args.freeze_bn)
    elif args.model == 'deeplab_v3_s8':
        network = DeepLabV3('train', args.num_classes, 8, args.freeze_bn)
    else:
        raise NotImplementedError('model [{:s}] not recognized'.format(args.model))

    # 构建损失函数
    loss_ = SoftmaxCrossEntropyLoss(args.num_classes, args.ignore_label)
    loss_.add_flags_recursive(fp32=True)
    train_net = BuildTrainNetwork(network, loss_)

    # l加载预训练模型
    if os.path.exists(args.ckpt_file):
        param_dict = ms.load_checkpoint(args.ckpt_file)
        ms.load_param_into_net(train_net, param_dict)

    # 优化器
    iters_per_epoch = dataset.get_dataset_size()
    total_train_steps = iters_per_epoch * args.train_epochs
    if args.lr_type == 'cos':
        lr_iter = cosine_lr(args.base_lr, total_train_steps, total_train_steps)
    elif args.lr_type == 'poly':
        lr_iter = poly_lr(args.base_lr, total_train_steps, total_train_steps, end_lr=0.0, power=0.9)
    elif args.lr_type == 'exp':
        lr_iter = exponential_lr(args.base_lr, args.lr_decay_step, args.lr_decay_rate,
                                                total_train_steps, staircase=True)
    else:
        raise ValueError('unknown learning rate type')
    opt = nn.Momentum(params=train_net.trainable_params(), learning_rate=lr_iter, momentum=0.9, weight_decay=0.0001,
                      loss_scale=args.loss_scale)

    # 损失梯度缩放
    manager_loss_scale = amp.FixedLossScaleManager(args.loss_scale, drop_overflow_update=False)
    model = Model(train_net, optimizer=opt, amp_level="O3", loss_scale_manager=manager_loss_scale)

    # 回调函数，用于保存 ckpts
    time_cb = TimeMonitor(data_size=iters_per_epoch)
    loss_cb = LossMonitor()
    cbs = [time_cb, loss_cb]

    if args.rank == 0:
        config_ck = CheckpointConfig(save_checkpoint_steps=iters_per_epoch,
                                     keep_checkpoint_max=args.keep_checkpoint_max)
        ckpoint_cb = ModelCheckpoint(prefix=args.model, directory=args.train_dir, config=config_ck)
        cbs.append(ckpoint_cb)
    # 开始训练
    model.train(args.train_epochs, dataset, callbacks=cbs,dataset_sink_mode=True)

## 模型训练

##设定相关参数并转为edict对象
cfg = edict({
    "batch_size": batch_SZ,
    "crop_size": 513,
    "image_mean": [103.53, 116.28, 123.675],    #图片均值
    "image_std": [57.375, 57.120, 58.395],      #图片标准差
    "min_scale": 0.5,                           #最小缩放比例
    "max_scale": 2.0,                           #最大缩放比例
    "ignore_label": 255,                        #忽略标签
    "num_classes": 21,                          #分类数
    "train_epochs" : 50,                         #训练轮数
    "lr_type": 'cos',                           #学习率变化方式
    "base_lr": 0.0,                             #基础学习率
    "lr_decay_step": 3*91,                      #学习率递减步数
    "lr_decay_rate" :0.1,                       #学习率递减率
    "loss_scale": 2048,                         #损失函数缩放比例
    "model": 'deeplab_v3_s8',                   #模型类型
    'rank': 0,                                  #排名
    'group_size':1,                             #组大小
    'keep_checkpoint_max':1,                    #最大保存点数
    'train_dir': 'model',                       #训练目录
    'is_distributed':False,                     #是否分布式训练
    'freeze_bn':True                            #是否冻结BN层
})

if __name__ == '__main__':
    device_target = ms.get_context("device_target")
    mode = ms.GRAPH_MODE
    print("device:", device_target)
    ms.context.set_context(mode=mode, device_target="CPU")

    #如果训练目录存在，则删除
    if os.path.exists(cfg.train_dir):
        shutil.rmtree(cfg.train_dir)

    #数据路径和checkpoint路径
    data_path = './seg2'

    cfg.data_file = data_path
    ckpt_path = './ckpt/deeplab_v3_s8-300_11.ckpt'
    cfg.ckpt_file = ckpt_path

    #开始训练模型
    train(cfg)
    exit(0)




