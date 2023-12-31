
# 基于MindSpore实现目标检测示例
from mindspore import context, set_seed
set_seed(0)
context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

### 设置超参数

# YOLOv3模型Config 参数
batch_size = 4

class ConfigYOLOV3ResNet18:
    # YOLOv3模型Config 参数
    #
    # 示例:
    #     ConfigYoloV3ResNet18.

    # 输入图像的形状
    img_shape = [352, 640]
    # 特征图的形状
    feature_shape = [batch_size, 3, 352, 640]
    # 目标分类的数量
    num_classes = 2
    # 非极大值抑制（NMS）后最多的检测框数量
    nms_max_num = 100
    # 检测框的最大数量
    _NUM_BOXES = 100

    # 网络骨干的输入形状
    backbone_input_shape = [64, 64, 128, 256]
    # 网络骨干的输出形状
    backbone_shape = [64, 128, 256, 512]
    # 网络骨干中各个层的数量
    backbone_layers = [2, 2, 2, 2]
    # 网络骨干中各个层的步幅
    backbone_stride = [1, 2, 2, 2]

    # 目标预测框与真实框之间的IoU阈值，低于此阈值将被视为背景
    ignore_threshold = 0.5
    # 预测目标存在的置信度阈值，低于此阈值将被视为背景
    obj_threshold = 0.3
    # NMS的IoU阈值，用于合并重叠的检测框
    nms_threshold = 0.4

    # 锚框的大小
    anchor_scales = [(5,3),(10, 13), (16, 30),(33, 23),(30, 61),(62, 45),(59, 119),(116, 90),(156, 198)]
    # 输出通道数
    out_channel = int(len(anchor_scales) / 3 * (num_classes + 5))

### 数据加载
# YOLOv3 dataset

# matplotlib.colors.rgb_to_hsv()函数用于将0到1范围内的float rgb转换为hsv值的numpy数组
# matplotlib.colors.hsv_to_rgb()函数用于将hsv值转换为rgb
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
# Python图像处理库中，Image是其中的核心库
from PIL import Image
# 提供加载和处理各种通用数据集的API
from mindspore import dataset

# 用于图像数据增强
from mindspore.dataset import vision


def preprocess_fn(image, box, file, is_training):
    # 数据集预处理函数
    config_anchors = []
    temp = ConfigYOLOV3ResNet18.anchor_scales
    for i in temp:
        config_anchors+=list(i)
    
    anchors = np.array([float(x) for x in config_anchors]).reshape(-1, 2)
    do_hsv = False
    max_boxes = ConfigYOLOV3ResNet18._NUM_BOXES
    num_classes = ConfigYOLOV3ResNet18.num_classes

    def _rand(a=0., b=1.):
        return np.random.rand() * (b - a) + a

    # 得到真实Box
    def _preprocess_true_boxes(true_boxes, anchors, in_shape=None):
        # 将真实的bounding box转化为网络输出所需要的格式
        # 特征图数量
        num_layers = anchors.shape[0] // 3
        # 三个特征图所用的anchor的索引
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        true_boxes = np.array(true_boxes, dtype='float32')
        input_shape = np.array(in_shape, dtype='int32')
        # 真实框的中心点坐标
        boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2.
        # 真实框的宽和高
        boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
        # 归一化后的真实框中心坐标
        true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
        # 归一化后的真实框宽高
        true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]
        # 三个特征图的大小
        grid_shapes = [input_shape // 32, input_shape // 16, input_shape // 8]
        # 初始化label列表
        y_true = [np.zeros((grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]),
                            5 + num_classes), dtype='float32') for l in range(num_layers)]
        # 将三维anchors转为四维
        anchors = np.expand_dims(anchors, 0)
        # anchor的最大值
        anchors_max = anchors / 2.
        # anchor的最小值
        anchors_min = -anchors_max
        # 宽>=1的真实框是有效框
        valid_mask = boxes_wh[..., 0] >= 1
        # 有效框的宽高
        wh = boxes_wh[valid_mask]

        if len(wh) >= 1:# 如果有有效框
            # 给wh添加一个维度
            wh = np.expand_dims(wh, -2)
            # 计算box最大值
            boxes_max = wh / 2.
            # 计算box最小值
            boxes_min = -boxes_max
            # 计算交集最小值
            intersect_min = np.maximum(boxes_min, anchors_min)
            # 计算交集最大值
            intersect_max = np.minimum(boxes_max, anchors_max)
            # 计算交集宽高
            intersect_wh = np.maximum(intersect_max - intersect_min, 0.)
            # 计算交集面积
            intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
            # 计算box面积
            box_area = wh[..., 0] * wh[..., 1]
            # 计算anchors面积
            anchor_area = anchors[..., 0] * anchors[..., 1]
            # 计算交并比
            iou = intersect_area / (box_area + anchor_area - intersect_area)
            # 找到最佳anchors索引
            best_anchor = np.argmax(iou, axis=-1)
            for t, n in enumerate(best_anchor):# 遍历最佳anchors索引
                for l in range(num_layers):# 遍历层数
                    if n in anchor_mask[l]:# 如果该索引在该层的anchor_mask中出现
                        # 计算第i列的坐标
                        i = np.floor(true_boxes[t, 0] * grid_shapes[l][1]).astype('int32')
                        # 计算第j行的坐标
                        j = np.floor(true_boxes[t, 1] * grid_shapes[l][0]).astype('int32')
                        # 找到对应的anchor索引
                        k = anchor_mask[l].index(n)

                        # 获取真实框的类别
                        c = true_boxes[t, 4].astype('int32')
                        # 存储真实框的坐标
                        y_true[l][j, i, k, 0:4] = true_boxes[t, 0:4]
                        # 标记该点有目标
                        y_true[l][j, i, k, 4] = 1.
                        # 存储真实框的类别
                        y_true[l][j, i, k, 5 + c] = 1.
        # 分别初始化一个全0的数组
        pad_gt_box0 = np.zeros(shape=[ConfigYOLOV3ResNet18._NUM_BOXES, 4], dtype=np.float32)
        pad_gt_box1 = np.zeros(shape=[ConfigYOLOV3ResNet18._NUM_BOXES, 4], dtype=np.float32)
        pad_gt_box2 = np.zeros(shape=[ConfigYOLOV3ResNet18._NUM_BOXES, 4], dtype=np.float32)

        # 获取y_true[0]的box
        mask0 = np.reshape(y_true[0][..., 4:5], [-1])
        gt_box0 = np.reshape(y_true[0][..., 0:4], [-1, 4])
        gt_box0 = gt_box0[mask0 == 1]
        pad_gt_box0[:gt_box0.shape[0]] = gt_box0

        # 获取y_true[1]的box
        mask1 = np.reshape(y_true[1][..., 4:5], [-1])
        gt_box1 = np.reshape(y_true[1][..., 0:4], [-1, 4])
        gt_box1 = gt_box1[mask1 == 1]
        pad_gt_box1[:gt_box1.shape[0]] = gt_box1

        # 获取y_true[2]的box
        mask2 = np.reshape(y_true[2][..., 4:5], [-1])
        gt_box2 = np.reshape(y_true[2][..., 0:4], [-1, 4])
        gt_box2 = gt_box2[mask2 == 1]
        pad_gt_box2[:gt_box2.shape[0]] = gt_box2

        return y_true[0], y_true[1], y_true[2], pad_gt_box0, pad_gt_box1, pad_gt_box2

    # 数据增强
    def _infer_data(img_data, input_shape, box):
        # 获取图片尺寸和目标尺寸
        w, h = img_data.size
        input_h, input_w = input_shape
        # 计算图片的缩放比例
        scale = min(float(input_w) / float(w), float(input_h) / float(h))
        nw = int(w * scale)
        nh = int(h * scale)
        # 对图片进行缩放
        img_data = img_data.resize((nw, nh), Image.Resampling.BICUBIC)

        # 创建一张新的图片，并进行填充
        new_image = np.zeros((input_h, input_w, 3), np.float32)
        new_image.fill(128)
        # 将原图片转为np数组
        img_data = np.array(img_data)
        # 如果是单通道的灰度图片，将其转为三通道图片
        if len(img_data.shape) == 2:
            img_data = np.expand_dims(img_data, axis=-1)
            img_data = np.concatenate([img_data, img_data, img_data], axis=-1)

        # 计算填充后图片的高度和宽度
        dh = int((input_h - nh) / 2)
        dw = int((input_w - nw) / 2)
        # 将原图片拷贝到填充后的图片的中心位置
        new_image[dh:(nh + dh), dw:(nw + dw), :] = img_data
        # 像素值归一化
        new_image /= 255.
        # 调整维度顺序
        new_image = np.transpose(new_image, (2, 0, 1))
        # 扩展维度
        new_image = np.expand_dims(new_image, 0)
        # 返回增强后的图片、原图尺寸以及目标检测框
        return new_image, np.array([h, w], np.float32), box

    # 数据增强
    def _data_aug(image, box, is_training, jitter=0.3, hue=0.1, sat=1.5, val=1.5, image_size=(352, 640)):

        # 若image非Image.Image则将其从array转换为Image.Image
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        iw, ih = image.size
        ori_image_shape = np.array([ih, iw], np.int32)
        h, w = image_size

        if not is_training:
            return _infer_data(image, image_size, box)

        # 以0.5的概率翻转图像
        flip = _rand() < .5
        # 正确的Box
        box_data = np.zeros((max_boxes, 5))
        flag =0
        
        while True:
            # 防止所有Box都被排除，new_ar为新宽高比系数
            # 生成一个介于(1-jitter)和(1+jitter)之间的随机浮点数，再与图像的宽高比相乘得到新的宽高比
            new_ar = float(w) / float(h) * _rand(1 - jitter, 1 + jitter) / \
                     _rand(1 - jitter, 1 + jitter)
            # 生成一个介于0.25和2之间的随机浮点数，作为缩放比例
            scale = _rand(0.25, 2)

            # 根据new_ar计算图像的新高度和新宽度
            if new_ar < 1:
                nh = int(scale * h)
                nw = int(nh * new_ar)
            else:
                nw = int(scale * w)
                nh = int(nw / new_ar)

            # 随机选择图片的位置
            dx = int(_rand(0, w - nw))
            dy = int(_rand(0, h - nh))
            flag = flag + 1
            
            if len(box) >= 1:
                # 随机扰动Box的位置
                # 复制原始的box
                t_box = box.copy()
                # 打乱box中的行
                np.random.shuffle(t_box)
                # 根据缩放和平移的值，调整box中的坐标
                t_box[:, [0, 2]] = t_box[:, [0, 2]] * float(nw) / float(iw) + dx
                t_box[:, [1, 3]] = t_box[:, [1, 3]] * float(nh) / float(ih) + dy
                # 如果图像被翻转了，则需要调整box中的坐标
                if flip:
                    t_box[:, [0, 2]] = w - t_box[:, [2, 0]]
                # 确保box中的坐标不会超出图像的边界
                t_box[:, 0:2][t_box[:, 0:2] < 0] = 0
                t_box[:, 2][t_box[:, 2] > w] = w
                t_box[:, 3][t_box[:, 3] > h] = h
                # 根据box的宽高，舍弃不合法的box
                box_w = t_box[:, 2] - t_box[:, 0]
                box_h = t_box[:, 3] - t_box[:, 1]
                t_box = t_box[np.logical_and(box_w > 1, box_h > 1)]
            # 如果还有合法的box，则返回t_box，否则继续循环生成新的图像和box
            if len(t_box) >= 1:
                box = t_box
                break

        # 将扭曲后有效的边界框信息存入 box_data
        box_data[:len(box)] = box
        # 调整尺寸image
        image = image.resize((nw, nh), Image.Resampling.BICUBIC)
        # 放置图像到新图像中，并使之在新图像上的位置发生随机位移
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        # 判断是否需要翻转图像
        if flip:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        # 判断是否需要将图像转为灰度图
        gray = _rand() < .25
        if gray:
            image = image.convert('L').convert('RGB')

        # 将图像转为 numpy 数组
        image = np.array(image)
        # 当通道数为 1 时，将其复制 3 份，变为 RGB 图像
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
            image = np.concatenate([image, image, image], axis=-1)

        # 对图像进行色彩扭曲，包括色调、饱和度和亮度的变化
        hue = _rand(-hue, hue)
        sat = _rand(1, sat) if _rand() < .5 else 1 / _rand(1, sat)
        val = _rand(1, val) if _rand() < .5 else 1 / _rand(1, val)
        image_data = image / 255.
        if do_hsv:
            x = rgb_to_hsv(image_data)
            x[..., 0] += hue
            x[..., 0][x[..., 0] > 1] -= 1
            x[..., 0][x[..., 0] < 0] += 1
            x[..., 1] *= sat
            x[..., 2] *= val
            x[x > 1] = 1
            x[x < 0] = 0
            # numpy array, 从 0 到 1
            image_data = hsv_to_rgb(x)
        image_data = image_data.astype(np.float32)

        # 预处理边界框信息
        bbox_true_1, bbox_true_2, bbox_true_3, gt_box1, gt_box2, gt_box3 = \
            _preprocess_true_boxes(box_data, anchors, image_size)

        # 返回扭曲后的图像数据、边界框信息、原始图像形状和预处理的边界框信息
        return image_data, bbox_true_1, bbox_true_2, bbox_true_3, \
               ori_image_shape, gt_box1, gt_box2, gt_box3

    # 判断是否是训练集
    if is_training:
        images, bbox_1, bbox_2, bbox_3, image_shape, gt_box1, gt_box2, gt_box3 = _data_aug(image, box, is_training)
        return images, bbox_1, bbox_2, bbox_3, gt_box1, gt_box2, gt_box3

    images, shape, anno = _data_aug(image, box, is_training)
    return images, shape, anno, file

# 通过MindDataset创建YOLOv3 dataset
def create_yolo_dataset(mindrecord_dir, batch_size=32, repeat_num=1, device_num=1, rank=0,
                        is_training=True, num_parallel_workers=8):
    # 从MindRecord文件中读取数据，并对数据进行操作
    ds = dataset.MindDataset(mindrecord_dir, columns_list=["image", "annotation","file"], num_shards=device_num, shard_id=rank,
                             num_parallel_workers=num_parallel_workers, shuffle=is_training)
    # 解码图片数据
    decode = vision.Decode()
    ds = ds.map(operations=decode, input_columns=["image"])
    # 定义数据预处理函数，并将其和解码后的图片数据一起作为参数传入ds.map()函数中，进行并行处理
    compose_map_func = (lambda image, annotation, file: preprocess_fn(image, annotation,file, is_training))

    if is_training:
        # 图像由HWC格式转为CHW格式
        hwc_to_chw = vision.HWC2CHW()
        # 为训练数据定义数据预处理函数，包括图像增强和标注处理，并将其和训练数据一起作为参数传入ds.map()函数中，进行并行处理
        ds = ds.map(operations=compose_map_func, input_columns=["image", "annotation","file"],
                    output_columns=["image", "bbox_1", "bbox_2", "bbox_3", "gt_box1", "gt_box2", "gt_box3"],
                    num_parallel_workers=num_parallel_workers)
        # 只保留输出数据中的部分列
        ds = ds.project(["image", "bbox_1", "bbox_2", "bbox_3", "gt_box1", "gt_box2", "gt_box3"])
        ds = ds.map(operations=hwc_to_chw, input_columns=["image"], num_parallel_workers=num_parallel_workers)
        # 批处理数据，每批次的大小为batch_size，去除不足一批次的数据，并将其重复repeat_num次
        ds = ds.batch(batch_size, drop_remainder=True)
        ds = ds.repeat(repeat_num)
    else:
        # 为测试数据定义数据预处理函数，并将其和测试数据一起作为参数传入ds.map()函数中，进行并行处理
        ds = ds.map(operations=compose_map_func, input_columns=["image", "annotation","file"],
                    output_columns=["image", "image_shape", "annotation","file"],
                    num_parallel_workers=num_parallel_workers)
        # 只保留输出数据中的部分列
        ds = ds.project(["image", "image_shape", "annotation","file"])
    return ds

## 模型构建
### 定义网络

# YOLOv3 based on ResNet18.
# 神经网络Cell。用于构建神经网络中的预定义构建块或计算单元。
from mindspore import nn
# 提供各种算子
from mindspore import ops

# 权重变量
def weight_variable():
    # 生成一个服从截断正态（高斯）分布的随机数组用于初始化Tensor
    return ms.common.initializer.TruncatedNormal(0.02)


# 带 padding的Conv2d
class _conv2d(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(_conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride, padding=0, pad_mode='same',
                              weight_init=weight_variable())
    def construct(self, x):
        x = self.conv(x)
        return x


# 得到融合的batchnorm
def _fused_bn(channels, momentum=0.99):
    return nn.BatchNorm2d(channels, momentum=momentum)


# conv2d batchnorm 和 relu 层
def _conv_bn_relu(in_channel,
                  out_channel,
                  ksize,
                  stride=1,
                  padding=0,
                  dilation=1,
                  alpha=0.1,
                  momentum=0.99,
                  pad_mode="same"):
    return nn.SequentialCell(
        [nn.Conv2d(in_channel,
                   out_channel,
                   kernel_size=ksize,
                   stride=stride,
                   padding=padding,
                   dilation=dilation,
                   pad_mode=pad_mode),
         nn.BatchNorm2d(out_channel, momentum=momentum),
         nn.LeakyReLU(alpha)]
    )


class BasicBlock(nn.Cell):
    # ResNet 基本块
    #
    # 参数：
    #     in_channels (int): 输入通道
    #     out_channels (int): 输出通道
    #     stride (int): 初始卷积层的跨步，默认为1
    #     momentum (float): batchnorm层的momentum 默认为0.1
    #
    # 返回：
    #     Tensor, output tensor
    #
    # 示例：
    #     BasicBlock(3,256,stride=2,down_sample=True)
    expansion = 1

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 momentum=0.99):
        super(BasicBlock, self).__init__()

        # x->conv->bn->relu->conv->bn-->Add(x,identity)->result
        # identity = x 若输入通道数与输出通道数不同，则还需要通过1*1的conv进行downsample
        self.conv1 = _conv2d(in_channels, out_channels, 3, stride=stride)
        self.bn1 = _fused_bn(out_channels, momentum=momentum)
        self.conv2 = _conv2d(out_channels, out_channels, 3)
        self.bn2 = _fused_bn(out_channels, momentum=momentum)
        self.relu = nn.ReLU()
        self.down_sample_layer = None
        # 若输入通道数与输出通道数不同则需要downsample
        self.downsample = (in_channels != out_channels)
        # 使用1*1的conv进行下采样
        if self.downsample:
            self.down_sample_layer = _conv2d(in_channels, out_channels, 1, stride=stride)

    def construct(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample:
            identity = self.down_sample_layer(identity)

        out = ops.add(x, identity)
        out = self.relu(out)

        return out


class ResNet(nn.Cell):
    # ResNet network
    #
    # 参数：
    #     block (Cell): 网络块
    #     layer_nums (list): 不同层数目
    #     in_channels (int): 输入通道
    #     out_channels (int): 输出通道
    #     num_classes (int): 类数目 默认：100
    #
    # 返回：
    #     Tensor, output tensor.
    #
    # 示例：
    #     ResNet(ResidualBlock,
    #            [3, 4, 6, 3],
    #            [64, 256, 512, 1024],
    #            [256, 512, 1024, 2048],
    #            100).

    def __init__(self,
                 block,
                 layer_nums,
                 in_channels,
                 out_channels,
                 strides=None,
                 num_classes=80):
        super(ResNet, self).__init__()

        # 若三者的len不皆等于4
        if not len(layer_nums) == len(in_channels) == len(out_channels) == 4:
            raise ValueError("the length of "
                             "layer_num, inchannel, outchannel list must be 4!")

        self.conv1 = _conv2d(3, 64, 7, stride=2)
        self.bn1 = _fused_bn(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')

        # 四个层
        self.layer1 = self._make_layer(block,
                                       layer_nums[0],
                                       in_channel=in_channels[0],
                                       out_channel=out_channels[0],
                                       stride=strides[0])
        self.layer2 = self._make_layer(block,
                                       layer_nums[1],
                                       in_channel=in_channels[1],
                                       out_channel=out_channels[1],
                                       stride=strides[1])
        self.layer3 = self._make_layer(block,
                                       layer_nums[2],
                                       in_channel=in_channels[2],
                                       out_channel=out_channels[2],
                                       stride=strides[2])
        self.layer4 = self._make_layer(block,
                                       layer_nums[3],
                                       in_channel=in_channels[3],
                                       out_channel=out_channels[3],
                                       stride=strides[3])

        self.num_classes = num_classes
        if num_classes:
            # 使用指定维度的平均值代替该维度的其他元素，以移除该维度
            # keep_dims=True表示 保留缩小的维度，大小为1 否则移除维度
            self.reduce_mean = ops.ReduceMean(keep_dims=True)
            # 致密层
            self.end_point = nn.Dense(out_channels[3], num_classes, has_bias=True,
                                      weight_init=weight_variable(),
                                      bias_init=weight_variable())
            # 返回删除指定 axis 中大小为1的维度后的Tensor
            self.squeeze = lambda out:ops.squeeze(out,axis = (2, 3))

    def _make_layer(self, block, layer_num, in_channel, out_channel, stride):
        # 为ResNet制造层.
        #
        # 参数：
        #     block (Cell): Resnet 块
        #     layer_num (int): 层数
        #     in_channel (int): 输入通道
        #     out_channel (int): 输出通道
        #     stride (int): 初始卷积层的跨度大小
        #
        # 返回：
        #     SequentialCell, the output layer.
        #
        # 示例：
        #     _make_layer(BasicBlock, 3, 128, 256, 2).
        layers = []

        # 添加layer_num个Block进layers中，最终以级联方式连接
        # 第一个block的stride = stride 其余的stride = 1
        resblk = block(in_channel, out_channel, stride = stride)
        layers.append(resblk)

        for _ in range(1, layer_num - 1):
            resblk = block(out_channel, out_channel, stride = 1)
            layers.append(resblk)

        resblk = block(out_channel, out_channel, stride = 1)
        layers.append(resblk)

        return nn.SequentialCell(layers)

    def construct(self, x):
        # conv-bn-relu-maxpool-layer1-layer2-layer3-layer4
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        c1 = self.maxpool(x)

        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        out = c5
        if self.num_classes:
            out = self.reduce_mean(c5, (2, 3))
            out = self.squeeze(out)
            out = self.end_point(out)
        # c3为layer2的输出
        # c4为layer3的输出
        # out为layer4的输出经reduce_mean、squeeze、end_point的结果
        return c3, c4, out


def resnet18(class_num=10):
    # 得到ResNet18神经网络.
    #
    # 参数：
    #     class_num (int): 类数
    #
    # 返回：
    #     Cell,ResNet18神经网络的cell实例.
    #
    # 示例：
    #     resnet18(100).
    return ResNet(BasicBlock,
                  [2, 2, 2, 2],
                  [64, 64, 128, 256],
                  [64, 128, 256, 512],
                  [1, 2, 2, 2],
                  num_classes=class_num)


class YoloBlock(nn.Cell):
    # YoloBlock 用于 YOLOv3.
    #
    # 参数：
    #     in_channels (int): 输入通道
    #     out_chls (int): 中间通道
    #     out_channels (int): 输出通道
    #
    # 返回：
    #     Tuple, tuple of output tensor,(f1,f2,f3).
    #
    # 示例：
    #     YoloBlock(1024, 512, 255).
    def __init__(self, in_channels, out_chls, out_channels):
        super(YoloBlock, self).__init__()
        out_chls_2 = out_chls * 2

        self.conv0 = _conv_bn_relu(in_channels, out_chls, ksize=1)
        self.conv1 = _conv_bn_relu(out_chls, out_chls_2, ksize=3)

        self.conv2 = _conv_bn_relu(out_chls_2, out_chls, ksize=1)
        self.conv3 = _conv_bn_relu(out_chls, out_chls_2, ksize=3)

        self.conv4 = _conv_bn_relu(out_chls_2, out_chls, ksize=1)
        self.conv5 = _conv_bn_relu(out_chls, out_chls_2, ksize=3)

        self.conv6 = nn.Conv2d(out_chls_2, out_channels, kernel_size=1, stride=1, has_bias=True)

    def construct(self, x):
        c1 = self.conv0(x)
        c2 = self.conv1(c1)

        c3 = self.conv2(c2)
        c4 = self.conv3(c3)

        c5 = self.conv4(c4)
        c6 = self.conv5(c5)

        out = self.conv6(c6)
        return c5, out


class YOLOv3(nn.Cell):
     # YOLOv3 Network.
     #
     # 笔记:
     #     backbone = resnet18.
     #
     # 参数：
     #     feature_shape (list):输入image形状, [N,C,H,W]
     #     backbone_shape (list): resnet18输出通道形状
     #     backbone (Cell): Backbone网络
     #     out_channel (int): 输出通道
     #
     # 返回：
     #     Tensor, output tensor.
     #
     # 示例：
     #     YOLOv3(feature_shape=[1,3,416,416],
     #            backbone_shape=[64, 128, 256, 512, 1024]
     #            backbone=darknet53(),
     #            out_channel=255).

    def __init__(self, feature_shape, backbone_shape, backbone, out_channel):
        super(YOLOv3, self).__init__()
        self.out_channel = out_channel
        self.net = backbone
        self.backblock0 = YoloBlock(backbone_shape[-1], out_chls=backbone_shape[-2], out_channels=out_channel)

        self.conv1 = _conv_bn_relu(in_channel=backbone_shape[-2], out_channel=backbone_shape[-2]//2, ksize=1)
        self.upsample1 = ops.ResizeNearestNeighbor((feature_shape[2]//16, feature_shape[3]//16))
        self.backblock1 = YoloBlock(in_channels=backbone_shape[-2]+backbone_shape[-3],
                                    out_chls=backbone_shape[-3],
                                    out_channels=out_channel)

        self.conv2 = _conv_bn_relu(in_channel=backbone_shape[-3], out_channel=backbone_shape[-3]//2, ksize=1)
        self.upsample2 = ops.ResizeNearestNeighbor((feature_shape[2]//8, feature_shape[3]//8))
        self.backblock2 = YoloBlock(in_channels=backbone_shape[-3]+backbone_shape[-4],
                                    out_chls=backbone_shape[-4],
                                    out_channels=out_channel)
        # 在指定轴上拼接输入Tensor
        self.concat = lambda x:ops.cat(x,axis=1)

    def construct(self, x):
        # x 的 input_shape 为 (batch_size, 3, h, w)的
        # feature_map1 是 (batch_size, backbone_shape[2], h/8, w/8)的
        # feature_map2 是 (batch_size, backbone_shape[3], h/16, w/16)的
        # feature_map3 是 (batch_size, backbone_shape[4], h/32, w/32)的
        feature_map1, feature_map2, feature_map3 = self.net(x)

        # 经第一个yoloblock产生big_object_output
        con1, big_object_output = self.backblock0(feature_map3)

        # conv-bn-relu upsample concat
        con1 = self.conv1(con1)
        ups1 = self.upsample1(con1)
        con1 = self.concat((ups1, feature_map2))

        # 经第二个yoloblock产生medium_object_output
        con2, medium_object_output = self.backblock1(con1)
        # conv-bn-relu upsample concat
        con2 = self.conv2(con2)
        ups2 = self.upsample2(con2)
        con3 = self.concat((ups2, feature_map1))
        # 经第三个yoloblock产生small_object_output
        _, small_object_output = self.backblock2(con3)

        return big_object_output, medium_object_output, small_object_output


class DetectionBlock(nn.Cell):
     # YOLOv3 检测网络  其最终输出检测结果
     #
     # 参数：
     #     scale (str): Character, scale.
     #     config (Class): YOLOv3 config.
     #
     # 返回：
     #     Tuple, tuple of output tensor,(f1,f2,f3).
     #
     # 示例：
     #     DetectionBlock(scale='l',stride=32).

    def __init__(self, scale, config):
        super(DetectionBlock, self).__init__()

        self.config = config
        if scale == 's':
            idx = (0, 1, 2)
        elif scale == 'm':
            idx = (3, 4, 5)
        elif scale == 'l':
            idx = (6, 7, 8)
        else:
            raise KeyError("Invalid scale value for DetectionBlock")
        self.anchors = Tensor([self.config.anchor_scales[i] for i in idx], ms.float32)
        self.num_anchors_per_scale = 3
        self.num_attrib = 4 + 1 + self.config.num_classes
        self.ignore_threshold = 0.5
        self.lambda_coord = 1

        self.sigmoid = nn.Sigmoid()
        self.tile = lambda x,y:ops.tile(x,y)
        self.concat = lambda x:ops.cat(x,axis=-1)
        self.input_shape = Tensor(tuple(config.img_shape[::-1]), ms.float32)

    def construct(self, x):
        num_batch = ops.shape(x)[0]
        grid_size = ops.shape(x)[2:4]

        # Reshape并transpose特征至[n, 3, grid_size[0], grid_size[1], num_attrib]
        prediction = ops.reshape(x, (num_batch,
                                     self.num_anchors_per_scale,
                                     self.num_attrib,
                                     grid_size[0],
                                     grid_size[1]))
        prediction = ops.transpose(prediction, (0, 3, 4, 1, 2))

        range_x = range(grid_size[1])
        range_y = range(grid_size[0])
        grid_x = ops.Cast()(ops.tuple_to_array(range_x), ms.float32)
        grid_y = ops.Cast()(ops.tuple_to_array(range_y), ms.float32)
        # 形状为[grid_size[0], grid_size[1], 1, 1]的张量，代表每个网格的X/Y轴的坐标。
        grid_x = self.tile(ops.reshape(grid_x, (1, 1, -1, 1, 1)), (1, grid_size[0], 1, 1, 1))
        grid_y = self.tile(ops.reshape(grid_y, (1, -1, 1, 1, 1)), (1, 1, grid_size[1], 1, 1))
        # 形状为[grid_size[0], grid_size[1], 1, 2]
        grid = self.concat((grid_x, grid_y))

        box_xy = prediction[:, :, :, :, :2]
        box_wh = prediction[:, :, :, :, 2:4]
        box_confidence = prediction[:, :, :, :, 4:5]
        box_probs = prediction[:, :, :, :, 5:]

        box_xy = (self.sigmoid(box_xy) + grid) / ops.Cast()(ops.tuple_to_array((grid_size[1], grid_size[0])), ms.float32)
        box_wh = ops.exp(box_wh) * self.anchors / self.input_shape
        box_confidence = self.sigmoid(box_confidence)
        box_probs = self.sigmoid(box_probs)

        if self.training:
            return grid, prediction, box_xy, box_wh
        return box_xy, box_wh, box_confidence, box_probs


# 计算box的iou
class Iou(nn.Cell):
    def __init__(self):
        super(Iou, self).__init__()

    def construct(self, box1, box2):
        box1_xy = box1[:, :, :, :, :, :2]
        box1_wh = box1[:, :, :, :, :, 2:4]

        box1_mins = box1_xy - box1_wh / Tensor(2.0)
        box1_maxs = box1_xy + box1_wh / Tensor(2.0)

        box2_xy = box2[:, :, :, :, :, :2]
        box2_wh = box2[:, :, :, :, :, 2:4]

        box2_mins = box2_xy - box2_wh / Tensor(2.0)
        box2_maxs = box2_xy + box2_wh / Tensor(2.0)

        intersect_mins = ops.maximum(box1_mins, box2_mins)
        intersect_maxs = ops.minimum(box1_maxs, box2_maxs)

        intersect_wh = ops.maximum(intersect_maxs - intersect_mins, Tensor(0.0))

        intersect_area = ops.squeeze(intersect_wh[:, :, :, :, :, 0:1],-1) * \
                         ops.squeeze(intersect_wh[:, :, :, :, :, 1:2],-1)
        box1_area = ops.squeeze(box1_wh[:, :, :, :, :, 0:1],-1) * ops.squeeze(box1_wh[:, :, :, :, :, 1:2],-1)
        box2_area = ops.squeeze(box2_wh[:, :, :, :, :, 0:1],-1) * ops.squeeze(box2_wh[:, :, :, :, :, 1:2],-1)

        iou = intersect_area / (box1_area + box2_area - intersect_area)
        return iou


class YoloLossBlock(nn.Cell):
     # YOLOv3 损失块cell 其最终输出scale的损失
     # 参数：
     #     scale (str): 三个scale, 's', 'm' and 'l'
     #     config (Class): YOLOv3默认config
     #
     # 返回：
     #     Tensor,scale的损失
     #
     # 示例：
     #     YoloLossBlock('l', ConfigYOLOV3ResNet18())

    def __init__(self, scale, config):
        super(YoloLossBlock, self).__init__()
        self.config = config
        if scale == 's':
            idx = (0, 1, 2)
        elif scale == 'm':
            idx = (3, 4, 5)
        elif scale == 'l':
            idx = (6, 7, 8)
        else:
            raise KeyError("Invalid scale value for DetectionBlock")
        self.anchors = Tensor([self.config.anchor_scales[i] for i in idx], ms.float32)
        self.ignore_threshold = Tensor(self.config.ignore_threshold, ms.float32)
        self.concat = lambda x:ops.cat(x,axis=-1)
        self.iou = Iou()
        self.cross_entropy = ops.SigmoidCrossEntropyWithLogits()
        self.reduce_sum = ops.ReduceSum()
        self.reduce_max = ops.ReduceMax(keep_dims=False)
        self.input_shape = Tensor(tuple(config.img_shape[::-1]), ms.float32)

    def construct(self, grid, prediction, pred_xy, pred_wh, y_true, gt_box):

        object_mask = y_true[:, :, :, :, 4:5]
        class_probs = y_true[:, :, :, :, 5:]

        grid_shape = ops.shape(prediction)[1:3]
        grid_shape = ops.Cast()(ops.tuple_to_array(grid_shape[::-1]), ms.float32)

        pred_boxes = self.concat((pred_xy, pred_wh))
        true_xy = y_true[:, :, :, :, :2] * grid_shape - grid
        true_wh = y_true[:, :, :, :, 2:4]
        true_wh = ops.select(ops.equal(true_wh, 0.0),
                             ops.fill(ops.DType()(true_wh), ops.shape(true_wh), 1.0),
                             true_wh)
        true_wh = ops.log(true_wh / self.anchors * self.input_shape)
        box_loss_scale = 2 - y_true[:, :, :, :, 2:3] * y_true[:, :, :, :, 3:4]

        gt_shape = ops.shape(gt_box)
        gt_box = ops.reshape(gt_box, (gt_shape[0], 1, 1, 1, gt_shape[1], gt_shape[2]))

        iou = self.iou(ops.ExpandDims()(pred_boxes, -2), gt_box) # [batch, grid[0], grid[1], num_anchor, num_gt]
        best_iou = self.reduce_max(iou, -1) # [batch, grid[0], grid[1], num_anchor]
        ignore_mask = best_iou < self.ignore_threshold
        ignore_mask = ops.Cast()(ignore_mask, ms.float32)
        ignore_mask = ops.ExpandDims()(ignore_mask, -1)
        ignore_mask = ops.stop_gradient(ignore_mask)

        xy_loss = object_mask * box_loss_scale * self.cross_entropy(prediction[:, :, :, :, :2], true_xy)
        wh_loss = object_mask * box_loss_scale * 0.5 * ops.square(true_wh - prediction[:, :, :, :, 2:4])
        confidence_loss = self.cross_entropy(prediction[:, :, :, :, 4:5], object_mask)
        confidence_loss = object_mask * confidence_loss + (1 - object_mask) * confidence_loss * ignore_mask
        class_loss = object_mask * self.cross_entropy(prediction[:, :, :, :, 5:], class_probs)

        # smooth loss
        xy_loss = self.reduce_sum(xy_loss, ())
        wh_loss = self.reduce_sum(wh_loss, ())
        confidence_loss = self.reduce_sum(confidence_loss, ())
        class_loss = self.reduce_sum(class_loss, ())

        loss = xy_loss + wh_loss + confidence_loss + class_loss
        return loss / ops.shape(prediction)[0]


class yolov3_resnet18(nn.Cell):
    # 基于ResNet18的yolov3网络
    #
    # 参数：
    #     config (Class): YOLOv3 config.
    #
    # 返回：
    #     Cell,基于YOLOv3神经网络的ResNet的cell实例
    #
    # 示例：
    #     yolov3_resnet18(80, [1,3,416,416]).

    def __init__(self, config):
        super(yolov3_resnet18, self).__init__()
        self.config = config

        # YOLOv3 network
        self.feature_map = YOLOv3(feature_shape=self.config.feature_shape,
                                  backbone=ResNet(BasicBlock,
                                                  self.config.backbone_layers,
                                                  self.config.backbone_input_shape,
                                                  self.config.backbone_shape,
                                                  self.config.backbone_stride,
                                                  num_classes=None),
                                  backbone_shape=self.config.backbone_shape,
                                  out_channel=self.config.out_channel)

        # 默认anchor box上的预测
        self.detect_1 = DetectionBlock('l', self.config)
        self.detect_2 = DetectionBlock('m', self.config)
        self.detect_3 = DetectionBlock('s', self.config)

    def construct(self, x):
        big_object_output, medium_object_output, small_object_output = self.feature_map(x)
        output_big = self.detect_1(big_object_output)
        output_me = self.detect_2(medium_object_output)
        output_small = self.detect_3(small_object_output)

        return output_big, output_me, output_small


class YoloWithLossCell(nn.Cell):
    # 通过网络提供YOLOv3训练损失
    #
    # 参数：
    #     network (Cell): 训练网络
    #     config (Class): YOLOv3 config
    #
    # 返回：
    #     Tensor,网络损失

    def __init__(self, network, config):
        super(YoloWithLossCell, self).__init__()
        self.yolo_network = network
        self.config = config
        self.loss_big = YoloLossBlock('l', self.config)
        self.loss_me = YoloLossBlock('m', self.config)
        self.loss_small = YoloLossBlock('s', self.config)

    def construct(self, x, y_true_0, y_true_1, y_true_2, gt_0, gt_1, gt_2):
        # 网络输出结果
        yolo_out = self.yolo_network(x)
        # large损失
        loss_l = self.loss_big(yolo_out[0][0], yolo_out[0][1], yolo_out[0][2], yolo_out[0][3], y_true_0, gt_0)
        # medium损失
        loss_m = self.loss_me(yolo_out[1][0], yolo_out[1][1], yolo_out[1][2], yolo_out[1][3], y_true_1, gt_1)
        # small损失
        loss_s = self.loss_small(yolo_out[2][0], yolo_out[2][1], yolo_out[2][2], yolo_out[2][3], y_true_2, gt_2)
        return loss_l + loss_m + loss_s


class TrainingWrapper(nn.Cell):
    # YOLOv3网络训练的封装类
    #
    # 在construct后在训练网络中添加一个优化器
    # 函数可以被调用来创建后向图
    #
    # 参数：
    #     network (Cell): 训练网络  注意，应该加入损失函数
    #     optimizer (Optimizer): 更新权重的优化器
    #     sens (Number): 调整参数。默认值：1.0
    def __init__(self, network, optimizer, sens=1.0):
        super(TrainingWrapper, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.weights = ms.ParameterTuple(network.trainable_params())
        self.optimizer = optimizer
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = None
        self.parallel_mode = ms.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ms.ParallelMode.DATA_PARALLEL, ms.ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        if self.reducer_flag:
            mean = ms.get_auto_parallel_context("gradients_mean")
            if ms.parallel._auto_parallel_context.auto_parallel_context().get_device_num_is_set():
                degree = ms.get_auto_parallel_context("device_num")
            else:
                degree = ms.communication.get_group_size()
            self.grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, degree)

    def construct(self, *args):
        weights = self.weights
        loss = self.network(*args)
        sens = ops.fill(ops.DType()(loss), ops.shape(loss), self.sens)
        grads = self.grad(self.network, weights)(*args, sens)
        if self.reducer_flag:
            # apply grad reducer on grads
            grads = self.grad_reducer(grads)
        return ops.depend(loss, self.optimizer(grads))


class YoloBoxScores(nn.Cell):
    # 计算原始图片尺寸的box和每个box的分数.
    #
    # 参数：
    #     config (Class): YOLOv3 config.
    #
    # 返回：
    #     Tensor, 原始图片size的box
    #     Tensor, 每个box的分数

    def __init__(self, config):
        super(YoloBoxScores, self).__init__()
        self.input_shape = Tensor(np.array(config.img_shape), ms.float32)
        self.num_classes = config.num_classes

    def construct(self, box_xy, box_wh, box_confidence, box_probs, image_shape):
        batch_size = ops.shape(box_xy)[0]
        x = box_xy[:, :, :, :, 0:1]
        y = box_xy[:, :, :, :, 1:2]
        box_yx = ops.cat((y, x),-1)
        w = box_wh[:, :, :, :, 0:1]
        h = box_wh[:, :, :, :, 1:2]
        box_hw = ops.cat((h, w),-1)

        new_shape = ops.round(image_shape * ops.ReduceMin()(self.input_shape / image_shape))
        offset = (self.input_shape - new_shape) / 2.0 / self.input_shape
        scale = self.input_shape / new_shape
        box_yx = (box_yx - offset) * scale
        box_hw = box_hw * scale

        box_min = box_yx - box_hw / 2.0
        box_max = box_yx + box_hw / 2.0
        boxes = ops.cat((box_min[:, :, :, :, 0:1],
                              box_min[:, :, :, :, 1:2],
                              box_max[:, :, :, :, 0:1],
                              box_max[:, :, :, :, 1:2]),
                           -1)
        image_scale = ops.tile(image_shape, (1, 2))
        boxes = boxes * image_scale
        boxes = ops.reshape(boxes, (batch_size, -1, 4))
        boxes_scores = box_confidence * box_probs
        boxes_scores = ops.reshape(boxes_scores, (batch_size, -1, self.num_classes))
        return boxes, boxes_scores


class YoloWithEval(nn.Cell):
    # YOLOv3评估的封装类
    #
    # 参数：
    #     network (Cell): 训练网络  注意，不应该加入损失函数和优化器
    #     config (Class): YOLOv3 config.
    #
    # 返回：
    #     Tensor, 原始图片size的box
    #     Tensor, 每个box的分数
    #     Tensor, 原始图片size
    def __init__(self, network, config):
        super(YoloWithEval, self).__init__()
        self.yolo_network = network
        self.box_score_0 = YoloBoxScores(config)
        self.box_score_1 = YoloBoxScores(config)
        self.box_score_2 = YoloBoxScores(config)

    def construct(self, x, image_shape):
        yolo_output = self.yolo_network(x)
        boxes_0, boxes_scores_0 = self.box_score_0(*yolo_output[0], image_shape)
        boxes_1, boxes_scores_1 = self.box_score_1(*yolo_output[1], image_shape)
        boxes_2, boxes_scores_2 = self.box_score_2(*yolo_output[2], image_shape)
        boxes = ops.cat((boxes_0, boxes_1, boxes_2),1)
        boxes_scores = ops.cat((boxes_scores_0, boxes_scores_1, boxes_scores_2),1)
        return boxes, boxes_scores, image_shape

### 定义评价指标



# 科学计算库
import numpy as np

# 计算预测bbox和ground truth的iou
def calc_iou(bbox_pred, bbox_ground):
    # 获取预测bbox和ground truth的坐标信息
    x1 = bbox_pred[0]
    y1 = bbox_pred[1]
    width1 = bbox_pred[2] - bbox_pred[0]
    height1 = bbox_pred[3] - bbox_pred[1]

    x2 = bbox_ground[0]
    y2 = bbox_ground[1]
    width2 = bbox_ground[2] - bbox_ground[0]
    height2 = bbox_ground[3] - bbox_ground[1]

    # 计算两个bbox相交的区域面积和总面积，从而求得iou
    endx = max(x1 + width1, x2 + width2)
    startx = min(x1, x2)
    width = width1 + width2 - (endx - startx)

    endy = max(y1 + height1, y2 + height2)
    starty = min(y1, y2)
    height = height1 + height2 - (endy - starty)

    if width <= 0 or height <= 0:
        iou = 0
    else:
        area = width * height
        area1 = width1 * height1
        area2 = width2 * height2
        iou = area * 1. / (area1 + area2 - area)

    return iou


# 应用NMS至bbox
def apply_nms(all_boxes, all_scores, thres, max_boxes):
    # 提取bbox的左上角和右下角坐标
    x1 = all_boxes[:, 0]
    y1 = all_boxes[:, 1]
    x2 = all_boxes[:, 2]
    y2 = all_boxes[:, 3]
    # 计算bbox的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # 根据bbox的置信度分数进行排序
    order = all_scores.argsort()[::-1]

    # 用于保存被保留的bbox的下标
    keep = []

    # 迭代地进行非极大值抑制
    while order.size > 0:
        # 将置信度最高的bbox添加到keep列表中
        i = order[0]
        keep.append(i)

        # 如果已经保留了足够数量的bbox，则停止
        if len(keep) >= max_boxes:
            break

        # 提取除当前bbox之外的所有bbox的坐标
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        # 计算当前bbox与其他所有bbox的重叠面积
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        # 计算当前bbox与其他所有bbox的重叠率
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        # 找到重叠率小于阈值的bbox的下标
        inds = np.where(ovr <= thres)[0]

        # 更新order数组，用于下一轮非极大值抑制
        order = order[inds + 1]
    # 返回被保留的bbox的下标
    return keep


# 计算预测bbox的precison和recall
def metrics(pred_data):
    # 加载配置
    config = ConfigYOLOV3ResNet18()
    # 获取类别数
    num_classes = config.num_classes
    # 正确检测计数，防止分母为0
    count_corrects = [1e-6 for _ in range(num_classes)]
    # 标注计数，防止分母为0
    count_grounds = [1e-6 for _ in range(num_classes)]
    # 预测计数，防止分母为0
    count_preds = [1e-6 for _ in range(num_classes)]

    # 遍历每个样本
    for i, sample in enumerate(pred_data):
        # 获取标注
        gt_anno = sample["annotation"]
        # 获取预测分数
        box_scores = sample['box_scores']
        # 获取预测框坐标
        boxes = sample['boxes']
        # 分数高于obj_threshold的预测框作为mask
        mask = box_scores >= config.obj_threshold
        boxes_ = []
        scores_ = []
        classes_ = []
        # 获取nms的最大框数
        max_boxes = config.nms_max_num
        for c in range(num_classes):
            # 获取属于第c类的预测框坐标
            class_boxes = np.reshape(boxes, [-1, 4])[np.reshape(mask[:, c], [-1])]
            # 获取属于第c类的预测框分数
            class_box_scores = np.reshape(box_scores[:, c], [-1])[np.reshape(mask[:, c], [-1])]
            # 应用nms
            nms_index = apply_nms(class_boxes, class_box_scores, config.nms_threshold, max_boxes)
            # 获取通过nms后的预测框坐标
            class_boxes = class_boxes[nms_index]
            # 获取通过nms后的预测框分数
            class_box_scores = class_box_scores[nms_index]
            # 获取预测框类别
            classes = np.ones_like(class_box_scores, 'int32') * c

            boxes_.append(class_boxes)
            scores_.append(class_box_scores)
            classes_.append(classes)

        # 合并所有预测框坐标
        boxes = np.concatenate(boxes_, axis=0)
        # 合并所有预测框类别
        classes = np.concatenate(classes_, axis=0)

        # 度量
        # 正确检测计数，防止分母为0
        count_correct = [1e-6 for _ in range(num_classes)]
        # 标注计数，防止分母为0
        count_ground = [1e-6 for _ in range(num_classes)]
        # 预测计数，防止分母为0
        count_pred = [1e-6 for _ in range(num_classes)]

        # 遍历每个标注
        for anno in gt_anno:
            # 计数
            count_ground[anno[4]] += 1

        for box_index, box in enumerate(boxes):
            # 将bbox从(ymin, xmin, ymax, xmax)的格式转换为(xmin, ymin, xmax, ymax)的格式
            bbox_pred = [box[1], box[0], box[3], box[2]]
            # 对属于每个类别的预测框的数量进行计数
            count_pred[classes[box_index]] += 1

            # 计算每个真实标签与每个预测框的交并比并更新度量计数器
            for anno in gt_anno:
                class_ground = anno[4]

                # 如果该预测框属于当前真实标签类别
                if classes[box_index] == class_ground:
                    # 计算交并比
                    iou = calc_iou(bbox_pred, anno)
                    # 如果交并比大于等于0.5，则将该预测框视为正确的预测，并更新计数器
                    if iou >= 0.5:
                        count_correct[class_ground] += 1
                        break

        # 将该批次中每个类别的度量计数器的值累加到总的度量计数器中
        count_corrects = [count_corrects[i] + count_correct[i] for i in range(num_classes)]
        count_preds = [count_preds[i] + count_pred[i] for i in range(num_classes)]
        count_grounds = [count_grounds[i] + count_ground[i] for i in range(num_classes)]

    # 计算每个类别的精确度和召回率，并将结果存储为numpy数组形式并返回
    precision = np.array([count_corrects[ix] / count_preds[ix] for ix in range(num_classes)])
    recall = np.array([count_corrects[ix] / count_grounds[ix] for ix in range(num_classes)])
    return precision, recall

### 定义训练网络的函数
######################## train YOLOv3 example ########################
# 多种操作系统接口
import os
# 用于命令项选项与参数解析
import argparse
# 抽象语法树
import ast
# 可以使得以属性的方式去访问字典的值
from easydict import EasyDict as edict
# 提供了对文件(夹)复制，移动，删除，压缩和解压缩的方法
import shutil

# 科学计算
import numpy as np
# mindspore库
import mindspore as ms
# 神经网络Cell 用于构建神经网络中的预定义构建块或计算单元
from mindspore import nn
# 张量
from mindspore import Tensor
# 初始化通信服务需要的分布式后端
from mindspore.communication.management import init
# CheckpointConfig 保存checkpoint时的配置策略
# ModelCheckpoint checkpoint的回调函数
# LossMintor训练场景下，监控训练的loss；边训练边推理场景下，监控训练的loss和推理的metrics
from mindspore.train import CheckpointConfig, ModelCheckpoint, LossMonitor
# 模型训练或推理的高阶接口。
from mindspore.train import Model
# 加载点 加载参数
from mindspore import load_checkpoint, load_param_into_net
# 创建并初始化一个Tensor。
from mindspore.common.initializer import initializer
# 设置随机种子
from mindspore import set_seed
# 提供了许多函数和变量来处理 Python 运行时环境的不同部分
import sys

set_seed(1)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

# 初始化网络参数
def init_net_param(network, init_value='ones'):
    params = network.trainable_params()
    for p in params:
        if isinstance(p.data, Tensor) and 'beta' not in p.name and 'gamma' not in p.name and 'bias' not in p.name:
            p.set_data(initializer(init_value, p.data.shape, p.data.dtype))


# main函数
def main(args_opt):
    device_target = ms.get_context("device_target")
    mode = ms.GRAPH_MODE
    print("mode:",device_target)
    if device_target == "GPU":
        ms.set_context(mode=mode, device_target="GPU", device_id=args_opt.device_id)
    elif device_target == "Ascend":
        ms.set_context(mode=mode, device_target="Ascend", device_id=args_opt.device_id)
    else:
        ms.set_context(mode=mode, device_target="CPU")

    if args_opt.distribute:
        device_num = args_opt.device_num
        ms.reset_auto_parallel_context()
        ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                          device_num=device_num)
        init()
        rank = args_opt.device_id % device_num
    else:
        rank = 0
        device_num = 1

    loss_scale = float(args_opt.loss_scale)

    # 当创造MindDataset，使用第一mindrecord文件，例如yolo.mindrecord0
    dataset = create_yolo_dataset(args_opt.mindrecord_file,
                                  batch_size=args_opt.batch_size, device_num=device_num, rank=rank)
    dataset_size = dataset.get_dataset_size()
    print('The dataset size: ', dataset_size)
    print("Create dataset done!")

    net = yolov3_resnet18(ConfigYOLOV3ResNet18())
    net = YoloWithLossCell(net, ConfigYOLOV3ResNet18())
    init_net_param(net, "XavierUniform")

    # checkpoint
    ckpt_config = CheckpointConfig(save_checkpoint_steps=dataset_size * args_opt.save_checkpoint_epochs,
                                   keep_checkpoint_max=args_opt.keep_checkpoint_max)

    ckpoint_cb = ModelCheckpoint(prefix="yolov3", directory=cfg.ckpt_dir, config=ckpt_config)

    # 如果有预加载模型存在 则加载预加载模型
    if args_opt.pre_trained and os.path.exists(args_opt.pre_trained):
        if args_opt.pre_trained_epoch_size <= 0:
            raise KeyError("pre_trained_epoch_size must be greater than 0.")
        param_dict = load_checkpoint(args_opt.pre_trained)
        load_param_into_net(net, param_dict)
        print("Load pre_trained")

    lr = args_opt.lr
    opt = nn.Adam(filter(lambda x: x.requires_grad, net.get_parameters()), lr, loss_scale=loss_scale)
    net = TrainingWrapper(net, opt, loss_scale)

    callback = [LossMonitor(10 * dataset_size), ckpoint_cb]
    model = Model(net)
    dataset_sink_mode = cfg.dataset_sink_mode
    print('The epoch size: ', args_opt.epoch_size)
    print("Start train YOLOv3, the first epoch will be slower because of the graph compilation.")
    model.train(args_opt.epoch_size, dataset, callbacks=callback, dataset_sink_mode=dataset_sink_mode)

## 模型训练
# ------------yolov3 train -----------------------------
# 训练参数
cfg = edict({
    "distribute": False,
    "device_id": 0,
    "device_num": 1,
    "dataset_sink_mode": True,

    "lr": 0.001,
    "epoch_size": 200,
    "batch_size": batch_size,
    "loss_scale": 1024,

    "pre_trained": 'pre_trained/pre.ckpt',    #预训练文件地址
    "pre_trained_epoch_size": 60,             #初始值为0，进行预训练的时候要进行修改

    "ckpt_dir": "./ckpt",
    "save_checkpoint_epochs": 10,
    'keep_checkpoint_max': 15,

    "train_url": './output',                   # 此处需要修改成自己桶地址
})

if __name__ == '__main__':

    #if os.path.exists(cfg.ckpt_dir):
    #    shutil.rmtree(cfg.ckpt_dir)

    data_path = "."
    mindrecord_dir_train = os.path.join(data_path, 'mindrecord/train')

    print("Start create dataset!")
    # 它将在args_opt.mindrecord_dir生成mindrecord文件，文件名是yolo.mindrecord
    prefix = "yolo.mindrecord"
    cfg.mindrecord_file = os.path.join(mindrecord_dir_train, prefix)

    main(cfg)
    exit(0)

