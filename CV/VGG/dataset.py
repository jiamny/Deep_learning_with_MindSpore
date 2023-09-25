import numpy as np
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as C
import mindspore.dataset.vision as CV
from mindspore import dtype as mstype

def create_dataset_cifar10(dataset_dir, usage, resize, batch_size, workers):
    """
    :param dataset_dir: 数据集根目录
    :param usage: 值可以为"train"或"test"，表示是构建训练集还是测试集
    :param resize:处理后的数据集图像大小，本程序中设置为32
    :param batch_size:批量大小
    :param workers:并行线程个数
    :return:返回处理好的数据集

    shuffle=True表示需要混洗数据集，即随机在其中取数据而不是按照顺序
    """
    """
    利用mindspore.dataset中的函数Cifar10Dataset对CIFAR-10数据集进行处理。
    该函数读取和解析CIFAR-10数据集的源文件构建数据集。
    生成的数据集有两列: [image, label] 。 image 列的数据类型是uint8。label 列的数据类型是uint32。
    具体说明查看API文档：https://www.mindspore.cn/docs/zh-CN/master/api_python/dataset/mindspore.dataset.Cifar10Dataset.html?highlight=cifar10dataset
    """
    data_set = ds.Cifar10Dataset(dataset_dir=dataset_dir,
                                 usage=usage,
                                 num_parallel_workers=workers,
                                 shuffle=True)

    trans = []#需要做的变化的集合
    """
    对于训练集，首先进行随机裁剪和随机翻转的操作。
    使用mindspore.dataset.vision.RandomCrop对输入图像进行随机区域的裁剪,大小为(32, 32)。(4, 4, 4, 4)表示在裁剪前，将在图像上下左右各填充4个像素的空白。
    使用mindspore.dataset.RandomHorizontalFlip,对输入图像按50%的概率进行水平随机翻转
    """
    if usage == "train":
        trans += [
            vision.RandomCrop((32, 32), (4, 4, 4, 4)),
            vision.RandomHorizontalFlip(prob=0.5)
        ]

    """
    再对数据集进行一些操作
    """
    trans += [
        vision.Resize(resize),
        vision.Rescale(1.0 / 255.0, 0.0),
        vision.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        vision.HWC2CHW()
    ]

    #对于label进行的操作
    target_trans = [(lambda x: np.array([x]).astype(np.int32)[0])]

    # 数据映射操作
    data_set = data_set.map(
        operations=trans,
        input_columns='image',
        num_parallel_workers=workers)

    data_set = data_set.map(
        operations=target_trans,
        input_columns='label',
        num_parallel_workers=workers)

    # 批量操作
    data_set = data_set.batch(batch_size)
    return data_set


def create_flower_dataset(cfg, sample_num=None):
    #从目录中读取图像的源数据集。
    de_dataset = ds.ImageFolderDataset(cfg.data_path, num_samples=sample_num,
                                       class_indexing={'daisy':0,'dandelion':1,'roses':2,'sunflowers':3,'tulips':4})
    #解码前将输入图像裁剪成任意大小和宽高比。
    transform_img = CV.RandomCropDecodeResize([cfg.image_width, cfg.image_height], scale=(0.08, 1.0), ratio=(0.75, 1.333))  #改变尺寸

    #转换输入图像；形状（H, W, C）为形状（C, H, W）。
    hwc2chw_op = CV.HWC2CHW()
    #转换为给定MindSpore数据类型的Tensor操作。
    type_cast_op = C.TypeCast(mstype.float32)

    #将操作中的每个操作应用到此数据集。
    de_dataset = de_dataset.map(input_columns="image", num_parallel_workers=2, operations=transform_img)
    de_dataset = de_dataset.map(input_columns="image", operations=hwc2chw_op, num_parallel_workers=2)
    de_dataset = de_dataset.map(input_columns="image", operations=type_cast_op, num_parallel_workers=2)
    de_dataset = de_dataset.shuffle(buffer_size=cfg.data_size)
    return de_dataset


def get_flower_dataset(cfg, sample_num=None):
    '''
    读取并处理数据
    '''
    de_dataset = create_flower_dataset(cfg, sample_num=sample_num)

    #划分训练集测试集
    (de_train,de_test)=de_dataset.split([0.8,0.2])
    #设置每个批处理的行数
    #drop_remainder确定是否删除最后一个可能不完整的批（default=False）。
    #如果为True，并且如果可用于生成最后一个批的batch_size行小于batch_size行，则这些行将被删除，并且不会传播到子节点。
    de_train=de_train.batch(cfg.batch_size, drop_remainder=True)
    #重复此数据集计数次数。
    de_test=de_test.batch(cfg.batch_size, drop_remainder=True)
    return de_train, de_test