
# 基于MindSpore实现目标分割


import os
import mindspore
from mindspore import context
device_id = int(0)
context.set_context(mode=context.GRAPH_MODE, device_target="GPU", save_graphs=False)
mindspore.set_seed(1)

import math
import random
import numpy as np
from PIL import Image, ImageSequence
import matplotlib.pyplot as plt

#显示下载好的数据
train_image_path = "data/train-volume.tif"
train_masks_path = "data/train-labels.tif"
image = np.array([np.array(p) for p in ImageSequence.Iterator(Image.open(train_image_path))])
masks = np.array([np.array(p) for p in ImageSequence.Iterator(Image.open(train_masks_path))])

#展示图片
def show_image(image_list,num = 6):
    img_titles = []
    img_draws = []
    for ind,img in enumerate(image_list):
        if ind == num:
            break
        img_titles.append(ind)
        img_draws.append(img)

    for i in range(len(img_titles)):
        if len(img_titles) > 6:
            row = 3
        elif 3<len(img_titles)<=6:
            row = 2
        else:
            row = 1
        col = math.ceil(len(img_titles)/row)
        plt.subplot(row,col,i+1),plt.imshow(img_draws[i],'gray')
        plt.title(img_titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()


### 数据加载
import cv2
import os
import random
import mindspore
import mindspore.dataset as ds
import glob
import mindspore.dataset.vision as vision_C
import mindspore.dataset.transforms as C_transforms
from mindspore.dataset.vision import Inter

def train_transforms(img_size):
    return [
        vision_C.Resize(img_size, interpolation=Inter.NEAREST),
        vision_C.Rescale(1./255., 0.0),          # 将像素值缩放到范围 [0, 1]
        vision_C.RandomHorizontalFlip(prob=0.5), # 以 0.5 的概率进行随机水平翻转
        vision_C.RandomVerticalFlip(prob=0.5),   # 以 0.5 的概率进行随机垂直翻转
        vision_C.HWC2CHW()                       # 将图像的通道维度从 "HWC"（高度、宽度、通道）顺序转换为 "CHW"（通道、高度、宽度）顺序
    ]


def val_transforms(img_size):
    return [
        vision_C.Resize(img_size, interpolation=Inter.NEAREST),
        vision_C.Rescale(1/255., 0),# 将像素值缩放到范围 [0, 1]
        vision_C.HWC2CHW()          # 将图像的通道维度从 "HWC"（高度、宽度、通道）顺序转换为 "CHW"（通道、高度、宽度）顺序
    ]


class Data_Loader:
    def __init__(self, data_path):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'image/*.png')) # 获取所有图像文件的路径
        self.label_path = glob.glob(os.path.join(data_path, 'mask/*.png')) # 获取所有标签文件的路径

    def __getitem__(self, index):
        # 根据index读取图片
        image = cv2.imread(self.imgs_path[index])                           # 读取图像文件
        label = cv2.imread(self.label_path[index], cv2.IMREAD_GRAYSCALE)    # 以灰度模式读取标签文件
        label = label.reshape((label.shape[0], label.shape[1], 1))          # 将标签的通道维度从 2D 转换为 3D

        return image, label

    @property
    def column_names(self):
        # 返回列名
        column_names = ['image', 'label']
        return column_names

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)


def create_dataset(data_dir, img_size, batch_size, augment, shuffle):
    # 创建 Data_Loader 对象
    mc_dataset = Data_Loader(data_path=data_dir)
    # 创建 GeneratorDataset 对象
    dataset = ds.GeneratorDataset(mc_dataset, mc_dataset.column_names, shuffle=shuffle)
    # 根据是否进行数据增强选择转换操作
    if augment:
        transform_img = train_transforms(img_size)
    else:
        transform_img = val_transforms(img_size)
    # 设置随机种子
    seed = random.randint(1,1000)
    mindspore.set_seed(seed)
    # 对标签进行转换操作
    dataset = dataset.map(input_columns='image', num_parallel_workers=1, operations=transform_img)
    mindspore.set_seed(seed)
    # 对图像进行转换操作
    dataset = dataset.map(input_columns="label", num_parallel_workers=1, operations=transform_img)
    # 如果需要进行打乱，将数据集打乱
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)
    # 将数据集按批次划分
    dataset = dataset.batch(batch_size, num_parallel_workers=1)
    # 打印数据集大小信息
    if augment == True and shuffle == True:
        print("训练集数据量：", len(mc_dataset))
    elif augment == False and shuffle == False:
        print("验证集数据量：", len(mc_dataset))
    else:
        pass
    return dataset

if __name__ == '__main__':
    show_image(image,num = 12)
    show_image(masks,num = 12)

    # 创建验证集的数据集
    train_dataset = create_dataset('data/ISBI/val', img_size=224, batch_size=3, augment=False, shuffle=False)
    # 遍历数据集并打印图像和标签信息
    for item, (image, label) in enumerate(train_dataset):
        if item < 5:
            # 打印每个批次的图像和标签的形状和数据类型信息
            print(f"Shape of image [N, C, H, W]: {image.shape} {image.dtype}",'---',
                  f"Shape of label [N, C, H, W]: {label.shape} {label.dtype}")

    exit(0)