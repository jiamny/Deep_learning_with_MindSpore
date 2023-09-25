

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
# 引入MindSpore
import mindspore as ms
# 引入神经网络模块
import mindspore.nn as nn
# 引入张量模块
from mindspore import Tensor
from mindspore import load_checkpoint
from mindspore import load_param_into_net
from mindspore import dtype as mstype
# 引入python解释器和它的环境有关的函数
from easydict import EasyDict as edict
import matplotlib.pyplot as plt
# 引入可视化库
import matplotlib as mpl
# 引入可视化库色彩模块
import matplotlib.colors as colors
from train import DeepLabV3, SegDataset

## 模型预测

## 构建预测模块
#设置MindSpore的模式为图模式，设备类型为CPU
ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU", save_graphs=False)

#计算直方图
def cal_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(np.int32) + b[k], minlength=n ** 2).reshape(n, n)

#长边缩放函数
def resize_long(img, long_size=513):
    h, w, _ = img.shape
    if h > w:
        new_h = long_size
        new_w = int(1.0 * long_size * w / h)
    else:
        new_w = long_size
        new_h = int(1.0 * long_size * h / w)
    imo = cv2.resize(img, (new_w, new_h))
    return imo

#构建评估网络
class BuildEvalNetwork(nn.Cell):
    def __init__(self, network):
        super(BuildEvalNetwork, self).__init__()
        self.network = network
        self.softmax = nn.Softmax(axis=1)

    def construct(self, input_data):
        output = self.network(input_data)
        output = self.softmax(output)
        return output

#预处理函数
def pre_process(args, img_, crop_size=513):
    # resize
    img_ = resize_long(img_, crop_size)
    resize_h, resize_w, _ = img_.shape

    # mean, std
    image_mean = np.array(args.image_mean)
    image_std = np.array(args.image_std)
    img_ = (img_ - image_mean) / image_std

    # pad to crop_size
    pad_h = crop_size - img_.shape[0]
    pad_w = crop_size - img_.shape[1]
    if pad_h > 0 or pad_w > 0:
        img_ = cv2.copyMakeBorder(img_, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

    # hwc to chw
    img_ = img_.transpose((2, 0, 1))
    return img_, resize_h, resize_w

# 定义了一个函数eval_batch，用于在输入一张或多张图像时，评估模型的输出值参数args为命令行参数，eval_net为评估用网络，img_lst为输入图像的列表，crop_size为裁剪后的大小，默认为513flip为是否对图像进行镜像翻转，默认为True
def eval_batch(args, eval_net, img_lst, crop_size=513, flip=True):
    # 初始化结果列表
    result_lst = []
    # 获取batch_size
    batch_size = len(img_lst)
    # 初始化batch_img矩阵，尺寸为(batch_size, 3, crop_size, crop_size)
    batch_img = np.zeros((args.batch_size, 3, crop_size, crop_size), dtype=np.float32)
    # 初始化resize_hw列表，用于记录每张图片经过预处理后的尺寸
    resize_hw = []
    # 循环处理每张图片
    for l in range(batch_size):
        # 获取当前图片
        img_ = img_lst[l]
        # 对当前图片进行预处理，返回预处理后的图片，以及裁剪后的高度和宽度
        img_, resize_h, resize_w = pre_process(args, img_, crop_size)
        # 将预处理后的图片加入batch_img矩阵中
        batch_img[l] = img_
        # 将裁剪后的高度和宽度加入resize_hw列表中
        resize_hw.append([resize_h, resize_w])
    # 将batch_img矩阵以连续的方式存储
    batch_img = np.ascontiguousarray(batch_img)
    # 通过评估用网络（eval_net）对batch_img矩阵进行评估，得到输出结果net_out
    net_out = eval_net(Tensor(batch_img, mstype.float32))
    # 将输出结果转换为numpy数组
    net_out = net_out.asnumpy()
    # 如果flip为True，则对batch_img矩阵进行镜像翻转，并再次对翻转后的batch_img矩阵进行评估，将得到的结果加到net_out中
    if flip:
        batch_img = batch_img[:, :, :, ::-1]
        net_out_flip = eval_net(Tensor(batch_img, mstype.float32))
        net_out += net_out_flip.asnumpy()[:, :, :, ::-1]
    # 循环处理每个batch
    for bs in range(batch_size):
        # 获取输出结果的概率值
        probs_ = net_out[bs][:, :resize_hw[bs][0], :resize_hw[bs][1]].transpose((1, 2, 0))
        # 获取原始图像的高度和宽度
        ori_h, ori_w = img_lst[bs].shape[0], img_lst[bs].shape[1]
        # 将概率值的尺寸调整为原始图像的尺寸
        probs_ = cv2.resize(probs_, (ori_w, ori_h))
        # 将处理后的结果加到结果列表中
        result_lst.append(probs_)
    # 返回结果列表
    return result_lst

#定义了一个函数eval_batch_scales，用于在输入一张或多张图像时，按照不同比例分别进行评估，并将结果加起来参数args为命令行参数，eval_net为评估用网络，img_lst为输入图像的列表，scales为不同比例的列表base_crop_size为基准裁剪尺寸，默认为513，flip为是否对图像进行镜像翻转，默认为True
def eval_batch_scales(args, eval_net, img_lst, scales,
                      base_crop_size=513, flip=True):
    # 根据比例列表计算不同尺寸的裁剪尺寸
    sizes_ = [int((base_crop_size - 1) * sc) + 1 for sc in scales]
    # 对第一个尺寸进行评估，并将结果加入probs_lst列表
    probs_lst = eval_batch(args, eval_net, img_lst, crop_size=sizes_[0], flip=flip)
    # 对其他尺寸进行评估，并将结果加到probs_lst中
    for crop_size_ in sizes_[1:]:
        probs_lst_tmp = eval_batch(args, eval_net, img_lst, crop_size=crop_size_, flip=flip)
        for pl, _ in enumerate(probs_lst):
            probs_lst[pl] += probs_lst_tmp[pl]

    result_msk = []
    for i in probs_lst:
        # 将评估得到的概率值转换为类别号，加入result_msk列表中
        result_msk.append(i.argmax(axis=2))
    # 返回结果列表
    return result_msk

# The color source: print(list(colors.cnames.keys()))
#print(list(colors.cnames.keys()))
num_class = {0: 'background', 1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat', 5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat',
             9: 'chair', 10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse', 14: 'motorbike', 15: 'person', 16: 'pottedplant',
             17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor', 21: 'edge'}

num_color = {0:'aliceblue', 1:'grey', 2:'red', 3:'green', 4:'darkorange', 5:'lime', 6:'bisque',
             7:'black', 8:'blanchedalmond', 9:'blue', 10:'blueviolet', 11:'brown', 12:'burlywood', 13:'cadetblue',
             14:'darkorange', 15:'tan', 16:'darkviolet', 17:'cornflowerblue', 18:'yellow', 19:'crimson', 20:'darkcyan'}

color_dic = [num_color[k] for k in sorted(num_color.keys())]
bounds = list(range(21))
cmap = mpl.colors.ListedColormap(color_dic)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
# 定义一个函数num_to_ClassAndColor，用于将类别号转换为颜色和类别名称
def num_to_ClassAndColor(num_list):
    # 初始化颜色列表和类别列表
    color_ = []
    class_ = []
    # 循环处理每个类别号
    for num in num_list:
        # 将类别号对应的颜色加入颜色列表中
        color_.append(num_class[num])
        # 将类别号对应的类别名称加入类别列表中
        class_.append(num_color[num])
    # 返回颜色列表和类别列表
    return color_,class_


def net_eval(args):
    # 根据命令行参数和模型类型（args.model），创建评估用网络
    if args.model == 'deeplab_v3_s16':
        network = DeepLabV3('eval', args.num_classes, 16, args.freeze_bn)
    elif args.model == 'deeplab_v3_s8':
        network = DeepLabV3('eval', args.num_classes, 8, args.freeze_bn)
    else:
        raise NotImplementedError('model [{:s}] not recognized'.format(args.model))

    eval_net = BuildEvalNetwork(network)

    # 加载训练好的模型参数
    param_dict = load_checkpoint(args.ckpt_file)
    load_param_into_net(eval_net, param_dict)
    # 设置评估用网络为测试状态
    eval_net.set_train(False)

    # 读取数据列表
    with open(args.data_lst) as f:
        img_lst = f.readlines()

    # evaluate函数,初始化hist矩阵，大小为(args.num_classes, args.num_classes)初始化batch_img_lst和batch_msk_lstbi表示batch中的图片数量，image_num表示总共处理的图片数量

    hist = np.zeros((args.num_classes, args.num_classes))
    batch_img_lst = []
    batch_msk_lst = []
    bi = 0
    image_num = 0
    # 遍历img_lst中的每个id
    for i, line in enumerate(img_lst):
        id_ = line.strip()
        img_path = os.path.join(cfg.voc_img_dir, id_ + '.jpg')
        msk_path = os.path.join(cfg.voc_anno_gray_dir, id_ + '.png')
        # 读取图像和标签
        img_ = cv2.imread(img_path)
        msk_ = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
        batch_img_lst.append(img_)
        batch_msk_lst.append(msk_)
        # 判断是否需要输出图片评估结果
        if args.if_png:
            # 对当前batch中的所有图片进行评估
            batch_res = eval_batch_scales(args, eval_net, batch_img_lst, scales=args.scales,
                                          base_crop_size=args.crop_size, flip=args.flip)
            # 获取图像大小
            height ,weight = batch_res[0].shape
            # 将标签中的ignore_label设置为0
            batch_msk_lst[0][batch_msk_lst[0]==args.ignore_label] = 0
            # 绘制三张显示图像、预测结果和标签的子图
            plt.figure(figsize=(3 * weight/1024*10, 2 * height/1024*10))
            plt.subplot(1,3,1)
            image = Image.open(img_path)
            plt.imshow(image)

            plt.subplot(1,3,2)
            plt.imshow(image)
            plt.imshow(batch_res[0],alpha=0.8,interpolation='none', cmap=cmap, norm=norm)

            plt.subplot(1,3,3)
            plt.imshow(image)
            plt.imshow(batch_msk_lst[0],alpha=0.8,interpolation='none', cmap=cmap, norm=norm)
            plt.show()
            # 获取预测结果和真实标签中的唯一值
            prediction_num = np.unique(batch_res[0])
            real_num = np.unique(batch_msk_lst[0])
            # 获取预测结果和真实标签对应的颜色值和类别名
            prediction_color,prediction_class = num_to_ClassAndColor(prediction_num)
            print('prediction num:',prediction_num)
            print('prediction color:',prediction_color)
            print('prediction class:',prediction_class)
            real_color,real_class = num_to_ClassAndColor(real_num)
            print('groundtruth num:',real_num)
            print('groundtruth color:',real_color)
            print('groundtruth class:',real_class)
            # 清空batch_img_lst和batch_msk_lst，用于存储下一个batch的图像和标签
            batch_img_lst = []
            batch_msk_lst = []
            # 判断当前处理的图片数是否小于args.num_png
            if i < args.num_png-1:
                continue
            else:
                return
        # 如果当前batch已经存满，则对这些图片进行评估
        bi += 1
        if bi == args.batch_size:
            batch_res = eval_batch_scales(args, eval_net, batch_img_lst, scales=args.scales,
                                          base_crop_size=args.crop_size, flip=args.flip)
            # 计算当前batch的混淆矩阵，并将其累加到hist中
            for mi in range(args.batch_size):
                hist += cal_hist(batch_msk_lst[mi].flatten(), batch_res[mi].flatten(), args.num_classes)

            bi = 0
            batch_img_lst = []
            batch_msk_lst = []
            # 每处理100张图像，输出处理进度
            if (i+1)%100 == 0:
                print('processed {} images'.format(i+1))
        image_num = i
    # 如果batch中还有剩余的图片，则对其进行评估
    if bi > 0:
        batch_res = eval_batch_scales(args, eval_net, batch_img_lst, scales=args.scales,
                                      base_crop_size=args.crop_size, flip=args.flip)
        # 计算当前batch的混淆矩阵，并将其累加到hist中
        for mi in range(bi):
            hist += cal_hist(batch_msk_lst[mi].flatten(), batch_res[mi].flatten(), args.num_classes)
        if (i+1) % 100 == 0:
            print('processed {} images'.format(image_num + 1))
    # 计算每个类别的IoU，求其平均值作为结果输出
    np.seterr(divide="ignore", invalid="ignore")
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print('mean IoU', np.nanmean(iu))

## 验证模型

# test  1
# 定义配置文件
cfg = edict({
    "batch_size": 1, # 批大小
    "crop_size": 513, # 图像裁剪大小
    "image_mean": [103.53, 116.28, 123.675], # RGB图像为3通道，定义图像RGB均值
    "image_std": [57.375, 57.120, 58.395], # RGB图像为3通道，定义图像RGB标准差
    "scales": [1.0], # 数据增强的尺度范围
    # [0.5,0.75,1.0,1.25,1.75]
    'flip': True, # 是否进行翻转
    'ignore_label': 255, # 忽略标签的像素值
    'num_classes':21, # 类别数，此处为PASCAL VOC数据集
    'model': 'deeplab_v3_s8', # 模型名称
    'freeze_bn': True, # BN冻结，是否使用BN层

    'if_png':False,        # 数据是否使用png格式
    'num_png':10           # 数据集中的png图像数量
})

# 数据集路径
data_path = './seg2'
# if not os.path.exists(data_path):
#mox.file.copy_parallel(src_url="s3://share-course/dataset/voc2012_raw/", dst_url=data_path)
cfg.data_file = data_path

# 定义数据集
dataset = SegDataset(image_mean=cfg.image_mean,
                     image_std=cfg.image_std,
                     data_file=cfg.data_file)
# 获取灰度图像数据集
dataset.get_gray_dataset()
# 数据集的txt标签路径
cfg.data_lst = os.path.join(cfg.data_file,'val.txt')
# VOC数据集中的图像路径
cfg.voc_img_dir = os.path.join(cfg.data_file,'JPEG')
# VOC数据集中的标签图像路径
cfg.voc_anno_gray_dir = os.path.join(cfg.data_file,'SegmentationClassGray')

ckpt_path = './ckpt'
# if not os.path.exists(ckpt_path):
#     mox.file.copy_parallel(src_url="s3://yyq-3/DATA/code/deeplabv3/model", dst_url=ckpt_path)   #if yours model had saved
# 加载模型文件路径
cfg.ckpt_file = os.path.join(ckpt_path,'deeplab_v3_s8-300_11.ckpt')
print('loading checkpoing:',cfg.ckpt_file)
# 模型验证评估
net_eval(cfg)


# test 2
cfg = edict({
    "batch_size": 1, # 批大小
    "crop_size": 513, # 图像裁剪大小
    "image_mean": [103.53, 116.28, 123.675], # RGB图像为3通道，定义图像RGB均值
    "image_std": [57.375, 57.120, 58.395], # RGB图像为3通道，定义图像RGB标准差
    "scales": [1.0], # 数据增强的尺度范围
    # [0.5,0.75,1.0,1.25,1.75]
    'flip': True, # 是否进行翻转
    'ignore_label': 255, # 忽略标签的像素值
    'num_classes':21, # 类别数，此处为PASCAL VOC数据集
    'model': 'deeplab_v3_s8', # 模型名称
    'freeze_bn': True, # BN冻结，是否使用BN层

    'if_png':True,         # 图像数据是否使用png格式
    'num_png':3            # png图像数量
})



# import moxing as mox
data_path = './seg2'
# if not os.path.exists(data_path):
#     mox.file.copy_parallel(src_url="s3://share-course/dataset/voc2012_raw/", dst_url=data_path)
# 数据集路径
cfg.data_file = data_path

# 定义数据集
dataset = SegDataset(image_mean=cfg.image_mean,
                     image_std=cfg.image_std,
                     data_file=cfg.data_file)
dataset.get_gray_dataset()
# 数据集的txt标签路径
cfg.data_lst = os.path.join(cfg.data_file,'val.txt')
# VOC数据集中的图像路径
cfg.voc_img_dir = os.path.join(cfg.data_file,'JPEG')
# VOC数据集中的标签图像路径
cfg.voc_anno_gray_dir = os.path.join(cfg.data_file,'SegmentationClassGray')

ckpt_path = './ckpt'
# if not os.path.exists(ckpt_path):
#     mox.file.copy_parallel(src_url="s3://yyq-3/DATA/code/deeplabv3/model", dst_url=ckpt_path)     #if yours model had saved
cfg.ckpt_file = os.path.join(ckpt_path,'deeplab_v3_s8-300_11.ckpt') \
    # 加载模型文件路径
print('loading checkpoing:',cfg.ckpt_file)
# 模型验证评估
net_eval(cfg)