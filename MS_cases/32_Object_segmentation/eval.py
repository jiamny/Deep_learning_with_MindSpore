import os
import mindspore
import mindspore.numpy as np
import mindspore.dataset.transforms as C_transforms
import mindspore.dataset.vision as vision_C
import glob
from mindspore.dataset.vision import Inter
import cv2
import mindspore.dataset as ds
import random, math
import matplotlib.pyplot as plt

from train import metrics_, UNet


def show_image(image_list,num = 6):
    img_titles = []
    img_draws = []
    for ind,img in enumerate(image_list):
        if ind == num:
            break
        img_titles.append(ind)
        img_draws.append(mindspore.Tensor(img).asnumpy())

    for i in range(len(img_titles)):
        if len(img_titles) > 6:
            row = 3
        elif 3<len(img_titles)<=6:
            row = 2
        else:
            row = 1
        col = math.ceil(len(img_titles)/row)
        plt.subplot(row, col, i+1), plt.imshow(img_draws[i],'gray')
        plt.title(img_titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()


# 模型预测

from tqdm import tqdm
def val_transforms(img_size):
    return C_transforms.Compose([
        vision_C.Resize(img_size, interpolation=Inter.NEAREST),  # 调整图像大小为指定的img_size，插值方式为最近邻插值
        vision_C.Rescale(1/255., 0),  # 将像素值缩放到范围 [0, 1]，将输入图像除以255
        vision_C.HWC2CHW()  # 将图像的通道维度从 "HWC"（高度、宽度、通道）顺序转换为 "CHW"（通道、高度、宽度）顺序
    ])
class Data_Loader:
    def __init__(self, data_path, have_mask):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path  # 数据集路径
        self.have_mask = have_mask  # 是否有掩码
        self.imgs_path = glob.glob(os.path.join(data_path, 'image/*.png'))    # 获取所有图像文件的路径
        if self.have_mask:
            self.label_path = glob.glob(os.path.join(data_path, 'mask/*.png')) # 获取所有标签文件的路径

    def __getitem__(self, index):
        # 根据index读取图片
        image = cv2.imread(self.imgs_path[index])# 读取图像
        if self.have_mask:
            label = cv2.imread(self.label_path[index], cv2.IMREAD_GRAYSCALE)# 读取灰度图像标签
            label = label.reshape((label.shape[0], label.shape[1], 1))      # 将标签的形状调整为 (H, W, 1)
        else:
            label = image                        # 如果没有标签，则将图像作为标签
        return image, label

    @property
    def column_names(self):
        column_names = ['image', 'label']        # 定义列名
        return column_names

    def __len__(self):
        return len(self.imgs_path)               # 返回数据集的长度


def create_dataset(data_dir, img_size, batch_size, shuffle, have_mask=False):
    mc_dataset = Data_Loader(data_path=data_dir, have_mask=have_mask)  # 创建Data_Loader对象，加载数据集
    print(len(mc_dataset))                                             # 打印数据集中的图像文件数量
    # 创建GeneratorDataset对象，使用Data_Loader对象作为数据源
    dataset = ds.GeneratorDataset(mc_dataset, mc_dataset.column_names, shuffle=shuffle)
    transform_img = val_transforms(img_size)                           # 创建图像数据的转换操作
    seed = random.randint(1, 1000)                                     # 生成随机种子
    mindspore.set_seed(seed)                                           # 设置随机种子
    dataset = dataset.map(input_columns='image', num_parallel_workers=1, operations=transform_img)  # 对图像数据应用转换操作
    mindspore.set_seed(seed)                                           # 设置随机种子
    dataset = dataset.map(input_columns="label", num_parallel_workers=1, operations=transform_img)  # 对标签数据应用转换操作
    dataset = dataset.batch(batch_size, num_parallel_workers=1)        # 批量化数据
    return dataset                                                     # 返回创建的数据集对象

def model_pred(model, test_loader, result_path, have_mask):
    model.set_train(False)  # 设置模型为推理模式
    test_pred = []          # 存储预测结果
    test_label = []         # 存储标签数据
    for batch, (data, label) in enumerate(test_loader.create_tuple_iterator()):
        pred = model(data)  # 使用模型进行预测

        pred[pred > 0.5] = float(1)   # 将预测结果大于0.5的像素置为1
        pred[pred <= 0.5] = float(0)  # 将预测结果小于等于0.5的像素置为0

        preds = np.squeeze(pred, axis=0)      # 去除预测结果的批次维度
        img = np.transpose(preds, (1, 2, 0))  # 转换预测结果的通道维度顺序为"HWC"

        if not os.path.exists(result_path):
            os.makedirs(result_path)          # 创建保存结果的文件夹
        cv2.imwrite(os.path.join(result_path, "%04d.png" % batch), img.asnumpy() * 255.)  # 保存预测结果为图像文件

        test_pred.extend(pred.asnumpy())      # 将预测结果添加到test_pred列表中
        test_label.extend(label.asnumpy())    # 将标签数据添加到test_label列表中

    if have_mask:
        mtr = ['acc', 'iou', 'dice', 'sens', 'spec']  # 定义评估指标
        metric = metrics_(mtr, smooth=1e-5)           # 创建评估指标的计算对象
        metric.clear()                                # 清除评估指标的历史数据
        metric.update(test_pred, test_label)          # 更新评估指标计算结果
        res = metric.eval()                           # 获取评估指标的结果
        # 打印评估指标结果
        print(f'丨acc: %.3f丨丨iou: %.3f丨丨dice: %.3f丨丨sens: %.3f丨丨spec: %.3f丨' % (res[0], res[1], res[2], res[3], res[4]))
    else:
        # 如果没有标签数据，则无法计算评估指标
        print("Evaluation metrics cannot be calculated without Mask")

if __name__ == '__main__':
    # 创建一个UNet模型对象net，输入通道数为3，输出通道数为1
    net = UNet(3, 1)
    mindspore.load_checkpoint("checkpoint/best_UNet.ckpt", net=net)
    #保存预测结果路径为"predict"
    result_path = "predict"
    #创建一个测试数据集加载器test_dataset
    test_dataset = create_dataset("data/ISBI/val/", 224, 1, shuffle=False, have_mask=True)
    #根据net模型进行预测，并将预测结果保存在"predict"目录下
    model_pred(net, test_dataset, result_path, have_mask=True)

## 可视化预测结果
image_path = "data/ISBI/val/image/"
pred_path = "predict/"

image_list = os.listdir(image_path)     #读取测试图像
pred_list = os.listdir(pred_path)       #读取预测结果
print(image_list)
print()

#读取图像文件
test_image = np.array([cv2.imread(image_path + image_list[p], -1) for p in range(len(image_list))])
pred_masks = np.array([cv2.imread(pred_path + pred_list[p], -1) for p in range(len(pred_list))])

#显示测试图像和预测结果。num参数表示要显示的图像数量
show_image(test_image, num = 9)
show_image(pred_masks, num = 9)