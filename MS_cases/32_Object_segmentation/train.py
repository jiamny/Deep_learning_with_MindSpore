# 基于MindSpore实现目标分割


import os
import mindspore
from mindspore import context
device_id = int(0)
context.set_context(mode=context.GRAPH_MODE, device_target="GPU", save_graphs=False)
mindspore.set_seed(1)

from dataset import create_dataset, show_image

# 模型构建

## 自定义评估指标

from mindspore import nn
import mindspore.numpy as np

# import mindspore.ops as F
# 创建包含两个卷积层和归一化层的序列模块
def double_conv(in_ch, out_ch):
    return nn.SequentialCell(nn.Conv2d(in_ch, out_ch, 3),        # 第一个卷积层，输入通道数为 in_ch，输出通道数为 out_ch，卷积核大小为 3x3
                             nn.BatchNorm2d(out_ch), nn.ReLU(), # 归一化层，对输出通道进行归一化；使用ReLU 激活函数
                             nn.Conv2d(out_ch, out_ch, 3),      # 第二个卷积层，输入通道数为 out_ch，输出通道数为 out_ch，卷积核大小为 3x3
                             nn.BatchNorm2d(out_ch), nn.ReLU()) # 归一化层，对输出通道进行归一化；使用ReLU 激活函数

class UNet(nn.Cell):
    def __init__(self, in_ch = 3, n_classes = 1):
        super(UNet, self).__init__()
        # 定义 U-Net 模型的各个组件
        self.double_conv1 = double_conv(in_ch, 64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.double_conv2 = double_conv(64, 128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.double_conv3 = double_conv(128, 256)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.double_conv4 = double_conv(256, 512)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.double_conv5 = double_conv(512, 1024)
        self.upsample1 = nn.ResizeBilinear()
        self.double_conv6 = double_conv(1024 + 512, 512)
        self.upsample2 = nn.ResizeBilinear()
        self.double_conv7 = double_conv(512 + 256, 256)
        self.upsample3 = nn.ResizeBilinear()
        self.double_conv8 = double_conv(256 + 128, 128)
        self.upsample4 = nn.ResizeBilinear()
        self.double_conv9 = double_conv(128 + 64, 64)
        self.final = nn.Conv2d(64, n_classes, 1)
        self.sigmoid = ops.sigmoid

    def construct(self, x):
        # U-Net 模型的前向传播逻辑
        feature1 = self.double_conv1(x)
        tmp = self.maxpool1(feature1)
        feature2 = self.double_conv2(tmp)
        tmp = self.maxpool2(feature2)
        feature3 = self.double_conv3(tmp)
        tmp = self.maxpool3(feature3)
        feature4 = self.double_conv4(tmp)
        tmp = self.maxpool4(feature4)
        feature5 = self.double_conv5(tmp)
        up_feature1 = self.upsample1(feature5, scale_factor=2)
        #up_feature1 = self.upsample1(feature5)
        tmp = ops.concat((feature4, up_feature1),axis=1)
        tmp = self.double_conv6(tmp)
        up_feature2 = self.upsample2(tmp, scale_factor=2)
        #up_feature2 = self.upsample2(tmp)
        tmp = ops.concat((feature3, up_feature2),axis=1)
        tmp = self.double_conv7(tmp)
        up_feature3 = self.upsample3(tmp, scale_factor=2)
        #up_feature3 = self.upsample3(tmp)
        tmp = ops.concat((feature2, up_feature3),axis=1)
        tmp = self.double_conv8(tmp)
        up_feature4 = self.upsample4(tmp, scale_factor=2)
        #up_feature4 = self.upsample4(tmp)
        tmp = ops.concat((feature1, up_feature4),axis=1)
        tmp = self.double_conv9(tmp)
        output = self.sigmoid(self.final(tmp))
        return output

import numpy as np
from mindspore import Metric
from mindspore import Tensor
from mindspore._checkparam import check_positive_float

class metrics_(Metric):
    # 初始化
    def __init__(self, metrics, smooth=1e-5):
        super(metrics_, self).__init__()
        self.metrics = metrics
        self.smooth = check_positive_float(smooth, "smooth")
        self.metrics_list = [0. for i in range(len(self.metrics))]
        self._samples_num = 0
        self.clear()
    # 计算准确率指标
    def Acc_metrics(self,y_pred, y):
        tp = np.sum(y_pred.flatten() == y.flatten(), dtype=y_pred.dtype)
        total = len(y_pred.flatten())
        single_acc = float(tp) / float(total)
        return single_acc
    # 计算 IoU (Intersection over Union) 指标
    def IoU_metrics(self,y_pred, y):
        intersection = np.sum(y_pred.flatten() * y.flatten())
        unionset = np.sum(y_pred.flatten() + y.flatten()) - intersection
        single_iou = float(intersection) / float(unionset + self.smooth)
        return single_iou
    # 计算 Dice 系数指标
    def Dice_metrics(self,y_pred, y):
        intersection = np.sum(y_pred.flatten() * y.flatten())
        unionset = np.sum(y_pred.flatten()) + np.sum(y.flatten())
        single_dice = 2*float(intersection) / float(unionset + self.smooth)
        return single_dice
    # 计算敏感性指标
    def Sens_metrics(self,y_pred, y):
        tp = np.sum(y_pred.flatten() * y.flatten())
        actual_positives = np.sum(y.flatten())
        single_sens = float(tp) / float(actual_positives + self.smooth)
        return single_sens
    # 计算特异性指标
    def Spec_metrics(self,y_pred, y):
        true_neg = np.sum((1 - y.flatten()) * (1 - y_pred.flatten()))
        total_neg = np.sum((1 - y.flatten()))
        single_spec = float(true_neg) / float(total_neg + self.smooth)
        return single_spec
    # 清空内部的评估结果
    def clear(self):
        """Clears the internal evaluation result."""
        self.metrics_list = [0. for i in range(len(self.metrics))]
        self._samples_num = 0
    # 更新评估结果
    def update(self, *inputs):

        if len(inputs) != 2:
            raise ValueError("For 'update', it needs 2 inputs (predicted value, true value), ""but got {}.".format(len(inputs)))

        y_pred = Tensor(inputs[0]).asnumpy()  # 将输入的预测值转换为NumPy数组
        # y_pred = np.array(Tensor(inputs[0]))  #cpu

        y_pred[y_pred > 0.5] = float(1)       # 将预测值大于0.5的部分设置为1
        y_pred[y_pred <= 0.5] = float(0)      # 将预测值小于等于0.5的部分设置为0

        y = Tensor(inputs[1]).asnumpy()       # 将输入的真实值转换为NumPy数组
        # y = np.array(Tensor(inputs[1]))     #cpu

        self._samples_num += y.shape[0]

        if y_pred.shape != y.shape:
            raise ValueError(f"For 'update', predicted value (input[0]) and true value (input[1]) "
                             f"should have same shape, but got predicted value shape: {y_pred.shape}, "
                             f"true value shape: {y.shape}.")

        for i in range(y.shape[0]):
            if "acc" in self.metrics:
                single_acc = self.Acc_metrics(y_pred[i], y[i])
                self.metrics_list[0] += single_acc
            if "iou" in self.metrics:
                single_iou = self.IoU_metrics(y_pred[i], y[i])
                self.metrics_list[1] += single_iou
            if "dice" in self.metrics:
                single_dice = self.Dice_metrics(y_pred[i], y[i])
                self.metrics_list[2] += single_dice
            if "sens" in self.metrics:
                single_sens = self.Sens_metrics(y_pred[i], y[i])
                self.metrics_list[3] += single_sens
            if "spec" in self.metrics:
                single_spec = self.Spec_metrics(y_pred[i], y[i])
                self.metrics_list[4] += single_spec
    # 评估模型性能并返回评估指标结果
    def eval(self):
        if self._samples_num == 0:
            raise RuntimeError("The 'metrics' can not be calculated, because the number of samples is 0, "
                               "please check whether your inputs(predicted value, true value) are empty, or has "
                               "called update method before calling eval method.")
        for i in range(len(self.metrics_list)):
            self.metrics_list[i] = self.metrics_list[i] / float(self._samples_num)

        return self.metrics_list
    #样本点
x = Tensor(np.array([[[[0.2, 0.5, 0.7], [0.3, 0.1, 0.2], [0.9, 0.6, 0.8]]]]))
y = Tensor(np.array([[[[0, 1, 1], [1, 0, 0], [0, 1, 1]]]]))
#实例化了metrics_类，传入评估指标列表["acc", "iou", "dice", "sens", "spec"]和平滑参数smooth=1e-5
metric = metrics_(["acc", "iou", "dice", "sens", "spec"],smooth=1e-5)
#调用clear方法清除之前的评估结果
metric.clear()
#更新评估指标
metric.update(x, y)
#返回最终的评估结果
res = metric.eval()
print( '丨acc: %.4f丨丨iou: %.4f丨丨dice: %.4f丨丨sens: %.4f丨丨spec: %.4f丨' % (res[0], res[1], res[2], res[3],res[4]), flush=True)

# 模型训练及评估

import mindspore.nn as nn
from mindspore import ops
import mindspore as ms
from mindspore import jit
import ml_collections
from mindspore import load_checkpoint

def get_config():
    # 定义模型参数
    config = ml_collections.ConfigDict()
    config.epochs = 500  # 训练的轮数
    config.train_data_path = "data/ISBI/train/"  # 训练数据集路径F
    config.val_data_path = "data/ISBI/val/"      # 验证数据集路径
    config.imgsize = 224   # 图像尺寸
    config.batch_size = 4  # 批大小
    config.pretrained_path = None  # 预训练模型路径
    config.in_channel = 3  # 输入通道数
    config.n_classes = 1   # 类别数
    config.lr = 0.0001     # 学习率
    return config
cfg = get_config()

#获取训练集和验证集
train_dataset = create_dataset(cfg.train_data_path, img_size=cfg.imgsize, batch_size= cfg.batch_size, augment=True, shuffle = True)
val_dataset = create_dataset(cfg.val_data_path, img_size=cfg.imgsize, batch_size= cfg.batch_size, augment=False, shuffle = False)


def train(model, dataset, loss_fn, optimizer, met):
    # 定义前向传播函数
    def forward_fn(data, label):
        logits = model(data)
        loss = loss_fn(logits, label)
        return loss, logits
    # 计算梯度
    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
    # 定义一步训练
    @jit
    def train_step(data, label):
        (loss, logits), grads = grad_fn(data, label)
        loss = ops.depend(loss, optimizer(grads))
        return loss, logits

    size = dataset.get_dataset_size()   # 获取数据集的大小（样本数量）
    model.set_train(True)               # 设置模型为训练模式

    train_loss = 0                      # 训练损失的累加和
    train_pred = []                     # 存储训练预测结果
    train_label = []                    # 存储训练标签
    # 遍历数据集的每个批次
    for batch, (data, label) in enumerate(dataset.create_tuple_iterator()):
        loss, logits = train_step(data, label)     # 调用训练步骤函数进行模型训练，返回损失和预测结果
        train_loss += loss.asnumpy()               # 将损失值累加到总和中
        train_pred.extend(logits.asnumpy())        # 将预测结果添加到训练预测列表中
        train_label.extend(label.asnumpy())        # 将标签添加到训练标签列表中

    train_loss /= size                             # 计算平均训练损失
    metric = metrics_(met, smooth=1e-5)            # 创建评估指标对象
    metric.clear()                                 # 清除评估指标的状态
    metric.update(train_pred, train_label)         # 更新评估指标，传入训练预测结果和标签
    res = metric.eval()                            # 计算评估指标的结果
    # 打印训练损失和评估指标的结果
    print(f'Train loss:{train_loss:>4f}','丨acc: %.3f丨丨iou: %.3f丨丨dice: %.3f丨丨sens: %.3f丨丨spec: %.3f丨' % (res[0], res[1], res[2], res[3], res[4]))


def val(model, dataset, loss_fn, met):
    size = dataset.get_dataset_size()  # 获取数据集的大小（样本数量）
    model.set_train(False)             # 设置模型为验证模式
    val_loss = 0                       # 验证损失的累加和
    val_pred = []                      # 存储验证预测结果
    val_label = []                     # 存储验证标签

    # 遍历数据集的每个批次
    for batch, (data, label) in enumerate(dataset.create_tuple_iterator()):
        pred = model(data)                          #使用模型进行预测
        val_loss += loss_fn(pred, label).asnumpy()  # 计算验证损失并累加到总和中
        val_pred.extend(pred.asnumpy())             # 将预测结果添加到验证预测列表中
        val_label.extend(label.asnumpy())           # 将标签添加到验证标签列表中

    val_loss /= size                      # 计算平均验证损失
    metric = metrics_(met, smooth=1e-5)   # 创建评估指标对象
    metric.clear()                        # 清除评估指标的状态
    metric.update(val_pred, val_label)    # 更新评估指标，传入验证预测结果和标签
    res = metric.eval()                   # 计算评估指标的结果
    # 打印验证损失和评估指标的结果
    print(f'Val loss:{val_loss:>4f}','丨acc: %.3f丨丨iou: %.3f丨丨dice: %.3f丨丨sens: %.3f丨丨spec: %.3f丨' % (res[0], res[1], res[2], res[3], res[4]))

    checkpoint = res[1]                    # 选择保存检查点的指标，此处为acc（准确率）
    return checkpoint, res[4]              # 返回用于判断是否保存检查点的指标和用于比较模型性能的指标


if __name__ == '__main__':
    net = UNet(cfg.in_channel, cfg.n_classes)                                # 创建UNet模型，输入通道数为cfg.in_channel，类别数为cfg.n_classes
    criterion = nn.BCEWithLogitsLoss()                                       # 创建二分类交叉熵损失函数

    if os.path.isfile("checkpoint/best_UNet_own.ckpt"):
        load_checkpoint("checkpoint/best_UNet_own.ckpt", net)   # 加载已有的模型参数

    parameters = net.final.get_parameters()                     # 只优化最后一层参数

    optimizer = nn.SGD(params=parameters, learning_rate=cfg.lr)  # 创建随机梯度下降优化器，传入可训练参数和学习率

    iters_per_epoch = train_dataset.get_dataset_size()  # 获取每个epoch的迭代次数
    total_train_steps = iters_per_epoch * cfg.epochs    # 计算总的训练步数
    print('iters_per_epoch: ', iters_per_epoch)         # 打印每个epoch的迭代次数
    print('total_train_steps: ', total_train_steps)     # 打印总的训练步数

    metrics_name = ["acc", "iou", "dice", "sens", "spec"]  # 定义评估指标的名称

    best_iou = 0                                           # 初始化最佳iou为0
    if not os.path.exists("checkpoint"):
        os.mkdir("./checkpoint")

    ckpt_path = 'checkpoint/best_UNet.ckpt'                # 设置保存最佳模型的路径

    for epoch in range(cfg.epochs):                 # 遍历每个epoch
        print(f"Epoch [{epoch+1} / {cfg.epochs}]")  # 打印当前epoch的信息
        train(net, train_dataset, criterion, optimizer, metrics_name)           # 在训练集上进行训练
        checkpoint_best, spec = val(net, val_dataset, criterion, metrics_name)  # 在验证集上进行评估，获取最佳检查点和特异性指标

        if epoch > 2 and spec > 0.2:                                                     # 如果当前epoch大于2且特异性指标大于0.2
            if checkpoint_best > best_iou:                                               # 如果最佳检查点的交并比大于当前最佳交并比
                print('IoU improved from %0.4f to %0.4f' % (best_iou, checkpoint_best))  # 打印交并比改善的信息
                best_iou = checkpoint_best                                     # 更新最佳交并比
                mindspore.save_checkpoint(net, ckpt_path)                      # 保存最佳模型的检查点
                print("saving best checkpoint at: {} ".format(ckpt_path))      # 打印保存检查点的路径
            else:
                print('IoU did not improve from %0.4f' % (best_iou),"\n-------------------------------")  # 打印交并比未改善的信息
    print("Done!")  # 打印训练完成的信息
    exit(0)


