# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
AlexNet example tutorial
Usage:
     python alexnet.py
with --device_target=GPU: After 20 epoch training, the accuracy is up to 80%
with --device_target=Ascend: After 30 epoch training, the accuracy is up to 88%
"""
import copy
import os
import numpy as np
import cv2
import ast
import argparse
from alexnet import AlexNet
from generator_lr import get_lr
import mindspore.dataset as ds
import mindspore.nn as nn
from mindspore import Tensor, context, load_checkpoint, load_param_into_net, Model, set_seed, Callback, save_checkpoint, \
    TimeMonitor
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
import mindspore.dataset.transforms as C
import mindspore.dataset.vision as CV
from mindspore.nn import Accuracy
from mindspore import dtype as mstype
from easydict import EasyDict as edict
import matplotlib.pyplot as plt
from EvalCallBack import apply_eval, EvalCallBack

'''
变量定义
'''
cfg = edict({
    'data_path': '../../data/flower_photos',
    'data_size': 1600,
    'image_width': 227,  # 图片宽度
    'image_height': 227,  # 图片高度
    'batch_size': 2,
    'channel': 3,   # 图片通道数
    'num_class':5,  # 分类类别
    'weight_decay': 0.01,
    'lr':0.0001,    # 学习率
    'dropout_ratio': 0.5,
    'epoch_size': 200,  # 训练次数
    'sigma':0.01,

    'save_checkpoint_steps': 60,  # 多少步保存一次模型
    'keep_checkpoint_max': 2,  # 最多保存多少个模型
    'output_directory': './ckpt',  # 保存模型路径
    'output_prefix': "checkpoint_classification"  # 保存模型文件名字
})

def get_dataset(cfg, sample_num=None):
    '''
    读取并处理数据
    '''
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

    #划分训练集测试集
    (de_train,de_test)=de_dataset.split([0.85,0.15])

    #设置每个批处理的行数
    #drop_remainder确定是否删除最后一个可能不完整的批（default=False）。
    #如果为True，并且如果可用于生成最后一个批的batch_size行小于batch_size行，则这些行将被删除，并且不会传播到子节点。
    de_train=de_train.batch(cfg.batch_size, drop_remainder=True)
    #重复此数据集计数次数。
    de_test=de_test.batch(cfg.batch_size, drop_remainder=True)
    return de_train, de_test


if __name__ == "__main__":
    print( os.getcwd())

    parser = argparse.ArgumentParser(description='MindSpore AlexNet Example')
    parser.add_argument('--device_target', type=str, default="GPU", choices=['Ascend', 'GPU'],
                        help='device where the code will be implemented (default: Ascend)')
    parser.add_argument('--ckpt_path', type=str, default="./ckpt", help='if mode is test, must provide\
                        path where the trained ckpt file')
    parser.add_argument('--dataset_sink_mode', type=ast.literal_eval, default=True,
                        help='dataset_sink_mode is False or True')

    args = parser.parse_args()

    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    set_seed(1234)

    (de_train, de_test) = get_dataset(cfg, cfg.data_size)
    batch_num = de_train.get_dataset_size()
    print('batch_num：', batch_num)
    print('训练数据集数量：',de_train.get_dataset_size()*cfg.batch_size) #get_dataset_size()获取批处理的大小。
    print('测试数据集数量：',de_test.get_dataset_size()*cfg.batch_size)

    d_test = copy.deepcopy(de_test)
    data_next=d_test.create_dict_iterator(output_numpy=True).__next__()
    print('通道数/图像长/宽：', data_next['image'].shape)
    print('一张图像的标签样式：', data_next['label'])  # 一共5类，用0-4的数字表达类别。

    print(data_next['image'][0].shape)

    plt.figure()
    img_cv = Tensor(data_next['image'][0]).permute((1, 2, 0)).asnumpy()
    # -------------------------------------------------------------------
    # MindSpore use PIL read image file, it already in BGR format
    # not needs to convert form RGB to BGR
    # np.array(cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)).astype(np.uint8)

    img_plt = img_cv.astype(np.uint8)
    plt.imshow(img_plt)
    plt.colorbar()
    plt.grid(False)
    plt.show()

    network = AlexNet(cfg.num_class)

    x = Tensor(np.ones([1, 3, 227, 227]), dtype=mstype.float32)
    print('net(images).shape: ', network(x).shape, x.dtype)

    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")

    # when batch_size=32, steps is 14
    lr = Tensor(get_lr(0, cfg.lr, cfg.epoch_size, batch_num))
    print(lr)
    opt = nn.Momentum(network.trainable_params(), lr, 0.9)
    model = Model(network, loss, opt, metrics={"Accuracy": Accuracy()})  # test

    print("============== Starting Training ==============")       
    #config_ck = CheckpointConfig(save_checkpoint_steps=batch_num, #cfg.save_checkpoint_steps,
    #                                 keep_checkpoint_max=cfg.keep_checkpoint_max)

    #ckpoint_cb = ModelCheckpoint(prefix="checkpoint_flower", directory=args.ckpt_path, config=config_ck)

    #model.train(cfg.epoch_size, de_train, callbacks=[ckpoint_cb, LossMonitor()],
    #                dataset_sink_mode=args.dataset_sink_mode)

    eval_param_dict = {"model":model,"dataset":de_test,"metrics_name":"Accuracy"}
    eval_cb = EvalCallBack(apply_eval, eval_param_dict, interval=1, eval_start_epoch=1,
                           num_epochs=cfg.epoch_size,
                           save_best_ckpt=True, ckpt_directory=args.ckpt_path,
                           besk_ckpt_name="best.ckpt", metrics_name="acc")

    # 训练模型
    model.train(cfg.epoch_size,de_train, callbacks=[eval_cb, TimeMonitor()], dataset_sink_mode=True)

    # 使用测试集评估模型，打印总体准确率
    print("============== Starting Testing ==============")
    metric = model.eval(de_test)
    print(metric)

    '''
    param_dict = load_checkpoint(os.path.join(args.ckpt_path, "best.ckpt"))
    load_param_into_net(network, param_dict)
    acc = model.eval(de_test, dataset_sink_mode=args.dataset_sink_mode)
    print("============== Accuracy:{} ==============".format(acc))
    '''
    exit(0)