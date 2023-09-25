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

import ast
import argparse
from config import alexnet_cfg as cfg
from alexnet import AlexNet
from generator_lr import get_lr
import mindspore.dataset as ds
import mindspore.nn as nn
from mindspore import Tensor, context, load_checkpoint, load_param_into_net, Model, set_seed
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
import mindspore.dataset.transforms as C
import mindspore.dataset.vision as CV
from mindspore.nn import Accuracy
from mindspore import dtype as mstype
import matplotlib.pyplot as plt
import numpy as np


def create_dataset(data_path, batch_size=32, repeat_size=1, mode="train", sample_mum=None):
    """
    create dataset for train or test
    """
    cifar_ds = ds.Cifar10Dataset(data_path, num_samples=sample_mum)
    rescale = 1.0 / 255.0
    shift = 0.0

    resize_op = CV.Resize((cfg.image_height, cfg.image_width))
    rescale_op = CV.Rescale(rescale, shift)
    normalize_op = CV.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    if mode == "train":
        random_crop_op = CV.RandomCrop([32, 32], [4, 4, 4, 4])
        random_horizontal_op = CV.RandomHorizontalFlip()
    channel_swap_op = CV.HWC2CHW()
    typecast_op = C.TypeCast(mstype.int32)
    cifar_ds = cifar_ds.map(operations=typecast_op, input_columns="label")
    if mode == "train":
        cifar_ds = cifar_ds.map(operations=random_crop_op, input_columns="image")
        cifar_ds = cifar_ds.map(operations=random_horizontal_op, input_columns="image")
    cifar_ds = cifar_ds.map(operations=resize_op, input_columns="image")
    cifar_ds = cifar_ds.map(operations=rescale_op, input_columns="image")
    cifar_ds = cifar_ds.map(operations=normalize_op, input_columns="image")
    cifar_ds = cifar_ds.map(operations=channel_swap_op, input_columns="image")

    cifar_ds = cifar_ds.shuffle(buffer_size=cfg.buffer_size)
    cifar_ds = cifar_ds.batch(batch_size, drop_remainder=True)
    cifar_ds = cifar_ds.repeat(repeat_size)
    return cifar_ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MindSpore AlexNet Example')
    parser.add_argument('--train_samples', type=int, default=None, help='number train images')
    parser.add_argument('--test_samples', type=int, default=None, help='number test images')
    parser.add_argument('--device_target', type=str, default="GPU", choices=['Ascend', 'GPU'],
                        help='device where the code will be implemented (default: Ascend)')
    parser.add_argument('--data_path', type=str, default="../../data/cifar-10-batches-bin",
                        help='path where the dataset is saved')
    parser.add_argument('--ckpt_path', type=str, default="./ckpt", help='if mode is test, must provide\
                        path where the trained ckpt file')
    parser.add_argument('--checkpoint_file', type=str, default="./ckpt/checkpoint_cifar-5_20.ckpt",
                        help='Checkpoint file path and name.')
    parser.add_argument('--dataset_sink_mode', type=ast.literal_eval, default=True,
                        help='dataset_sink_mode is False or True')

    args = parser.parse_args()

    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    set_seed(1234)

    network = AlexNet(cfg.num_classes)
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    repeat_size = 1

    ds_train = create_dataset(args.data_path,
                              cfg.batch_size,
                              repeat_size,
                              "train", args.train_samples)

    batch_num = ds_train.get_dataset_size()
    print(batch_num)  # batch steps

    # when batch_size=128, steps is 100
    lr = Tensor(get_lr(0, cfg.learning_rate, cfg.epoch_size, batch_num))
    opt = nn.Momentum(network.trainable_params(), lr, cfg.momentum)
    model = Model(network, loss, opt, metrics={"Accuracy": Accuracy()})  # test

    ds_eval = create_dataset(args.data_path,
                             mode="test", sample_mum=args.test_samples)
    data_next = ds_eval.create_dict_iterator(output_numpy=True).__next__()
    print('通道数/图像长/宽：', data_next['image'].shape)
    print('一张图像的标签样式：', data_next['label'])  # 一共5类，用0-4的数字表达类别。

    print(data_next['image'][0].shape)
    print(data_next['label'][0])

    img_cv = Tensor(data_next['image'][0]).permute((1, 2, 0)).asnumpy()
    img_plt = img_cv.astype(np.uint8)
    plt.figure()
    plt.imshow(img_plt)
    plt.colorbar()
    plt.grid(False)
    plt.show()

    trflag = True

    if trflag:
        print("============== Starting Training ==============")
        config_ck = CheckpointConfig(save_checkpoint_steps=batch_num,  # cfg.save_checkpoint_steps,
                                     keep_checkpoint_max=cfg.keep_checkpoint_max)

        ckpoint_cb = ModelCheckpoint(prefix="checkpoint_cifar", directory=args.ckpt_path, config=config_ck)

        model.train(cfg.epoch_size, ds_train, callbacks=[ckpoint_cb, LossMonitor()],
                    dataset_sink_mode=args.dataset_sink_mode)
    else:
        for data in ds_train.create_dict_iterator():
            print(data['image'].shape)
            print(data['label'])
            print('------------')
            break

    if trflag:
        print("============== Starting Testing ==============")
        param_dict = load_checkpoint(args.checkpoint_file)
        load_param_into_net(network, param_dict)
        ds_eval = create_dataset(args.data_path,
                                 mode="test", sample_mum=args.test_samples)
        acc = model.eval(ds_eval, dataset_sink_mode=args.dataset_sink_mode)
        print("============== Accuracy:{} ==============".format(acc))

    exit(0)
