import mindspore as ms
import mindspore.dataset as ds
import mindspore.nn as nn
from mindspore import ops
import mindspore.dataset.transforms as C
import mindspore.dataset.vision as CV
from mindspore import dtype as mstype
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '../utils')
from plotResults import showLossAndAccuracy

from mindspore import context, set_seed
set_seed(0)
context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

train_data_dir = '/media/hhj/localssd/DL_data/DogsVSCats/train'
test_data_dir = '/media/hhj/localssd/DL_data/DogsVSCats/test'

EPOCH = 20
BATCH_SIZE = 8

def get_dataset(data_path, shuffle=False, batch_size=0):

    #从目录中读取图像的源数据集。
    de_dataset = ds.ImageFolderDataset(data_path, shuffle=shuffle, class_indexing={'cat':0,'dog':1})
    image_transforms = [
        #CV.RandomCropDecodeResize([224, 224], scale=(0.08, 1.0), ratio=(0.75, 1.333)),
        CV.Decode(),
        CV.Resize([224, 224]),
        # 基于给定的缩放和平移因子调整图像的像素大小。输出图像的像素大小为：output = image * rescale + shift。
        # 此处rescale取1.0 / 255.0，shift取0
        CV.Rescale(1.0 / 255.0, 0),
        # 将输入图像的shape从 <H, W, C> 转换为 <C, H, W>
        CV.HWC2CHW(),
    ]

    label_transform = C.TypeCast(mstype.int32)

    #将操作中的每个操作应用到此数据集。
    de_dataset = de_dataset.map(input_columns="image", num_parallel_workers=2, operations=image_transforms)
    de_dataset = de_dataset.map(input_columns="label", operations=label_transform, num_parallel_workers=2)
    de_dataset = de_dataset.shuffle(buffer_size=len(de_dataset))

    #drop_remainder确定是否删除最后一个可能不完整的批（default=False）。
    #如果为True，并且如果可用于生成最后一个批的batch_size行小于batch_size行，则这些行将被删除，并且不会传播到子节点。

    if batch_size == 0: # load all data points
        de_dl = de_dataset.batch(len(de_dataset))
    else:
        de_dl = de_dataset.batch(batch_size, drop_remainder=False)
    return de_dl


train_dataset = get_dataset(train_data_dir, shuffle=True, batch_size=BATCH_SIZE)
test_dataset = get_dataset(test_data_dir, shuffle=True)
images, label = train_dataset.create_tuple_iterator().__next__()
print(images.shape, ' ', label.shape)
print(label)
im = ((images[2]*255).permute((1,2,0)).asnumpy()).astype(np.uint8)
print(type(im))
plt.imshow(im)
plt.show()
print(label[2])


def conv_layer(ni,no,kernel_size,stride=1):
    return nn.SequentialCell(
        nn.Conv2d(ni, no, kernel_size, stride),
        nn.ReLU(),
        nn.BatchNorm2d(no),
        nn.MaxPool2d(kernel_size=2,stride=2)
    )


model = nn.SequentialCell(
    conv_layer(3, 64, 3),
    conv_layer(64, 128, 3),
    conv_layer(128, 256 , 3),
    conv_layer(256, 512 , 3),
    conv_layer(512, 512 , 3),
    nn.Flatten(),
    nn.Dense(512*7*7, 1),
    nn.Sigmoid()
)

X = ops.randn(8, 3, 224, 224)
for blk in model:
    X = blk(X)
    print(blk.__class__.__name__,'output shape:\t',X.shape)

loss_fn = nn.BCELoss()
optimizer = nn.Adam(model.trainable_params(), learning_rate= 1e-3)

def accuracy(model, dataset, loss_fn):
    num_batches = dataset.get_dataset_size()
    model.set_train(False)
    total, test_loss, correct = 0, 0, 0
    for data, label in dataset.create_tuple_iterator():
        pred = model(data)
        total += len(data)
        pred = pred.flatten()
        test_loss += loss_fn(pred, label.astype(ms.float32)).asnumpy()
        correct += ((pred > 0.5) == label).asnumpy().sum()
    test_loss /= num_batches
    correct /= total
    return correct, test_loss

def train(model, dataset, loss_fn, optimizer):

    def forward_fn(data, label):
        pred = model(data)
        pred = pred.flatten()
        loss = loss_fn(pred, label.astype(ms.float32))
        return loss, pred

    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    def train_step(data, label):
        (loss, _), grads = grad_fn(data, label)
        # 获得损失 depend用来处理操作间的依赖关系
        loss = ops.depend(loss, optimizer(grads))
        return loss

    model.set_train()
    train_epoch_losses, train_epoch_accuracies = [], []

    for batch, (data, label) in enumerate(dataset.create_tuple_iterator()):
        # 批量训练获得损失值
        loss = train_step(data, label)
        train_epoch_losses.append(float(loss.item()))

    train_epoch_loss = np.array(train_epoch_losses).mean()
    return train_epoch_loss


print('*'*40, ' Model with all data points ', '*'*40)
train_losses, train_accuracies = [], []
val_losses, val_accuracies = [], []

for t in range(EPOCH):
    print(f"Epoch {t+1} ", '-'*80)
    loss = train(model, train_dataset, loss_fn, optimizer)
    train_losses.append(loss)
    accu, _ = accuracy(model, train_dataset, loss_fn)
    train_accuracies.append(accu)
    test_accu, test_loss = accuracy(model, test_dataset, loss_fn)
    val_losses.append(test_loss)
    val_accuracies.append(test_accu)
    print(f"Train Avg loss: {loss:>8f} Train accuracy: {(accu*100):>0.1f}% \
    Test Avg loss: {test_loss:>8f} Test accuracy: {(100*test_accu):>0.1f}%\n")

#showLossAndAccuracy(train_losses, train_accuracies, val_losses, val_accuracies, EPOCH, 'no regular')

import matplotlib.ticker as mticker
epochs = np.arange(EPOCH)+1
plt.plot(epochs, train_accuracies, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracies, 'r', label='Validation accuracy')
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
plt.title('Training and validation accuracy \nwith 4K total images used for training')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
yl = [x for x in plt.gca().get_yticks()]
plt.gca().set_yticks(yl)
plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in yl])
plt.legend()
plt.grid('off')
plt.show()