import mindspore as ms
from mindspore import nn
from mindspore import ops
from mindspore.dataset import vision
from mindspore.dataset import transforms
import mindspore.dataset as ds
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../utils')
from plotResults import showLossAndAccuracy, displayImages
from fashion_mnist import _FashionMNIST

from mindspore import context, set_seed
set_seed(0)
context.set_context(mode=context.GRAPH_MODE, device_target="GPU", device_id=0)


data_folder = '/media/hhj/localssd/DL_data/fashion_MNIST'
tmnist = _FashionMNIST(data_folder, is_train=True)
tr_images, tr_targets = tmnist.get_data_and_label()

fmnist_classes = tmnist.classes()
print('fmnist_classes: ', fmnist_classes)

print(tr_images.dtype, ' ', tr_targets.dtype)

val_mnist = _FashionMNIST(data_folder, is_train=False)
val_images, val_targets = val_mnist.get_data_and_label()

def accuracy(model, dataset, loss_fn):
    num_batches = dataset.get_dataset_size()
    model.set_train(False)
    total, test_loss, correct = 0, 0, 0
    for data, label in dataset.create_tuple_iterator():
        pred = model(data)
        total += len(data)
        test_loss += loss_fn(pred, label).asnumpy()
        correct += (pred.argmax(1) == label).asnumpy().sum()
    test_loss /= num_batches
    correct /= total
    return correct, test_loss


LEARNING_RATE = 1e-3
EPOCH = 5
BATCH_SIZE = 16

def datapipe(dataset, batch_size):
    image_transforms = [
        # 基于给定的缩放和平移因子调整图像的像素大小。输出图像的像素大小为：output = image * rescale + shift。
        # 此处rescale取1.0 / 255.0，shift取0
        vision.Rescale(1.0 / 255.0, 0),
        # 将输入图像的shape从 <H, W, C> 转换为 <C, H, W>
        vision.HWC2CHW()
    ]

    label_transform = transforms.TypeCast(ms.int32)

    dataset = dataset.map(image_transforms, 'image')
    dataset = dataset.map(label_transform, 'label')

    # 将数据集中连续 batch_size 条数据组合为一个批数据
    dataset = dataset.batch(batch_size)
    return dataset

class FMNISTDataset:
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __getitem__(self, ix):
        x, y = self.x[ix], self.y[ix]
        return x, y

    def __len__(self):
        return len(self.x)


# Add new dimensions with np.newaxis
tr_images = tr_images[:,:,:,np.newaxis]
val_images = val_images[:,:,:,np.newaxis]
print(type(tr_images))
trn_dataset = ds.GeneratorDataset(FMNISTDataset(tr_images, tr_targets),
                                    column_names=["image", "label"],
                                    num_parallel_workers=1,
                                    shuffle=True,
                                    sampler=None)

val_dataset = ds.GeneratorDataset(FMNISTDataset(val_images, val_targets),
                                  column_names=["image", "label"],
                                  num_parallel_workers=1,
                                  shuffle=True,
                                  sampler=None)

train_dataset = datapipe(trn_dataset, BATCH_SIZE)
test_dataset = datapipe(val_dataset, BATCH_SIZE)

#images, label = (train_dataset.create_tuple_iterator()).__next__()
#displayImages(images, label)

model = nn.SequentialCell(
    nn.Conv2d(1, 64, kernel_size=3),
    nn.MaxPool2d(2),
    nn.ReLU(),
    nn.Conv2d(64, 128, kernel_size=3),
    nn.MaxPool2d(kernel_size=2,stride=1),
    nn.ReLU(),
    nn.Flatten(),
    nn.Dense(128*26*26, 256),
    nn.ReLU(),
    nn.Dense(256, 10)
)

'''
img = (tr_images[2430]/255.0).squeeze()
plt.imshow(img)
plt.show()
img = img.reshape([28, 28])
img2 = np.roll(img, -5, axis=1)
img3 = (ms.Tensor(img2).view(-1,1,28,28)).astype(ms.float32)
np_output = model(img3).asnumpy()
pred = np.exp(np_output)/np.sum(np.exp(np_output))
print(np_output.shape)
print(pred.argmax())
print(fmnist_classes)
print(fmnist_classes[pred[0].argmax()])
plt.imshow(img2)
plt.show()
'''

# ----------------------------------------------------------------
# summary of model
# ----------------------------------------------------------------
X = ops.randn(1, 1, 28, 28)
for blk in model:
    X = blk(X)
    print(blk.__class__.__name__,'output shape:\t',X.shape)

loss_fn2 = nn.CrossEntropyLoss()
optimizer2 = nn.Adam(model.trainable_params(), learning_rate=LEARNING_RATE)


def train(model, dataset, loss_fn, optimizer):

    def forward_fn(data, label):
        logits = model(data)
        loss = loss_fn(logits, label)
        return loss, logits

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


print('*'*40, ' Model with CNN ', '*'*40)
train_losses, train_accuracies = [], []
val_losses, val_accuracies = [], []

for t in range(EPOCH):
    print(f"Epoch {t+1} ", '-'*80)
    loss = train(model, train_dataset, loss_fn2, optimizer2)
    train_losses.append(loss)
    accu, _ = accuracy(model, train_dataset, loss_fn2)
    train_accuracies.append(accu)
    test_accu, test_loss = accuracy(model, test_dataset, loss_fn2)
    val_losses.append(test_loss)
    val_accuracies.append(test_accu)
    print(f"Train Avg loss: {loss:>8f} Train accuracy: {(accu*100):>0.1f}% \
    Test Avg loss: {test_loss:>8f} Test accuracy: {(100*test_accu):>0.1f}%\n")

showLossAndAccuracy(train_losses, train_accuracies, val_losses, val_accuracies, EPOCH, 'CNN')


preds = []
ix = 24300

for px in range(-6,6):
    img = (tr_images[ix]/255.0).squeeze()
    img = img.reshape([28, 28])
    img2 = np.roll(img, px, axis=1)
    img3 = (ms.Tensor(img2).view(-1,1,28,28)).astype(ms.float32)
    np_output = model(img3).asnumpy()
    pred = np.exp(np_output)/np.sum(np.exp(np_output))
    preds.append(pred)
    plt.subplot(2, 6, px + 7)
    plt.imshow((img2 * 255).astype(np.uint8), cmap=plt.cm.gray)
    plt.title(fmnist_classes[pred[0].argmax()])
plt.show()
print('np.array(preds).shape: ', np.array(preds).shape)

import seaborn as sns
fig, ax = plt.subplots(1,1, figsize=(12,10))
plt.title('Probability of each class for various translations')
sns.heatmap(np.array(preds).reshape(12,10), annot=True, ax=ax, fmt='.2f', xticklabels=fmnist_classes,
    yticklabels=[str(i)+str(' pixels') for i in range(-6,6)], cmap='gray')
plt.show()

exit(0)
