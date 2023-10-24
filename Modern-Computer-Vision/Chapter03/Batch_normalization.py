
import mindspore.dataset as ds
from mindspore.dataset import vision
from mindspore.dataset import transforms

import sys
sys.path.insert(0, '../utils')
from fashion_mnist import _FashionMNIST
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
import matplotlib.pyplot as plt

ms.set_seed(0)
ms.context.set_context(mode=ms.context.GRAPH_MODE,  device_target="GPU", device_id=0)

def displayImages(images, label):
    # 显示10张图片
    images = (images*255).asnumpy().astype(np.uint8)

    for i in range(images.shape[0]):
        image = images[i].reshape((28,28))
        #print('image: ', type(image), ' ', image.dtype, ' ', image.shape)

        if i >= 10:
            break
        plt.subplot(2, 5, i + 1)
        plt.imshow(image.squeeze(), cmap=plt.cm.gray)
        plt.title(label[i])
    plt.show()


data_folder = '/media/hhj/localssd/DL_data/fashion_MNIST'
tmnist = _FashionMNIST(data_folder, is_train=True)
tr_images, tr_targets = tmnist.get_data_and_label()

print(tr_images.dtype, ' ', tr_targets.dtype)

val_mnist = _FashionMNIST(data_folder, is_train=False)
val_images, val_targets = val_mnist.get_data_and_label()


BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCH = 20

def datapipe(dataset, batch_size):
    image_transforms = [
        # 基于给定的缩放和平移因子调整图像的像素大小。输出图像的像素大小为：output = image * rescale + shift。
        # 此处rescale取1.0 / 255.0，shift取0
        vision.Rescale(1.0 / 255.0, 0),
    ]

    label_transform = transforms.TypeCast(ms.int32)

    dataset = dataset.map(image_transforms, 'image')
    dataset = dataset.map(label_transform, 'label')

    # 将数据集中连续 batch_size 条数据组合为一个批数据
    dataset = dataset.batch(batch_size)
    return dataset


def showResults(train_losses, train_accuracies, val_losses, val_accuracies, EPOCH, val_dl):
    epochs = np.arange(EPOCH)+1

    plt.figure(figsize=(10, 12))
    plt.subplot(211)
    plt.plot(epochs, train_losses, 'bo', label='Training loss')
    plt.plot(epochs, val_losses, 'r', label='Validation loss')
    plt.title('Training and validation loss with very small input values')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid('off')

    plt.subplot(212)
    plt.plot(epochs, train_accuracies, 'bo', label='Training accuracy')
    plt.plot(epochs, val_accuracies, 'r', label='Validation accuracy')
    plt.title('Training and validation accuracy with very small input values')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    yl = [x for x in plt.gca().get_yticks()]
    plt.gca().set_yticks(yl)
    plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in yl])
    plt.legend()
    plt.grid('off')
    plt.show()

    x, _ = val_dl.create_tuple_iterator().__next__()
    plt.hist(model(x)[1].asnumpy().flatten())
    plt.title("Hidden activation layer values' distribution")
    plt.show()

    plt.figure(figsize=(16, 16))
    for ix, par in enumerate(model.trainable_params()):
        if(ix==0):
            plt.subplot(2, 2, ix + 1)
            plt.hist(par.asnumpy().flatten())
            plt.title('Weights conencting input to hidden layer')
        elif(ix ==1):
            plt.subplot(2, 2, ix + 1)
            plt.hist(par.asnumpy().flatten())
            plt.title('Biases of hidden layer')
        elif(ix==2):
            plt.subplot(2, 2, ix + 1)
            plt.hist(par.asnumpy().flatten())
            plt.title('Weights conencting hidden to output layer')
        elif(ix ==3):
            plt.subplot(2, 2, ix + 1)
            plt.hist(par.asnumpy().flatten())
            plt.title('Biases of output layer')
    plt.show()


class FMNISTDataset:
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __getitem__(self, ix):
        x, y = self.x[ix], self.y[ix]
        return x, y

    def __len__(self):
        return len(self.x)


train_dataset = ds.GeneratorDataset(FMNISTDataset(tr_images, tr_targets),
                             column_names=["image", "label"],
                             num_parallel_workers=1,
                             shuffle=True,
                             sampler=None)

val_dataset = ds.GeneratorDataset(FMNISTDataset(val_images, val_targets),
                                    column_names=["image", "label"],
                                    num_parallel_workers=1,
                                    shuffle=True,
                                    sampler=None)

trn_dl = datapipe(train_dataset, BATCH_SIZE)
val_dl = datapipe(val_dataset, len(val_dataset))

images, label = trn_dl.create_tuple_iterator().__next__()
displayImages(images, label)

# -----------------------------------------------------------------
# Without BatchNormalization
# -----------------------------------------------------------------


class Network(nn.Cell):
    def __init__(self):
        super().__init__()
        # nn.Flatten为输入展成平图层，即去掉那些空的维度
        self.flatten = nn.Flatten()
        self.input_to_hidden_layer = nn.Dense(784,1000)
        self.hidden_layer_activation = nn.ReLU()
        self.hidden_to_output_layer = nn.Dense(1000,10)

    def construct(self, x):
        x = self.flatten(x)
        x = self.input_to_hidden_layer(x)
        x1 = self.hidden_layer_activation(x)
        logits = self.hidden_to_output_layer(x1)
        # 返回模型
        return logits, x1


model = Network()

loss_fn = nn.CrossEntropyLoss()
optimizer = nn.Adam(model.trainable_params(), learning_rate=LEARNING_RATE)

def train(model, dataset, loss_fn, optimizer):

    def forward_fn(data, label):
        logits, _ = model(data)
        loss = loss_fn(logits, label)
        return loss, logits

    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    def train_step(data, label):
        (loss, _), grads = grad_fn(data, label)
        # 获得损失 depend用来处理操作间的依赖关系
        loss = ops.depend(loss, optimizer(grads))
        return loss

    #size = dataset.get_dataset_size()
    model.set_train()
    train_epoch_losses, train_epoch_accuracies = [], []
    for batch, (data, label) in enumerate(dataset.create_tuple_iterator()):

        # 批量训练获得损失值
        loss = train_step(data, label)
        train_epoch_losses.append(float(loss.item()))

    train_epoch_loss = np.array(train_epoch_losses).mean()
    return train_epoch_loss


def accuracy(model, dataset, loss_fn):
    num_batches = dataset.get_dataset_size()
    model.set_train(False)
    total, test_loss, correct = 0, 0, 0
    for data, label in dataset.create_tuple_iterator():
        pred, _ = model(data)
        total += len(data)
        test_loss += loss_fn(pred, label).asnumpy()
        correct += (pred.argmax(1) == label).asnumpy().sum()
    test_loss /= num_batches
    correct /= total
    return correct, test_loss


print('*'*40, ' Without BatchNormalization ', '*'*40)
train_losses, train_accuracies = [], []
val_losses, val_accuracies = [], []

for t in range(EPOCH):
    print(f"Epoch {(t+1):>4d}  ", '-'*80)
    loss = train(model, trn_dl, loss_fn, optimizer)
    train_losses.append(loss)
    accu, _ = accuracy(model, trn_dl, loss_fn)
    train_accuracies.append(accu)
    test_accu, test_loss = accuracy(model, val_dl, loss_fn)
    val_losses.append(test_loss)
    val_accuracies.append(test_accu)
    print(f"Train Avg loss: {loss:>8f} Train accuracy: {(accu*100):>0.1f}% \
    Test Avg loss: {test_loss:>8f} Test accuracy: {(100*test_accu):>0.1f}%\n")

showResults(train_losses, train_accuracies, val_losses, val_accuracies, EPOCH, val_dl)

# -----------------------------------------------------------------
# With BatchNormalization
# -----------------------------------------------------------------


class BatchNormNetwork(nn.Cell):
    def __init__(self):
        super().__init__()
        # nn.Flatten为输入展成平图层，即去掉那些空的维度
        self.flatten = nn.Flatten()
        self.input_to_hidden_layer = nn.Dense(784,1000)
        self.batch_norm = nn.BatchNorm1d(1000)
        self.hidden_layer_activation = nn.ReLU()
        self.hidden_to_output_layer = nn.Dense(1000,10)

    def construct(self, x):
        x = self.flatten(x)
        x = self.input_to_hidden_layer(x)
        x0 = self.batch_norm(x)
        x1 = self.hidden_layer_activation(x0)
        logits = self.hidden_to_output_layer(x1)
        # 返回模型
        return logits, x1


model = BatchNormNetwork()

loss_fn = nn.CrossEntropyLoss()
optimizer = nn.Adam(model.trainable_params(), learning_rate=LEARNING_RATE)

print('*'*40, ' With BatchNormalization ', '*'*40)
train_losses, train_accuracies = [], []
val_losses, val_accuracies = [], []

for t in range(EPOCH):
    print(f"Epoch {(t+1):>4d}  ", '-'*80)
    loss = train(model, trn_dl, loss_fn, optimizer)
    train_losses.append(loss)
    accu, _ = accuracy(model, trn_dl, loss_fn)
    train_accuracies.append(accu)
    test_accu, test_loss = accuracy(model, val_dl, loss_fn)
    val_losses.append(test_loss)
    val_accuracies.append(test_accu)
    print(f"Train Avg loss: {loss:>8f} Train accuracy: {(accu*100):>0.1f}% \
    Test Avg loss: {test_loss:>8f} Test accuracy: {(100*test_accu):>0.1f}%\n")


showResults(train_losses, train_accuracies, val_losses, val_accuracies, EPOCH, val_dl)

exit(0)




