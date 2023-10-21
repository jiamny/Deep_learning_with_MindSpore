
import mindspore
from mindspore import nn
from mindspore import ops
from mindspore.dataset import vision
from mindspore.dataset import transforms
from mindspore.dataset import FashionMnistDataset
import numpy as np

from mindspore import context, set_seed
set_seed(0)
context.set_context(mode=context.GRAPH_MODE, device_target="GPU", device_id=0)

BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCH = 20


trn_dataset = FashionMnistDataset(dataset_dir='/media/hhj/localssd/DL_data/fashion_MNIST/FashionMNIST/',
                                    usage='train',
                                    num_parallel_workers = 1,
                                    shuffle=True,
                                    num_samples=None)
tst_dataset = FashionMnistDataset(dataset_dir='/media/hhj/localssd/DL_data/fashion_MNIST/FashionMNIST/',
                            usage='test',
                            num_parallel_workers = 1,
                            shuffle=False,
                            num_samples=None)

# -----------------------------------------------------------------
# Without BatchNormalization
# -----------------------------------------------------------------

def datapipe(dataset, batch_size):
    image_transforms = [
        # 基于给定的缩放和平移因子调整图像的像素大小。输出图像的像素大小为：output = image * rescale + shift。
        # 此处rescale取1.0 / 255.0，shift取0
        vision.Rescale(1.0 / 255.0, 0),
        # 将输入图像的shape从 <H, W, C> 转换为 <C, H, W>
        vision.HWC2CHW()
    ]

    label_transform = transforms.TypeCast(mindspore.int32)

    dataset = dataset.map(image_transforms, 'image')
    dataset = dataset.map(label_transform, 'label')

    # 将数据集中连续 batch_size 条数据组合为一个批数据
    dataset = dataset.batch(batch_size)
    return dataset


train_dataset = datapipe(trn_dataset, BATCH_SIZE)
test_dataset = datapipe(tst_dataset, BATCH_SIZE)


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
print(model)

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

    model.set_train()
    train_epoch_losses, train_epoch_accuracies = [], []
    first = True
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


train_losses, train_accuracies = [], []
val_losses, val_accuracies = [], []

for t in range(EPOCH):
    print(f"Epoch {t+1} -------------------------------")
    loss = train(model, train_dataset, loss_fn, optimizer)
    train_losses.append(loss)
    accu, _ = accuracy(model, train_dataset, loss_fn)
    train_accuracies.append(accu)
    test_accu, test_loss = accuracy(model, test_dataset, loss_fn)
    val_losses.append(test_loss)
    val_accuracies.append(test_accu)
    print(f"Train Avg loss: {loss:>8f} Train accuracy: {(accu*100):>0.1f}% \
    Test Avg loss: {test_loss:>8f} Test accuracy: {(100*test_accu):>0.1f}%\n")


epochs = np.arange(EPOCH)+1

import matplotlib.pyplot as plt

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

exit(0)