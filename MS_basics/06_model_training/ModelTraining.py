import numpy as np
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import nn
from mindvision.classification.models import lenet

'''
Hyperparameters
'''
epochs = 10
batch_size = 32
momentum = 0.9
learning_rate = 1e-2

'''
Loss Functions
'''
loss = nn.L1Loss()
output_data = Tensor(np.array([[1, 2, 3], [2, 3, 4]]).astype(np.float32))
target_data = Tensor(np.array([[0, 2, 5], [3, 1, 1]]).astype(np.float32))

print(loss(output_data, target_data))

'''
Optimizer Functions
'''
net = lenet(num_classes=10, pretrained=False)
optim = nn.Momentum(net.trainable_params(), learning_rate, momentum)

'''
Model Training

Model training consists of four steps:
1    Build a dataset.
2    Define a neural network.
3    Define hyperparameters, a loss function, and an optimizer.
4    Enter the epoch and dataset for training.
'''
import mindspore.nn as nn
from mindspore.train import Model

from mindvision.classification.dataset import Mnist
from mindvision.classification.models import lenet
from mindvision.engine.callback import LossMonitor

# 1. Build a dataset.
download_train = Mnist(path="../data/mnist", split="train", batch_size=batch_size, repeat_num=1, shuffle=True,
                       resize=32, download=False)
dataset_train = download_train.run()

# 2. Define a neural network.
network = lenet(num_classes=10, pretrained=False)
# 3.1 Define a loss function.
net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
# 3.2 Define an optimizer function.
net_opt = nn.Momentum(network.trainable_params(), learning_rate=learning_rate, momentum=momentum)
# 3.3 Initialize model parameters.
model = Model(network, loss_fn=net_loss, optimizer=net_opt, metrics={'acc'})

# 4. Train the neural network.
model.train(epochs, dataset_train, callbacks=[LossMonitor(learning_rate, 1875)])

exit(0)
