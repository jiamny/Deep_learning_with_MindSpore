import mindspore.nn as nn
from mindspore.train import Model

from mindvision.classification.dataset import Mnist
from mindvision.classification.models import lenet
from mindvision.engine.callback import LossMonitor

'''
Model Training
'''
epochs = 10  # Training epochs

# 1. Build a dataset.
download_train = Mnist(path="../data/mnist", split="train", batch_size=32, repeat_num=1, shuffle=True, resize=32,
                       download=False)
dataset_train = download_train.run()

# 2. Define a neural network.
network = lenet(num_classes=10, pretrained=False)
# 3.1 Define a loss function.
net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
# 3.2 Define an optimizer function.
net_opt = nn.Momentum(network.trainable_params(), learning_rate=0.01, momentum=0.9)
# 3.3 Initialize model parameters.
model = Model(network, loss_fn=net_loss, optimizer=net_opt, metrics={'accuracy'})

# 4. Train the neural network.
model.train(epochs, dataset_train, callbacks=[LossMonitor(0.01, 1875)])

'''
Saving the Model
'''
'''
# Directly Saving the Model
'''
import mindspore as ms

# The defined network model is net, which is used before or after training.
ms.save_checkpoint(network, "../model/MyNet.ckpt")

# Saving the Model During Training

from mindspore.train.callback import ModelCheckpoint, CheckpointConfig

'''
# Set the value of epoch_num.
'''
epoch_num = 5

# Set the model saving parameters.
config_ck = CheckpointConfig(save_checkpoint_steps=1875, keep_checkpoint_max=10)

# Apply the model saving parameters.
ckpoint = ModelCheckpoint(prefix="lenet", directory="../model/lenet", config=config_ck)
model.train(epoch_num, dataset_train, callbacks=[ckpoint])

'''
Loading the Model
'''
from mindspore import load_checkpoint, load_param_into_net

from mindvision.classification.dataset import Mnist
from mindvision.classification.models import lenet

# Save the model parameters to the parameter dictionary. The model parameters saved during the training are loaded.
param_dict = load_checkpoint("../model/lenet/lenet-5_1875.ckpt")

# Redefine a LeNet neural network.
net = lenet(num_classes=10, pretrained=False)

# Load parameters to the network.
load_param_into_net(net, param_dict)

# Redefine an optimizer function.
net_opt = nn.Momentum(net.trainable_params(), learning_rate=0.01, momentum=0.9)

model = Model(net, loss_fn=net_loss, optimizer=net_opt, metrics={"accuracy"})

'''
Validating the Model
'''
# Call eval() for inference.
download_eval = Mnist(path="../data/mnist", split="test", batch_size=32, resize=32, download=False)
dataset_eval = download_eval.run()
acc = model.eval(dataset_eval)

print("{}".format(acc))

'''
For Transfer Learning
'''
# Define a training dataset.
download_train = Mnist(path="../data/mnist", split="train", batch_size=32, repeat_num=1, shuffle=True, resize=32,
                       download=False)
dataset_train = download_train.run()

# Network model calls train() for training.
model.train(epoch_num, dataset_train, callbacks=[LossMonitor(0.01, 1875)])
