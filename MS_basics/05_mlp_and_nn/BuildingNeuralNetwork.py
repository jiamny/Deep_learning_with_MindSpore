import mindspore.nn as nn
import numpy as np
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, TimeMonitor
from mindvision.engine.callback import LossMonitor
from mindspore.train import Model
from mindvision.dataset import Mnist
from mindspore.nn.metrics import Accuracy
from mindspore import context

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

'''
LeNet-5 Model
'''

class LeNet5(nn.Cell):
    """
    LeNet-5 network structure
    """

    def __init__(self, num_class=10, num_channel=1):
        super(LeNet5, self).__init__()
        # Convolutional layer, where the number of input channels is num_channel,
        # the number of output channels is 6, and the convolutional kernel size is 5 x 5.
        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        # Convolutional layer, where the number of input channels is 6, the number of
        # output channels is 16, and the convolutional kernel size is 5 x 5.
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        # Fully-connected layer, where the number of inputs is 16 x 5 x 5 and the number of outputs is 120.
        self.fc1 = nn.Dense(16 * 5 * 5, 120)
        # Fully-connected layer, where the number of inputs is 120 and the number of outputs is 84.
        self.fc2 = nn.Dense(120, 84)
        # Fully-connected layer, where the number of inputs is 84 and the number of classes is num_class.
        self.fc3 = nn.Dense(84, num_class)
        # ReLU activation function
        self.relu = nn.ReLU()
        # Pooling layer
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        # Multidimensional arrays are flattened into one-dimensional arrays.
        self.flatten = nn.Flatten()

    def construct(self, x):
        # Use the defined operation to build a forward network.
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

#创建模型。模型包括5个卷积层和RELU激活函数，一个全连接输出层并使用softmax进行多分类，共分成（0-9）10类
class ForwardNN(nn.Cell):
    def __init__(self):
        super(ForwardNN, self).__init__()
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.fc1 = nn.Dense(784, 512, activation='relu')
        self.fc2 = nn.Dense(512, 256, activation='relu')
        self.fc3 = nn.Dense(256, 128, activation='relu')
        self.fc4 = nn.Dense(128, 64, activation='relu')
        self.fc5 = nn.Dense(64, 32, activation='relu')
        self.fc6 = nn.Dense(32, 10, activation='softmax')

    def construct(self, input_x):
        output = self.flatten(input_x)
        output = self.fc1(output)
        output = self.fc2(output)
        #print('fc2: ', output.shape)
        output = self.fc3(output)
        #print('fc3: ', output.shape)
        output = self.fc4(output)
        #print('fc4: ', output.shape)
        output = self.fc5(output)
        #print('fc5: ', output.shape)
        output = self.fc6(output)
        #print('fc6: ', output.shape)
        return output


net = LeNet5()
print(net)

input_x = Tensor(np.ones([1, 1, 32, 32]), mstype.float32)
print('net(input_x).shape: ', net(input_x).shape)

'''
Model Layers
'''

# The number of channels input is 1, the number of channels of output is 6,
# the convolutional kernel size is 5 x 5, and the parameters are initialized
# using the normal operator, and the pixels are not filled.
# nn.Conv2d
conv2d = nn.Conv2d(1, 6, 5, has_bias=False, weight_init='normal', pad_mode='same')
input_x = Tensor(np.ones([1, 1, 32, 32]), mstype.float32)

print(conv2d(input_x).shape)

# nn.ReLU
relu = nn.ReLU()

input_x = Tensor(np.array([-1, 2, -3, 2, -1]), mstype.float16)
output = relu(input_x)
print(output)

# nn.MaxPool2d
max_pool2d = nn.MaxPool2d(kernel_size=4, stride=4)
input_x = Tensor(np.ones([1, 6, 28, 28]), mstype.float32)

print(max_pool2d(input_x).shape)

# nn.Flatten
flatten = nn.Flatten()
input_x = Tensor(np.ones([1, 16, 5, 5]), mstype.float32)
output = flatten(input_x)

print(output.shape)

# nn.Dense
dense = nn.Dense(400, 120, weight_init='normal')
input_x = Tensor(np.ones([1, 400]), mstype.float32)
output = dense(input_x)

print(output.shape)

'''
Model Parameters
'''
for m in net.get_parameters():
    print(f"layer:{m.name}, shape:{m.shape}, dtype:{m.dtype}, requeires_grad:{m.requires_grad}")

'''
Quickly Building a LeNet-5 Model
'''
# `num_class` indicates the number of classes
net = LeNet5(num_class=10)

for m in net.get_parameters():
    print(f"layer:{m.name}, shape:{m.shape}, dtype:{m.dtype}, requeires_grad:{m.requires_grad}")

# Download and process the MNIST dataset.
download_train = Mnist(path="../data/mnist", split="train", batch_size=32, repeat_num=1,
                       shuffle=True, resize=32, download=False)
download_eval = Mnist(path="../data/mnist", split="test", batch_size=32, resize=32, download=False)

ds_eval = download_eval.run()
ds_train = download_train.run()

# Define the loss function.
net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

# Define the optimizer function.
net_opt = nn.Momentum(net.trainable_params(), learning_rate=0.01, momentum=0.9)

# Set the model saving parameters. The checkpoint steps are 1875.
config_ck = CheckpointConfig(save_checkpoint_steps=1875, keep_checkpoint_max=10)

# Apply the model saving parameters.
ckpoint = ModelCheckpoint(prefix="lenet_5", directory="../model/lenet5", config=config_ck)

# Initialize the model parameters.
model = Model(net, loss_fn=net_loss, optimizer=net_opt, metrics={"Accuracy": Accuracy()})

#训练模型
print("============== Starting Training ==============")
# Train the network model and save as lenet-1_1875.ckpt.
model.train(10, ds_train, callbacks=[ckpoint, LossMonitor(0.01, 1875)])

#使用测试集评估模型，打印总体准确率
metrics=model.eval(ds_eval)
loss_cb = LossMonitor(per_print_times=1)
time_cb = TimeMonitor(data_size=ds_train.get_dataset_size())
print(metrics)

exit(0)
