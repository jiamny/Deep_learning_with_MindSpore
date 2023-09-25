
# 基于MindSpore实现Softmax

## **softmax**实现: $${softmax}(X)_{ij}=\frac{exp(X_{ij})}{\Sigma_{k}{exp(X_{ik})}}.$$
import mindspore.ops as ops
from mindspore import context, set_seed
set_seed(0)
context.set_context(mode=context.GRAPH_MODE, device_target="GPU")


#softmax函数定义
def softmax(X):
    X_exp = ops.exp(X)
    partition = X_exp.sum(1, keepdims=True)
    return X_exp / partition
import numpy as np
import mindspore
#生成样本点
X = mindspore.Tensor(np.random.normal(0, 1, (2, 5)), mindspore.float32)
X
#将样本点转换成概率值
X_prob = softmax(X)
X_prob
#每行总和为1
X_prob.sum(1)

## 5、数据处理
### 5.1 数据准备

# Download data from open datasets
'''
from download import download

url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/" \
      "notebook/datasets/MNIST_Data.zip"
path = download(url, "./", kind="zip", replace=True)
'''
### 数据加载


from mindspore.dataset import vision, transforms    #数据可视化
import mindspore as ms                              #导入mindspore框架
from mindspore.dataset import MnistDataset          # 导入Mnist数据集
def datapipe(path, batch_size):
    image_transforms = [                                 # 定义图像的预处理管道
        vision.Rescale(1.0 / 255.0, 0),                  # 将像素值归一化到0-1之间 
        vision.Normalize(mean=(0.1307,), std=(0.3081,)), # 图像数据标准化
        vision.HWC2CHW()                                 # 将图像的通道维度从HWC转换为CHW
    ]
    label_transform = transforms.TypeCast(ms.int32)      #定义标签的处理函数，把标签转换为整数

    dataset = MnistDataset(path)                         # 加载Mnist数据集
    dataset = dataset.map(image_transforms, 'image')     # 对图像进行处理
    dataset = dataset.map(label_transform, 'label')      # 对标签进行处理
    dataset = dataset.batch(batch_size)                  # 按照batch_size分批处理数据
    return dataset                                       # 返回最终的数据管道

train_dataset = datapipe('/media/hhj/localssd/DL_data/mnist/MNIST_Data/train', 64)         #获取训练集
test_dataset = datapipe('/media/hhj/localssd/DL_data/mnist/MNIST_Data/test', 64)           #获取测试集

## 模型构建

from mindspore import nn
from mindspore.common.initializer import Normal

#定义模型。 nn.Flatten将输入的X维度从[256,1,28,28]变成[256,784]
net = nn.SequentialCell([nn.Flatten(), nn.Dense(784, 10, weight_init=Normal(0.01, 0), bias_init='zero')])

#定义损失函数。SoftmaxCrossEntropyWithLogits，交叉熵损失函数中传递未规范化的预测，并同时计算softmax及其损失
loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

# 定义优化器
optimizer = nn.Momentum(net.trainable_params(),  learning_rate=0.1, momentum=0.9)


# 定义用于训练的train_loop函数。
def train_loop(model, dataset, loss_fn, optimizer):
    # 定义正向计算函数,接收数据和标签作为输入，返回损失值
    def forward_fn(data, label):
        logits = model(data)
        loss = loss_fn(logits, label)
        return loss

    # 定义微分函数，使用mindspore.value_and_grad获得微分函数grad_fn,输出loss和梯度。
    # 由于是对模型参数求导,grad_position 配置为None，传入可训练参数。
    grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters)

    # 定义 one-step training函数
    def train_step(data, label):
        loss, grads = grad_fn(data, label)
        optimizer(grads)
        return loss

    size = dataset.get_dataset_size()
    model.set_train()
    #计算 loss
    for batch, (data, label) in enumerate(dataset.create_tuple_iterator()):
        #print(type(data),type(label))
        loss = train_step(data, label)
        if batch % 100 == 0:
            loss, current = loss.asnumpy(), batch
            print(f"loss: {loss:>7f}  [{current:>3d}/{size:>3d}]")


# 定义用于测试的test_loop函数。
def tt_loop(model, dataset, loss_fn):
    num_batches = dataset.get_dataset_size()
    model.set_train(False)
    #计算Avg loss 和 Accuracy  
    total, test_loss, correct = 0, 0, 0
    for data, label in dataset.create_tuple_iterator():
        pred = model(data)
        total += len(data)
        test_loss += loss_fn(pred, label).asnumpy()
        correct += (pred.argmax(1) == label).asnumpy().sum()
    test_loss /= num_batches
    correct /= total
    print(f"Test: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

## 模型训练

from mindspore.train import ModelCheckpoint, CheckpointConfig # 将训练过程保存为检查点文件

#指定训练次数
epochs = 10

#调用训练和测试函数
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(net, train_dataset, loss, optimizer)          # 训练，打印 loss
    ms.save_checkpoint(net, "./save_direct.ckpt")            # 保存中间过程
    tt_loop(net, test_dataset, loss)                       # 测试，打印 Acc和 Avg loss
print("Done!")

## 模型预测

from mindspore import Model          # 定义深度学习模型
from matplotlib import pyplot as plt # 导入绘图库
import numpy as np                   # 用于数值计算的扩展库

# 将模型参数存入parameter的字典中，采用load_checkpoint接口加载模型参数
param_dict = ms.load_checkpoint("./save_direct.ckpt")

# 将参数加载到网络中
ms.load_param_into_net(net, param_dict)

#将net, loss, optimizer打包成一个Model
model = Model(net, loss, optimizer)

#迭代获取测试集图像和标签
data_test = test_dataset.create_dict_iterator()
data = next(data_test)
images = data["image"].asnumpy()
labels = data["label"].asnumpy()

# 使用函数model.predict预测image对应分类
output = model.predict(ms.Tensor(data['image']))
pred = np.argmax(output.asnumpy(), axis=1)

#可视化预测结果
plt.figure()
for i in range(1, 9):
    plt.subplot(2, 4, i)
    plt.imshow(images[i-1].squeeze(), cmap="gray")
    plt.title(pred[i-1])
plt.show()
exit(0)