
# 使用MindSpore训练一个简单网络

# 从开放数据集中下载MNIST数据集
'''
from download import download

url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/" \
      "notebook/datasets/MNIST_Data.zip"
path = download(url, "./", kind="zip", replace=True)
'''
### 数据加载


# MindSpore库
import mindspore
# 神经网络模块
from mindspore import nn
# 常见算子操作
from mindspore import ops
# 图像增强模块
from mindspore.dataset import vision
# 通用数据增强
from mindspore.dataset import transforms
# 读取和解析Manifest数据文件构建数据集
from mindspore.dataset import MnistDataset

from mindspore import context, set_seed
set_seed(0)
context.set_context(mode=context.GRAPH_MODE, device_target="GPU")


BATCH_SIZE= 64       # batch的大小
LEARNING_RATE = 1e-2 # 学习率
EPOCH = 3            # 迭代次数


train_dataset = MnistDataset('/media/hhj/localssd/DL_data/mnist/MNIST_Data/train')
test_dataset = MnistDataset('/media/hhj/localssd/DL_data/mnist/MNIST_Data/test')


def datapipe(dataset, batch_size):
    image_transforms = [
        # 基于给定的缩放和平移因子调整图像的像素大小。输出图像的像素大小为：output = image * rescale + shift。
        # 此处rescale取1.0 / 255.0，shift取0
        vision.Rescale(1.0 / 255.0, 0),
        # 正则化 均值为0.1307，标准差为0.3081（查自官网）
        vision.Normalize(mean=(0.1307,), std=(0.3081,)),
        # 将输入图像的shape从 <H, W, C> 转换为 <C, H, W>
        vision.HWC2CHW()
    ]
    # 将输入的Tensor转换为指定的数据类型。
    label_transform = transforms.TypeCast(mindspore.int32)

    # map给定一组数据增强列表，按顺序将数据增强作用在数据集对象上。
    dataset = dataset.map(image_transforms, 'image')
    dataset = dataset.map(label_transform, 'label')
    # 将数据集中连续 batch_size 条数据组合为一个批数据
    dataset = dataset.batch(batch_size)
    return dataset
# 对数据集进行transfrom和batch
train_dataset = datapipe(train_dataset, BATCH_SIZE)
test_dataset = datapipe(test_dataset, BATCH_SIZE)

## 模型构建



# 定义模型
# MindSpore 中提供用户通过继承 nn.Cell 来方便用户创建和执行自己的网络
class Network(nn.Cell): 
    # 自定义的网络中，需要在__init__构造函数中申明各个层的定义
    def __init__(self): 
         # 继承父类nn.cell的__init__方法
        super().__init__()         
        # nn.Flatten为输入展成平图层，即去掉那些空的维度
        self.flatten = nn.Flatten()
        # 使用SequentialCell对网络进行管理
        self.dense_relu_sequential = nn.SequentialCell(
            # nn.Dense为致密连接层，它的第一个参数为输入层的维度，第二个参数为输出的维度，
            # 第三个参数为神经网络可训练参数W权重矩阵的初始化方式，默认为normal
            # nn.ReLU()非线性激活函数，它往往比论文中的sigmoid激活函数具有更好的效益
            nn.Dense(28 * 28, 512), # 致密连接层 输入28*28 输出512
            nn.ReLU(),              # ReLU层
            nn.Dense(512, 512),     # 致密连接层 输入512 输出512
            nn.ReLU(),              # ReLu层
            nn.Dense(512, 10)       # 致密连接层 输入512 输出10
        )
    # 在construct中实现层之间的连接关系，完成神经网络的前向构造
    def construct(self, x):
         #调用init中定义的self.flatten()方法 
        x = self.flatten(x)
        #调用init中的self.dense_relu_sequential()方法
        logits = self.dense_relu_sequential(x)
        # 返回模型
        return logits
model = Network()
print(model)

## 模型训练
# 实例化损失函数和优化器
# 计算预测值和目标值之间的交叉熵损失
loss_fn = nn.CrossEntropyLoss() 
#构建一个Optimizer对象，能够保持当前参数状态并基于计算得到的梯度进行参数更新 此处使用随机梯度下降算法
optimizer = nn.SGD(model.trainable_params(), learning_rate=LEARNING_RATE) 


def train(model, dataset, loss_fn, optimizer):
    # 定义 forward 函数
    def forward_fn(data, label):
        # 将数据载入模型
        logits = model(data)
        # 根据模型训练获取损失函数值
        loss = loss_fn(logits, label)
        return loss, logits
    # 调用梯度函数，value_and_grad()为生成求导函数，用于计算给定函数的正向计算结果和梯度
    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
    # 定义一步训练的函数
    def train_step(data, label):
        # 计算梯度，记录变量是怎么来的
        (loss, _), grads = grad_fn(data, label)
        # 获得损失 depend用来处理操作间的依赖关系
        loss = ops.depend(loss, optimizer(grads))
        return loss
    size = dataset.get_dataset_size()
    model.set_train()
    for batch, (data, label) in enumerate(dataset.create_tuple_iterator()):
        # 批量训练获得损失值
        loss = train_step(data, label)
        # 当完成所有数据样本的训练
        if batch % 100 == 0:
            loss, current = loss.asnumpy(), batch
            print(f"loss: {loss:>7f}  [{current:>3d}/{size:>3d}]")


def tt(model, dataset, loss_fn):
    num_batches = dataset.get_dataset_size()
    model.set_train(False)
    total, test_loss, correct = 0, 0, 0
    for data, label in dataset.create_tuple_iterator(): # 遍历所有测试样本数据
        pred = model(data)                              # 根据已训练模型获取预测值
        total += len(data)                              # 统计样本数
        test_loss += loss_fn(pred, label).asnumpy()     # 统计样本损失值
        correct += (pred.argmax(1) == label).asnumpy().sum()# 统计预测正确的样本个数
    test_loss /= num_batches                              # 求得平均损失
    correct /= total                                      # 计算accuracy
    print(f"Test: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


for t in range(EPOCH):
    print(f"Epoch {t+1}\n-------------------------------")
    train(model, train_dataset, loss_fn, optimizer)      # 训练模型
    tt(model, test_dataset, loss_fn)                   # 测试模型
print("Done!")

## 模型预测

###  保存模型

# 保存checkpoint时的配置策略
mindspore.save_checkpoint(model, "model.ckpt")
print("Saved Model to model.ckpt")

### 加载模型

# 实例化一个随机初始化的模型 
model = Network()
# 加载检查点，加载参数到模型 
param_dict = mindspore.load_checkpoint("model.ckpt")
param_not_load = mindspore.load_param_into_net(model, param_dict)
print(param_not_load)


model.set_train(False)
for data, label in test_dataset:
    pred = model(data)
    predicted = pred.argmax(1)
    print(f'Predicted: "{predicted[:10]}", Actual: "{label[:10]}"')
    break
exit(0)