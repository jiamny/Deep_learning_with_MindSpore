
# 基于MindSpore实现一维线性回归


import numpy as np
from mindspore import dataset as ds
from mindspore import context, set_seed
set_seed(0)
context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

# 自定义数据生成函数
def get_data(num,w=2,b=3):
    for data in range(num):
        # 在一定范围内随机生成自变量
        x=np.random.uniform(-10,10)
        # 生成随机噪声
        noise=np.random.normal(0,1)
        # 加入噪声之后得到因变量
        y=x*w+b+noise
        # 返回数据
        yield np.array([x]).astype(np.float32),np.array([y]).astype(np.float32)

# 展示部分数据
eval_data=list(get_data(6))
x_eval_label,y_eval_label=zip(*eval_data)
print(len(eval_data))


# 数据集生成函数
def create_dataset(num_data,batch_size=16,repeat_size=1):
    input_data=ds.GeneratorDataset(list(get_data(num_data)),column_names=['data','label'])
    # 设置数据批次
    input_data=input_data.batch(batch_size) 
    # 设置数据重复次数
    input_data=input_data.repeat(repeat_size) 
    return input_data

data_number=1600         # 数据数量
batch_number=16          # 数据批次
repeat_number=1          # 数据重复次数
# 生成数据集
ds_train=create_dataset(data_number,batch_number,repeat_number)
# 打印数据集批次
print('数据集批次：',ds_train.get_dataset_size())
# 创建数据集字典
dict_datasets=next(ds_train.create_dict_iterator())
# 打印数据集信息
print(dict_datasets.keys())
print('X:',dict_datasets['data'].shape)
print('y:',dict_datasets['label'].shape)


# 模型构建
## 导入python库
# 引入time模块
import time
# 引入numpy科学计算库
import numpy as np
# 引入绘图库
import matplotlib.pyplot as plt
# 引入MindSpore库
import mindspore 
# 神经网络模块
import mindspore.nn as nn
# 常见算子模块
import mindspore.ops as ops
# 张量,参数和参数元组
from mindspore import Tensor, ParameterTuple, Parameter 
# 数据类型
from mindspore import dtype as mstype
# 数据集
from mindspore import dataset as ds
# 引入Normal接口
from mindspore.common.initializer import Normal
# 引入Model模块
from mindspore.train import Model
# 引入LossMonitor模块
from mindspore.train import LossMonitor
# display模块
from IPython import display
# 引入回调函数
from mindspore.train import Callback
# 设置随机数生成种子 
np.random.seed(123)

## 构建一维线性回归模型
# 定义一个简单的一维线性回归模型
class LinearNet(nn.Cell):
    # 定义线性层
    def __init__(self):
        super(LinearNet,self).__init__()
        # 使用全连接层表示一维线性回归模型，初始化权重和偏置
        self.fc=nn.Dense(1,1,Normal(0.02),Normal(0.02),has_bias=True) 
    # 构造函数
    def construct(self,x):
        x=self.fc(x)
        return x

## 定义损失函数、优化器
# 一维线性回归模型实例化
net=LinearNet()
# 定义损失函数
net_loss=nn.loss.MSELoss()
# 传入模型的训练参数，以及学习率等参数
opt=nn.Momentum(net.trainable_params(),learning_rate=0.005,momentum=0.9)
# 定义模型
model=Model(net,net_loss,opt)
# 打印模型中的参数维度形状信息
model_params=net.trainable_params()
for param in model_params:
    print(param,param.asnumpy())

# 模型训练
## 可视化
# 定义函数画出模型拟合之直线以及真实的数据拟合直线的对比
def plot_model_and_datasets(net, eval_data):
    # 权重
    weight = net.trainable_params()[0]
    # 偏置
    bias = net.trainable_params()[1]
    x = np.arange(-10, 10, 0.1)
    y = x * Tensor(weight).asnumpy()[0][0] + Tensor(bias).asnumpy()[0]
    x1, y1 = zip(*eval_data)
    x_target = x
    y_target = x_target * 2 + 3
    # 绘制图像
    plt.axis([-11, 11, -20, 25])
    plt.scatter(x1, y1, color="red", s=5)
    plt.plot(x, y, color="blue")
    plt.plot(x_target, y_target, color="green")
    plt.show()


# 定义回调函数，训练过程中画出模型拟合的曲线
class ImageShowCallback(Callback):
    # 初始化
    def __init__(self, net, eval_data):
        self.net = net
        self.eval_data = eval_data

    # 绘制图像
    def step_end(self, run_context):
        cb_param = run_context.original_args()
        cur_step = cb_param.cur_step_num

        if cur_step % 10 == 0:
            plot_model_and_datasets(self.net, self.eval_data)
            # 清除打印内容，实现动态拟合效果
            display.clear_output(wait=True)


## 训练
# 训练回合数
epoch = 1
# 绘制训练过程中拟合直线图像
imageshow_cb = ImageShowCallback(net, eval_data)
# 开始训练
model.train(epoch, ds_train, callbacks=[imageshow_cb], dataset_sink_mode=False)

# 模型预测
# 传入线性回归模型和数据
plot_model_and_datasets(net, eval_data)
# 打印模型参数和训练结果
for net_param in net.trainable_params():
    print(net_param, net_param.asnumpy())

exit(0)