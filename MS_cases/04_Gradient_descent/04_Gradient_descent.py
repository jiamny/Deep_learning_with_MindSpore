
#  基于MindSpore实现梯度下降算法

import numpy as np
import matplotlib.pyplot as plt
from mindspore import Tensor
from mindspore import context, set_seed
set_seed(0)
context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

# 定义数据生成函数
def data_generate():            
# 给出x的值
    x = np.array([55,71,68,87,101,87,75,78,93,73])    
# 给出y的值
    y = np.array([91,101,87,109,129,98,95,101,104,93])
# 将x转换为Tensor
    x = Tensor(x.astype(np.float32))        
# 将y转换为Tensor
    y = Tensor(y.astype(np.float32))                  
    return x,y

## 模型构建
### 步骤 1	导入Python库&模块并配置运行信息

# 导入numpy包用于随机生成数据
import numpy as np
# 导入绘图库
import matplotlib.pyplot as plt
# 从MindSpore中导入Tensor库
from mindspore import Tensor

### 定义梯度下降求解函数的参数变量
#### 采用张量实现梯度下降方法

#定义梯度下降求解函数
# lr: 学习率
# num_iter: 迭代次数
def ols_gradient_descent(x,y,lr,num_iter):
    return w1,w0
    pass
# 参考答案：
#定义梯度下降求解函数
# lr: 学习率
# num_iter: 迭代次数
def ols_gradient_descent(x,y,lr,num_iter):
    w1 = 0
    w0 = 0
    for i in range(num_iter):
        y_hat = (w1 * x)+ w0
        w1_gradient = -2 * sum(x*(y-y_hat))
        w0_gradient = -2 * sum(y-y_hat)
        w1 -=lr * w1_gradient
        w0 -= lr* w0_gradient
    return w1,w0

### 数据读取与处理

# 定义数据生成函数
def data_generate():            
# 给出x的值
    x = np.array([55,71,68,87,101,87,75,78,93,73])    
# 给出y的值
    y = np.array([91,101,87,109,129,98,95,101,104,93])
# 使用MindSpore中的Tensor库将x和y数组转换成Tensor
    x = Tensor(x.astype(np.float32))        
    y = Tensor(y.astype(np.float32))      
#返回x,y数组
    return x,y

### 5.4 采用张量求出解析解并构建解析解模型

#定义解析解函数：
def ols_algebra(x, y):
    #根据解析计算方法求解w，补充代码  
    return w1,w0 
    pass
#参考答案
#定义解析解函数：
def ols_algebra(x, y):
    #根据解析计算方法求解w
    n = len(x)
    w1 = (n * sum(x * y) - sum(x) * sum(y)) / (n * sum(x * x) - sum(x) * sum(x))
    w0 = (sum(x * x) * sum(y) - sum(x) * sum(x * y)) / (n * sum(x * x) - sum(x) * sum(x))
    return w1,w0
#通过填充后的代码查看结果
x,y = data_generate()
w1,w0 = ols_algebra(x,y)
print(w0)
print(w1)

### 构建绘图函数进行可视化

# 定义画图函数：
def plot_pic(w1,w0,w1_,w0_,x,y):
# 采用subplots绘制子图1*2个15x5大小的子图
    fig, axes = plt.subplots(1,2, figsize=(15,5))  
# 返回具有从该数组复制的值的 numpy.ndarray 对象
    w1 = w1.asnumpy()                              
    w0 = w0.asnumpy()
    w1_ = w1_.asnumpy()
    w0_ = w0_.asnumpy()
    x = x.asnumpy()
    y = y.asnumpy()
# 绘制y与x的散点图，并使用不同的标记大小或颜色
    axes[0].scatter(x,y)
#根据给出的x和y画出红线
    axes[0].plot(np.array([50,110]), np.array([50,110]) * w1 + w0, 'r') 
    axes[0].set_title("OLS")
    axes[1].scatter(x,y)
    axes[1].plot(np.array([50,110]), np.array([50,110]) * w1_ + w0_, 'r')
    axes[1].set_title("Gradient descent")
    plt.show()

## 模型测试

# 调用data_generate函数生成数据
x,y = data_generate()         
 # 调用ols_algebra函数计算解析解
w1,w0 = ols_algebra(x,y)      
print(w1)
print(w0)
# 调用ols_gradient_descent函数使用梯度下降方法求解w
w1_,w0_ = ols_gradient_descent(x,y,lr = 0.00001, num_iter = 500) 
print(w1_)
print(w0_)

plot_pic(w1,w0,w1_,w0_,x,y)
# 增加迭代次数后查看求解w
w1_,w0_ = ols_gradient_descent(x,y,lr = 0.00001, num_iter = 1500) 
print(w1_)
print(w0_)

plot_pic(w1,w0,w1_,w0_,x,y)

exit(0)

