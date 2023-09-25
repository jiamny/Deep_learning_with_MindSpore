
# 随机梯度下降

# 导入库
# 科学计算库
import numpy as np
# 绘图
import matplotlib.pyplot as plt
# 张量
from mindspore import Tensor
from mindspore import context, set_seed
set_seed(0)
context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

x = np.array([55, 71, 68, 87, 101, 87, 75, 78, 93, 73])
y = np.array([91, 101, 87, 109, 129, 98, 95, 101, 104, 93])
# 将数据转换为Tensor类
x = Tensor(x.astype(np.float32))
y = Tensor(y.astype(np.float32))

## 模型构建


lr = 0.0001    #学习率
num_iter = 100 #迭代次数


# 求解析解
# x：所有数据在第一个维度上的值
# y：所有数据在第二个维度上的值
def ols_algebra(x, y):
    n = len(x)
    # w1：一次项系数
    w1 = (n * sum(x * y) - sum(x) * sum(y)) / (n * sum(x * x) - sum(x) * sum(x))
    # w0：偏差
    w0 = (sum(x * x) * sum(y) - sum(x) * sum(x * y)) / (n * sum(x * x) - sum(x) * sum(x))
    
    return w1, w0


# x：所有数据在第一个维度上的值
# y：所有数据在第二个维度上的值
# lr：学习率
# num_iter:迭代次数
def ols_gradient_descent(x, y, lr, num_iter):
    w1 = 0
    w0 = 0
    N = x.size
    for j in range(num_iter):
        for i in range(N):
            y_hat = (w1 * x[i]) + w0
            w1_gradient = -2 * (x[i] * (y[i] - y_hat))
            w0_gradient = -2 * (y[i] - y_hat)
            # w1：一次项系数
            w1 -= lr * w1_gradient
            # w0：偏差
            w0 -= lr * w0_gradient
    return w1, w0

## 模型训练


# 解析解结果
w1, w0 = ols_algebra(x, y)
# 最小二乘结果
w1_, w0_ = ols_gradient_descent(x, y, lr = lr, num_iter = num_iter)


y_hat_Analyse_train = x * w1 + w0
print("解析解训练集预测结果：", y_hat_Analyse_train)
L_Analyse_train = (y - y_hat_Analyse_train) ** 2
print("解析解训练集预测平方误差：", L_Analyse_train)

y_hat_SGD_train = x * w1_ + w0_
print("随机梯度下降训练集预测结果：", y_hat_SGD_train)
L_SGD_train = (y - y_hat_SGD_train) ** 2
print("随机梯度下降训练集预测平方误差：", L_SGD_train)

## 模型预测


# 画图
# w1：解析解函数得出的一次项系数
# w0：解析解函数得出的偏差
# w1_：随机梯度下降算法得出的一次项系数
# w0_：随机梯度下降算法得出的偏差
# x：数据在第一个维度上的值
# y：数据在第二个维度上的值
# 要求画出两张图，第一张图上是数据散点图和y=w0*x+w1的图像，第二张图是数据散点图和y=w0_*x+w1_的图像
def plot_pic(w1, w0, w1_, w0_, x, y):
    fig, axes = plt.subplots(1, 2, figsize=(15,5))
    w1 = w1.asnumpy()
    w0 = w0.asnumpy()
    w1_ = w1_.asnumpy()
    w0_ = w0_.asnumpy()
    x = x.asnumpy()
    y = y.asnumpy()
    # 解析解的图
    axes[0].scatter(x, y)
    axes[0].plot(np.array([50, 110]), np.array([50, 110]) * w1 + w0, 'r')
    axes[0].set_title('OLS')
    # 随机梯度下降的图
    axes[1].scatter(x, y)
    axes[1].plot(np.array([50, 110]), np.array([50, 110]) * w1_ + w0_, 'r')
    axes[1].set_title('Gradient descent')

    plt.show()


print(w1)  # w1：解析解函数得出的一次项系数
print(w0)  # w0：解析解函数得出的偏差
print(w1_) # w1_：随机梯度下降算法得出的一次项系数
print(w0_) # w0_：随机梯度下降算法得出的偏差
plot_pic(w1, w0, w1_, w0_, x, y)# 绘出两种算法的结果的图像


#  测试数据
test_x = 92
test_y = 100
# 解析解预测值
y_hat_Analyse = w1 * test_x + w0
# 随机梯度下降预测值
y_hat_SGD = w1_ * test_x + w0_
# 解析解方法平方误差
L_Analyse = (y_hat_Analyse - test_y) ** 2
# 随机梯度下降平方误差
L_SGD = (y_hat_SGD - test_y) ** 2

print("对于（92，100）数据\n")
print("解析解方法预测值：",y_hat_Analyse," 其平方误差为：",L_Analyse,"\n")
print("随机梯度下降方法预测值：",y_hat_SGD," 其平方误差为：",L_SGD,"\n")
exit(0)