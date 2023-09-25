
# 基于MindSpore构造激活函数

# 引入mindspore
import mindspore as ms
# 引入神经网络模块
from mindspore.nn import Cell
from mindspore import context, set_seed
set_seed(0)
context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

# 定义张量数据
tensor = ms.Tensor([[ 1.0,  2.0, -4.0,  1.3],
                    [-1.3,  2.0,  1.0, -6.0]], dtype=ms.float32)

# 模型构建
# 继承Cell类，构造ReLU函数
class My_Relu(Cell):
    def __init__(self):
        super(My_Relu, self).__init__()
    def construct(self, x):
        x[x<0] = 0
        return x

# 模型测试

# 实例化ReLU函数
my_relu = My_Relu()
# 输出
output = my_relu(tensor)
# 打印输出
print('output: ', output)
exit(0)
